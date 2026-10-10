"""Create reproducible, decoded-pixel-deduplicated binary benchmark folders.

Source image bytes are preserved. Manifests document all derived split decisions.
"""

import argparse
from collections import Counter, defaultdict
import concurrent.futures
import csv
import hashlib
import io
import json
import os
from pathlib import Path, PurePosixPath
import random
import shutil
import struct
import zipfile

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
VERSION = 1
SEED = 42
EXTENSIONS = (".jpg", ".jpeg", ".png")
GENERATOR_SUFFIXES = ("biggan", "vqdm", "sdv5", "wukong", "adm", "glide", "midjourney")


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".part")
    temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def describe_member(dataset, filename):
    parts = PurePosixPath(filename.replace("\\", "/")).parts
    lowered = [part.lower() for part in parts]
    split = next((value for value in lowered if value in ("train", "val", "validation", "test")), "unspecified")
    split = "val" if split == "validation" else split
    if dataset == "progan":
        split = "val"  # This downloader intentionally selects progan_val.zip.
    categories = {"real": 0, "fake": 1, "nature": 0, "ai": 1, "0_real": 0, "1_fake": 1}
    label_positions = [(index, categories[value]) for index, value in enumerate(lowered) if value in categories]
    if len(label_positions) != 1:
        raise ValueError(f"Cannot identify exactly one binary label in {filename}")
    label_index, label = label_positions[0]
    category = parts[label_index - 1] if dataset == "progan" and label_index > 0 else "unspecified"
    if dataset == "tiny_genimage":
        source = next((alias for alias in GENERATOR_SUFFIXES
                       if any(part.endswith(alias) for part in lowered[:label_index])), None)
        if source is None:
            raise ValueError(f"Unknown Tiny-GenImage source: {filename}")
    else:
        source = "sdv4" if dataset == "cifake" else "progan"
    return {"source_member": filename, "original_split": split,
            "label": label, "source_generator": source,
            "generator": source if label else "real", "category": category}


def scan_archive(dataset, archive_path, workers):
    index_path = ROOT / "dataset" / "downloads" / f"{dataset}.index.jsonl"
    source_report = json.loads(archive_path.with_suffix(".json").read_text(encoding="utf-8"))
    with archive_path.open("rb") as handle:
        actual_digest = hashlib.file_digest(handle, "sha256").hexdigest()
    if actual_digest != source_report["sha256"]:
        raise ValueError(f"Archive checksum differs from its metadata: {archive_path}. Run the downloader to verify the source.")
    header = {"version": VERSION, "archive_sha256": source_report["sha256"]}
    if index_path.exists():
        with index_path.open(encoding="utf-8") as handle:
            if json.loads(next(handle)) == header:
                records = [json.loads(line) for line in handle]
                print(f"{dataset}: using verified image index ({len(records)} records)", flush=True)
                return records, source_report
    records = []
    with zipfile.ZipFile(archive_path) as archive:
        members = [info for info in archive.infolist()
                   if info.filename.lower().endswith(EXTENSIONS)
                   and "__MACOSX/" not in info.filename
                   and not PurePosixPath(info.filename).name.startswith("._")]

        def inspect(info):
            record = describe_member(dataset, info.filename)
            try:
                payload = archive.read(info)  # ZIP CRC is checked during this read.
                with Image.open(io.BytesIO(payload)) as image:
                    rgb = image.convert("RGB")
                    rgb.load()
                    width, height = rgb.size
                    digest = hashlib.sha256(struct.pack("<II", width, height) + rgb.tobytes()).hexdigest()
                    file_format = image.format
                extension = {"PNG": ".png", "JPEG": ".jpg"}.get(file_format)
                if extension is None:
                    raise ValueError(f"Unsupported stored image format: {file_format}")
                record.update(pixel_sha256=digest, file_sha256=hashlib.sha256(payload).hexdigest(),
                              bytes=len(payload), width=width, height=height, extension=extension)
            except Exception as error:
                record["error"] = type(error).__name__ + ": " + str(error)
            return record

        # A bounded future window limits memory use even for CIFAKE's 120k files.
        with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
            iterator = iter(members)
            pending = set()
            for _ in range(workers * 4):
                info = next(iterator, None)
                if info is not None:
                    pending.add(executor.submit(inspect, info))
            while pending:
                finished, pending = concurrent.futures.wait(pending, return_when=concurrent.futures.FIRST_COMPLETED)
                for future in finished:
                    records.append(future.result())
                    info = next(iterator, None)
                    if info is not None:
                        pending.add(executor.submit(inspect, info))
                if len(records) % 2000 < len(finished):
                    print(f"{dataset}: validated {len(records)} / {len(members)} images", flush=True)
    records.sort(key=lambda record: record["source_member"])
    temporary = index_path.with_suffix(".jsonl.part")
    with temporary.open("w", encoding="utf-8") as handle:
        handle.write(json.dumps(header) + "\n")
        for record in records:
            handle.write(json.dumps(record) + "\n")
    temporary.replace(index_path)
    return records, source_report


def collapse_records(records):
    grouped = defaultdict(list)
    rejected = [record for record in records if "error" in record]
    for record in records:
        if "error" not in record:
            grouped[record["pixel_sha256"]].append(record)
    unique = []
    priority = {"test": 0, "val": 1, "train": 2, "unspecified": 3}
    for digest, members in sorted(grouped.items()):
        if len({member["label"] for member in members}) != 1:
            rejected.extend(dict(member, error="Identical decoded image has conflicting labels") for member in members)
            continue
        if members[0]["label"] and len({member["source_generator"] for member in members}) > 1:
            rejected.extend(dict(member, error="Identical fake has ambiguous generator provenance") for member in members)
            continue
        ordered = sorted(members, key=lambda member: (priority[member["original_split"]], member["source_member"]))
        canonical = dict(ordered[0])
        canonical["related_members"] = [member["source_member"] for member in ordered]
        canonical["source_generators"] = sorted({member["source_generator"] for member in ordered})
        canonical["original_splits"] = sorted({member["original_split"] for member in ordered})
        unique.append(canonical)
    return unique, rejected


def stable_shuffle(records, key):
    ordered = sorted(records, key=lambda record: record["pixel_sha256"])
    seed = int.from_bytes(hashlib.sha256(f"{SEED}:{key}".encode()).digest()[:8], "little")
    random.Random(seed).shuffle(ordered)
    return ordered


def assign_splits(dataset, records):
    strata = defaultdict(list)
    for record in records:
        if dataset == "progan":
            stratum = (record["label"], record["category"])
        else:
            stratum = (record["label"], record["generator"], record["original_split"])
        strata[stratum].append(record)
    assigned = []
    for stratum, members in sorted(strata.items(), key=lambda item: str(item[0])):
        members = stable_shuffle(members, (dataset, stratum))
        count = len(members)
        if dataset == "progan":
            train_count, val_count = int(count * 0.7), int(count * 0.15)
            splits = ["train"] * train_count + ["val"] * val_count + ["test"] * (count - train_count - val_count)
        elif dataset == "cifake":
            if members[0]["original_split"] == "test":
                splits = ["test"] * count
            else:
                val_count = int(count * 0.1)
                splits = ["val"] * val_count + ["train"] * (count - val_count)
        else:
            if members[0]["original_split"] == "train":
                splits = ["train"] * count
            elif members[0]["original_split"] == "val":
                val_count = count // 2
                splits = ["val"] * val_count + ["test"] * (count - val_count)
            elif members[0]["original_split"] == "test":
                splits = ["test"] * count
            else:
                raise ValueError("Tiny-GenImage member has no supplied split")
        assigned.extend(dict(record, split=split) for record, split in zip(members, splits))
    return assigned


def sample_balanced(records, per_class_caps=None):
    selected = []
    for split in ("train", "val", "test"):
        by_label = {label: [record for record in records if record["split"] == split and record["label"] == label]
                    for label in (0, 1)}
        quota = min(len(by_label[0]), len(by_label[1]))
        if per_class_caps:
            quota = min(quota, per_class_caps[split])
        if quota < 1:
            raise ValueError(f"No balanced examples available in {split}")
        for label, members in by_label.items():
            # Round-robin strata keep available generators/categories represented.
            strata = defaultdict(list)
            for member in members:
                strata[(member["generator"], member["category"])].append(member)
            queues = [stable_shuffle(group, (split, label, key)) for key, group in sorted(strata.items())]
            chosen = []
            while len(chosen) < quota:
                for queue in queues:
                    if queue and len(chosen) < quota:
                        chosen.append(queue.pop())
            selected.extend(chosen)
    return sorted(selected, key=lambda record: (record["split"], record["label"], record["pixel_sha256"]))


def export_variant(dataset, variant, records, archive_path, source_report, stats, workers, canonical_files):
    destination = ROOT / "dataset" / "prepared" / variant
    destination.mkdir(parents=True, exist_ok=True)
    for split in ("train", "val", "test"):
        for label_folder in ("ai", "nature"):
            (destination / split / label_folder).mkdir(parents=True, exist_ok=True)
    manifest_path = destination / "manifest.json"
    if manifest_path.exists():
        previous = json.loads(manifest_path.read_text(encoding="utf-8"))
        if previous["version"] != VERSION or previous["archive_sha256"] != source_report["sha256"]:
            raise RuntimeError(f"Existing prepared dataset has a different configuration: {destination}")
    manifest_rows = []
    with zipfile.ZipFile(archive_path) as archive:
        def export(record):
            folder = "ai" if record["label"] else "nature"
            relative = Path(record["split"]) / folder / (record["pixel_sha256"] + record["extension"])
            target = destination / relative
            if target.exists():
                if target.stat().st_size != record["bytes"]:
                    raise RuntimeError(f"Existing image has an unexpected size: {target}")
            elif record["pixel_sha256"] in canonical_files:
                original = canonical_files[record["pixel_sha256"]]
                try:
                    os.link(original, target)
                except OSError:
                    shutil.copyfile(original, target)
            else:
                payload = archive.read(record["source_member"])
                if hashlib.sha256(payload).hexdigest() != record["file_sha256"]:
                    raise RuntimeError("Source image checksum changed")
                temporary = target.with_suffix(target.suffix + ".part")
                temporary.write_bytes(payload)
                temporary.replace(target)
            return record["pixel_sha256"], target, dict(record, relative_path=relative.as_posix())

        with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
            for index, (digest, target, row) in enumerate(executor.map(export, records), 1):
                canonical_files.setdefault(digest, target)
                manifest_rows.append(row)
                if index % 5000 == 0:
                    print(f"{variant}: exported {index} / {len(records)} images", flush=True)
    fields = ["relative_path", "label", "split", "generator", "category", "original_split", "source_member",
              "pixel_sha256", "file_sha256", "width", "height", "bytes"]
    with (destination / "manifest.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(manifest_rows)
    with (destination / "provenance.jsonl").open("w", encoding="utf-8") as handle:
        for row in manifest_rows:
            handle.write(json.dumps(row) + "\n")
    expected = {row["relative_path"] for row in manifest_rows}
    actual = {path.relative_to(destination).as_posix() for path in destination.rglob("*")
              if path.is_file() and path.suffix.lower() in EXTENSIONS}
    if actual != expected:
        raise RuntimeError(f"Unexpected image files in {destination}")
    counts = Counter(f"{row['split']}/{'ai' if row['label'] else 'nature'}" for row in manifest_rows)
    split_hashes = {split: {row["pixel_sha256"] for row in manifest_rows if row["split"] == split}
                    for split in ("train", "val", "test")}
    assert not (split_hashes["train"] & split_hashes["val"] or split_hashes["train"] & split_hashes["test"]
                or split_hashes["val"] & split_hashes["test"])
    protocols = {
        "cifake": "Original test images remain held out; 10% of deduplicated original training groups form validation.",
        "progan": "Custom 70/15/15 split of the published ProGAN validation archive, stratified by binary label and LSUN category. Pilot protocol, not the original benchmark training protocol.",
        "tiny_genimage": "Community Tiny-GenImage original training groups retained; supplied validation groups split 50/50 into validation/test. Exact duplicate groups are kept entirely in the higher-priority held-out source split. SD1.4 is absent.",
    }
    report = {"version": VERSION, "dataset": dataset, "variant": variant, "seed": SEED,
              "archive_sha256": source_report["sha256"], "source_reference": source_report["reference"],
              "protocol": protocols[dataset], "label_map": {"nature": 0, "ai": 1},
              "counts": dict(sorted(counts.items())), "images": len(records),
              "generator_counts": dict(Counter(row["generator"] for row in manifest_rows)),
              "native_resolutions": dict(Counter(f"{row['width']}x{row['height']}" for row in manifest_rows)),
              "source_stats": stats, "image_bytes_preserved": True,
              "exact_decoded_pixel_overlap_across_splits": 0,
              "near_duplicates_checked": False,
              "training_input_size": 224,
              "notes": ["CIFAKE is native 32px and is upscaled by the existing training transform; this is a low-resolution experiment."] if dataset == "cifake" else []}
    write_json(manifest_path, report)
    print(f"{variant}: READY {dict(sorted(counts.items()))}", flush=True)
    return report


def prepare(dataset, workers):
    archive_path = ROOT / "dataset" / "downloads" / f"{dataset}.zip"
    records, source_report = scan_archive(dataset, archive_path, workers)
    unique, rejected = collapse_records(records)
    assigned = assign_splits(dataset, unique)
    stats = {"archive_image_entries": len(records), "decoded_unique_groups": len(unique),
             "excluded_records": len(rejected), "deduplicated_records": len(records) - len(unique) - len(rejected)}
    write_json(ROOT / "dataset" / "downloads" / f"{dataset}.excluded.json", rejected)
    canonical_files = {}
    variants = []
    if dataset == "cifake":
        variants = [("cifake", sample_balanced(assigned)),
                    ("cifake_poc", sample_balanced(assigned, {"train": 4000, "val": 1000, "test": 1000}))]
    elif dataset == "progan":
        variants = [("progan", sample_balanced(assigned)),
                    ("progan_poc", sample_balanced(assigned, {"train": 1400, "val": 300, "test": 300}))]
    else:
        variants = [("tiny_genimage", sample_balanced(assigned)),
                    ("tiny_genimage_poc", sample_balanced(assigned, {"train": 1000, "val": 250, "test": 250}))]
        for generator in GENERATOR_SUFFIXES:
            subset = [record for record in assigned if generator in record["source_generators"]]
            variants.append(("tiny_" + generator, sample_balanced(subset)))
    reports = [export_variant(dataset, variant, selected, archive_path, source_report, stats, workers, canonical_files)
               for variant, selected in variants]
    write_json(ROOT / "dataset" / "prepared" / f"{dataset}.summary.json", reports)
    return reports


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="+", choices=["cifake", "progan", "tiny_genimage"],
                        default=["cifake", "progan", "tiny_genimage"])
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("--workers must be positive")
    for name in args.datasets:
        prepare(name, args.workers)
    references = ROOT / "configs" / "dataset_sources.json"
    if references.is_file():
        write_json(ROOT / "dataset" / "prepared" / "source_references.json",
                   json.loads(references.read_text(encoding="utf-8")))
    print("All requested datasets are train-ready.", flush=True)
