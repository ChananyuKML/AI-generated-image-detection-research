"""Download the three public benchmark archives without account credentials."""

import argparse
import concurrent.futures
import hashlib
import json
from pathlib import Path
import time
import urllib.request
import zipfile


ROOT = Path(__file__).resolve().parents[1]
SOURCES = {
    "cifake": {
        "url": "https://www.kaggle.com/api/v1/datasets/download/birdy654/cifake-real-and-ai-generated-synthetic-images",
        "bytes": 109625224,
        "sha256": "1839206c1462371f5c885eda1660b448ea7abf52ee81fcc71e5e4651a4a11389",
        "reference": "https://github.com/jordan-bird/CIFAKE-Real-and-AI-Generated-Synthetic-Images",
    },
    "progan": {
        "url": "https://huggingface.co/datasets/sywang/CNNDetection/resolve/main/progan_val.zip?download=true",
        "bytes": 830792545,
        "sha256": "7bce9385bde5407875bb829e321f812db53587e59611e105caf6175cf323f002",
        "reference": "https://peterwang512.github.io/CNNDetection/",
    },
    "tiny_genimage": {
        "url": "https://www.kaggle.com/api/v1/datasets/download/yangsangtai/tiny-genimage",
        "bytes": 8345888545,
        "sha256": "49bc9ce3203d62485c3f50c58eddec324fab77693c0f2aa156e162cf5eb1d08f",
        "reference": "https://www.kaggle.com/datasets/yangsangtai/tiny-genimage",
    },
}


def digest_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def download(name):
    directory = ROOT / "dataset" / "downloads"
    directory.mkdir(parents=True, exist_ok=True)
    source = SOURCES[name]
    target = directory / f"{name}.zip"
    partial = directory / f"{name}.zip.part"
    if target.exists() and target.stat().st_size == source["bytes"] and zipfile.is_zipfile(target):
        print(f"{name}: using complete cached archive", flush=True)
    else:
        if target.exists():
            raise RuntimeError(f"Existing archive has unexpected size/format: {target}")
        for attempt in range(1, 6):
            offset = partial.stat().st_size if partial.exists() else 0
            if offset == source["bytes"] and zipfile.is_zipfile(partial):
                partial.replace(target)
                break
            headers = {"User-Agent": "aig-dec-benchmark-preparation/1.0"}
            if offset:
                headers["Range"] = f"bytes={offset}-"
            request = urllib.request.Request(source["url"], headers=headers)
            try:
                with urllib.request.urlopen(request, timeout=90) as response:
                    if offset and response.status != 206:
                        offset = 0
                    if response.status == 206:
                        content_range = response.headers.get("Content-Range", "")
                        if not content_range.startswith(f"bytes {offset}-"):
                            raise RuntimeError(f"Unexpected Content-Range: {content_range}")
                    mode = "ab" if offset else "wb"
                    downloaded = offset
                    next_report = downloaded + 128 * 1024 * 1024
                    started = time.monotonic()
                    with partial.open(mode) as handle:
                        while block := response.read(4 * 1024 * 1024):
                            handle.write(block)
                            downloaded += len(block)
                            if downloaded >= next_report:
                                print(f"{name}: {downloaded / 1024**2:.0f} / {source['bytes'] / 1024**2:.0f} MiB", flush=True)
                                next_report = downloaded + 128 * 1024 * 1024
                if downloaded != source["bytes"]:
                    raise RuntimeError(f"Incomplete download: {downloaded} / {source['bytes']} bytes")
                if not zipfile.is_zipfile(partial):
                    raise RuntimeError("The response is not a valid ZIP archive")
                partial.replace(target)
                print(f"{name}: download complete in {time.monotonic() - started:.1f}s", flush=True)
                break
            except Exception as error:
                print(f"{name}: attempt {attempt} failed: {error}", flush=True)
                if attempt == 5:
                    raise
                time.sleep(min(5 * attempt, 20))
    with zipfile.ZipFile(target) as archive:
        images = [entry.filename for entry in archive.infolist()
                  if entry.filename.lower().endswith((".jpg", ".jpeg", ".png"))
                  and "__MACOSX/" not in entry.filename and not Path(entry.filename).name.startswith("._")]
    actual_digest = digest_file(target)
    if actual_digest != source["sha256"]:
        raise RuntimeError(f"Source snapshot changed or archive is corrupted: {target}. Expected SHA-256 {source['sha256']}, got {actual_digest}. Preserve the file and review the source before changing the snapshot.")
    report = dict(source, archive=target.relative_to(ROOT).as_posix(),
                  sha256=actual_digest, image_entries=len(images),
                  first_image_paths=images[:8])
    (directory / f"{name}.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"{name}: {len(images)} image entries; SHA-256 recorded; examples={images[:3]}", flush=True)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="+", choices=list(SOURCES), default=list(SOURCES))
    args = parser.parse_args()
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
        futures = {executor.submit(download, name): name for name in args.datasets}
        for future in concurrent.futures.as_completed(futures):
            future.result()
    print("All requested downloads verified.", flush=True)
