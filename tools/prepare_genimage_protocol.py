"""Prepare the entire seven-generator Tiny release for held-out cross-generator testing."""

import argparse
from collections import Counter
import csv
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from aig.genimage_protocol import (GENERATORS, PREFIX, PROTOCOL, REFERENCE,
                                  assign_protocol_splits, audit_cross_generator_splits)
from aig.io import atomic_write_json
from tools.prepare_benchmarks import collapse_records, sample_balanced


def prepare(root=ROOT, seed=42):
    root = Path(root)
    archive_path = root / 'dataset/downloads/tiny_genimage.zip'
    metadata = json.loads(archive_path.with_suffix('.json').read_text(encoding='utf-8'))
    with archive_path.open('rb') as handle:
        digest = hashlib.file_digest(handle, 'sha256').hexdigest()
    if digest != metadata['sha256']:
        raise ValueError('Tiny GenImage archive checksum differs from the recorded source.')
    with (root / 'dataset/downloads/tiny_genimage.index.jsonl').open(encoding='utf-8') as handle:
        header = json.loads(next(handle))
        if header['archive_sha256'] != digest:
            raise ValueError('The image index belongs to another archive snapshot.')
        records = [json.loads(line) for line in handle]
    unique, rejected = collapse_records(records)
    assigned = assign_protocol_splits(unique, seed)
    exported = {PREFIX + name: sample_balanced([row for row in assigned if name in row['source_generators']])
                for name in GENERATORS}
    audit = audit_cross_generator_splits(exported)
    # Reuse original encoded images wherever possible; never re-encode the source.
    canonical = {}
    prepared = root / 'dataset/prepared'
    for variant in ['tiny_genimage', *['tiny_' + name for name in GENERATORS]]:
        provenance = prepared / variant / 'provenance.jsonl'
        if provenance.is_file():
            with provenance.open(encoding='utf-8') as handle:
                for line in handle:
                    row = json.loads(line)
                    canonical.setdefault(row['file_sha256'], prepared / variant / row['relative_path'])
    verified = set()
    reports = []
    with zipfile.ZipFile(archive_path) as archive:
        for name, rows in exported.items():
            destination = prepared / name
            old = None
            if (destination / 'manifest.json').is_file():
                old = json.loads((destination / 'manifest.json').read_text(encoding='utf-8'))
                if old.get('evaluation_protocol') != PROTOCOL or old.get('archive_sha256') != digest or old.get('seed') != seed:
                    raise ValueError(f'Existing protocol export has another configuration: {destination}')
            destination.mkdir(parents=True, exist_ok=True)
            exported_rows = []
            for row in rows:
                label = 'ai' if row['label'] else 'nature'
                relative = Path(row['split']) / label / (row['pixel_sha256'] + row['extension'])
                target = destination / relative
                target.parent.mkdir(parents=True, exist_ok=True)
                if not target.exists():
                    source = canonical.get(row['file_sha256'])
                    if source is not None and source.is_file():
                        if row['file_sha256'] not in verified:
                            with source.open('rb') as handle:
                                if hashlib.file_digest(handle, 'sha256').hexdigest() != row['file_sha256']:
                                    raise ValueError(f'Prepared source bytes changed: {source}')
                            verified.add(row['file_sha256'])
                        try:
                            os.link(source, target)
                        except OSError:
                            shutil.copyfile(source, target)
                    else:
                        payload = archive.read(row['source_member'])
                        if hashlib.sha256(payload).hexdigest() != row['file_sha256']:
                            raise ValueError('Source image checksum changed.')
                        target.write_bytes(payload)
                    canonical[row['file_sha256']] = target
                if target.stat().st_size != row['bytes']:
                    raise ValueError(f'Exported image size differs: {target}')
                exported_rows.append(dict(row, relative_path=relative.as_posix()))
            expected = {row['relative_path'] for row in exported_rows}
            actual = {path.relative_to(destination).as_posix() for path in destination.rglob('*')
                      if path.is_file() and path.suffix.lower() in ('.png', '.jpg', '.jpeg')}
            if expected != actual:
                raise ValueError(f'Unexpected image inventory: {destination}')
            fields = ['relative_path', 'label', 'split', 'generator', 'original_split', 'source_member',
                      'pixel_sha256', 'file_sha256', 'width', 'height', 'bytes']
            with (destination / 'manifest.csv').open('w', newline='', encoding='utf-8') as handle:
                writer = csv.DictWriter(handle, fieldnames=fields, extrasaction='ignore')
                writer.writeheader()
                writer.writerows(exported_rows)
            with (destination / 'provenance.jsonl').open('w', encoding='utf-8') as handle:
                for row in exported_rows:
                    handle.write(json.dumps(row) + '\n')
            report = {'version': 1, 'dataset': 'tiny_genimage', 'variant': name, 'seed': seed,
                      'evaluation_protocol': PROTOCOL, 'archive_sha256': digest, 'images': len(rows),
                      'counts': dict(sorted(Counter(f"{row['split']}/{'ai' if row['label'] else 'nature'}" for row in rows).items())),
                      'label_map': {'nature': 0, 'ai': 1}, 'source_reference': metadata['reference'],
                      'protocol_reference': REFERENCE, 'missing_official_generators': ['sdv4'],
                      'validation_fraction_from_source_train': .1, 'original_val_used_only_as_test': True,
                      'exact_decoded_pixel_overlap_across_splits': 0, 'cross_generator_global_split_audit': audit,
                      'near_duplicates_checked': False, 'image_bytes_preserved': True,
                      'source_stats': {'image_entries': len(records), 'unique_groups': len(unique), 'rejected': len(rejected)}}
            atomic_write_json(destination / 'manifest.json', report)
            reports.append(report)
            print(name, report['counts'], flush=True)
    atomic_write_json(prepared / 'genimage_protocol.summary.json', {'protocol': PROTOCOL, 'datasets': reports, 'audit': audit})
    return reports


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    prepare(seed=args.seed)
