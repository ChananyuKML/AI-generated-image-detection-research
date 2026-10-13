"""Seven-generator Tiny GenImage adaptation of GenImage's cross-generator task."""

from collections import defaultdict
import csv
import hashlib
import json
import math
from pathlib import Path
import random

GENERATORS = ('midjourney', 'sdv5', 'adm', 'glide', 'wukong', 'vqdm', 'biggan')
PREFIX = 'tiny_protocol_'
DATASETS = tuple(PREFIX + name for name in GENERATORS)
PROTOCOL = 'tiny_genimage_cross_generator_v1'
REFERENCE = 'https://github.com/GenImage-Dataset/GenImage#benchmark'


def assign_protocol_splits(records, seed=42, validation_fraction=.1):
    """All supplied val images are test; selection val comes only from source train."""
    if not 0 < validation_fraction < 1:
        raise ValueError('Validation fraction must be between zero and one.')
    assigned, train_strata = [], defaultdict(list)
    for row in records:
        splits = set(row.get('original_splits', [row['original_split']]))
        copied = dict(row)
        if splits & {'val', 'test'}:
            copied['split'] = 'test'
            assigned.append(copied)
        elif splits == {'train'}:
            # Canonical duplicate groups are assigned globally, including shared real images.
            key = (row['label'], row['generator'])
            train_strata[key].append(copied)
        else:
            raise ValueError('Every Tiny GenImage image needs a recognized supplied split.')
    for key, rows in sorted(train_strata.items()):
        rows.sort(key=lambda row: row['pixel_sha256'])
        derived_seed = int.from_bytes(hashlib.sha256(f'{PROTOCOL}:{seed}:{key}'.encode()).digest()[:8], 'little')
        random.Random(derived_seed).shuffle(rows)
        count = int(len(rows) * validation_fraction)
        if count < 1 or count >= len(rows):
            raise ValueError('Not enough training groups to reserve checkpoint-selection validation.')
        for index, row in enumerate(rows):
            row['split'] = 'val' if index < count else 'train'
            assigned.append(row)
    return sorted(assigned, key=lambda row: row['pixel_sha256'])


def audit_cross_generator_splits(datasets):
    """No decoded image identity can train/validate a detector and occur in any test."""
    identities = {split: set() for split in ('train', 'val', 'test')}
    for rows in datasets.values():
        seen = set()
        for row in rows:
            if row['pixel_sha256'] in seen:
                raise ValueError('Duplicate image identities inside a generator export.')
            seen.add(row['pixel_sha256'])
            identities[row['split']].add(row['pixel_sha256'])
            if row['split'] == 'test' and row['original_split'] not in ('val', 'test'):
                raise ValueError('Final test images must come from supplied held-out splits.')
            if row['split'] != 'test' and row['original_split'] != 'train':
                raise ValueError('Checkpoint selection must not consume supplied testing images.')
    if identities['train'] & identities['val'] or (identities['train'] | identities['val']) & identities['test']:
        raise ValueError('Cross-generator image identity leakage between split roles.')
    return {split: len(values) for split, values in identities.items()}


def read_manifest(root):
    root = Path(root)
    report = json.loads((root / 'manifest.json').read_text(encoding='utf-8'))
    if report.get('evaluation_protocol') != PROTOCOL:
        raise ValueError(f'{root} is not the GenImage-style prepared protocol.')
    if report.get('label_map') != {'nature': 0, 'ai': 1}:
        raise ValueError('Protocol labels differ from the binary research convention.')
    with (root / 'manifest.csv').open(newline='', encoding='utf-8') as handle:
        rows = list(csv.DictReader(handle))
    return report, rows


def audit_manifest_roots(data_root):
    rows_by_source, signatures = {}, {}
    for name in DATASETS:
        root = Path(data_root) / name
        report, rows = read_manifest(root)
        if report.get('variant') != name:
            raise ValueError('Protocol dataset name differs from its manifest.')
        rows_by_source[name] = rows
        signatures[name] = {'kind': 'manifest_sha256', 'sha256': hashlib.sha256((root / 'manifest.csv').read_bytes()).hexdigest()}
    counts = audit_cross_generator_splits(rows_by_source)
    return {'protocol': PROTOCOL, 'signatures': signatures, 'unique_split_images': counts,
            'missing_official_generators': ['sdv4'], 'near_duplicates_checked': False}


def matrix_summary(cells, generators=GENERATORS):
    """Equal cell weighting, plus explicit diagonal and unseen-generator summaries."""
    lookup = {}
    for cell in cells:
        key = (cell['training_generator'], cell['testing_generator'])
        if key[0] not in generators or key[1] not in generators or key in lookup:
            raise ValueError('Unknown or duplicate cross-generator matrix cell.')
        if not math.isfinite(cell['accuracy']) or not 0 <= cell['accuracy'] <= 1:
            raise ValueError('Matrix accuracy must be finite and between zero and one.')
        lookup[key] = cell
    def mean(keys):
        if not keys or not all(key in lookup for key in keys):
            return None
        return sum(lookup[key]['accuracy'] for key in keys) / len(keys)
    all_keys = [(source, target) for source in generators for target in generators]
    return {'generators': list(generators), 'completed_cells': len(lookup),
            'expected_cells': len(all_keys), 'complete': len(lookup) == len(all_keys),
            'accuracy': [[lookup.get((source, target), {}).get('accuracy') for target in generators] for source in generators],
            'roc_auc': [[lookup.get((source, target), {}).get('roc_auc') for target in generators] for source in generators],
            'row_means': {source: mean([(source, target) for target in generators]) for source in generators},
            'column_means': {target: mean([(source, target) for source in generators]) for target in generators},
            'macro_accuracy': mean(all_keys),
            'diagonal_accuracy': mean([(name, name) for name in generators]),
            'off_diagonal_accuracy': mean([(source, target) for source, target in all_keys if source != target]),
            'observed_cell_mean': sum(cell['accuracy'] for cell in lookup.values()) / len(lookup) if lookup else None}
