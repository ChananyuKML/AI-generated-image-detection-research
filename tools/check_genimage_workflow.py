"""Exercise real single-source training and seven-target evaluation on synthetic fixtures."""

import csv
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from aig.genimage_protocol import GENERATORS, PREFIX, PROTOCOL


def main():
    temporary_root = (ROOT / 'tmp' / 'genimage_checks').resolve()
    if not temporary_root.is_relative_to(ROOT / 'tmp'):
        raise ValueError('Synthetic check files must stay inside the project tmp folder.')
    temporary_root.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ, OMP_NUM_THREADS='2', MKL_NUM_THREADS='2', PYTHONUTF8='1')
    rng = np.random.default_rng(31)
    with tempfile.TemporaryDirectory(dir=temporary_root) as directory:
        temporary = Path(directory)
        data_root = temporary / 'data'
        for generator in GENERATORS:
            name = PREFIX + generator
            root = data_root / name
            rows = []
            for split in ('train', 'val', 'test'):
                for label, folder in enumerate(('nature', 'ai')):
                    for index in range(2):
                        relative = f'{split}/{folder}/{index}.png'
                        path = root / relative
                        path.parent.mkdir(parents=True, exist_ok=True)
                        pixels = rng.integers(0, 256, (32, 32, 3), dtype=np.uint8)
                        Image.fromarray(pixels).save(path)
                        rows.append({'relative_path': relative, 'label': label, 'split': split,
                                     'original_split': 'val' if split == 'test' else 'train',
                                     'pixel_sha256': hashlib.sha256(pixels.tobytes()).hexdigest()})
            with (root / 'manifest.csv').open('w', newline='', encoding='utf-8') as handle:
                writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)
            (root / 'manifest.json').write_text(json.dumps({'variant': name, 'evaluation_protocol': PROTOCOL,
                'label_map': {'nature': 0, 'ai': 1}, 'fixture': 'synthetic code check'}), encoding='utf-8')
        run = temporary / 'run'
        subprocess.run([sys.executable, str(ROOT / 'train.py'), '--model', 'fixed_filter_cnn',
                        '--dataset', PREFIX + 'midjourney', '--data-root', str(data_root), '--run-dir', str(run),
                        '--epochs', '1', '--batch-size', '2', '--device', 'cpu'], cwd=temporary, env=env, check=True)
        subprocess.run([sys.executable, str(ROOT / 'evaluate_genimage.py'), '--checkpoint', str(run / 'best.pt'),
                        '--data-root', str(data_root), '--batch-size', '2', '--device', 'cpu'],
                       cwd=temporary, env=env, check=True)
        report = json.loads((run / 'genimage_evaluation/matrix_row.json').read_text())
        assert len(report['results']) == 7
        assert {cell['testing_generator'] for cell in report['results']} == set(GENERATORS)
        assert all(cell['training_generator'] == 'midjourney' and cell['images'] == 4 for cell in report['results'])
        assert report['accuracy'] == sum(cell['accuracy'] for cell in report['results']) / 7
        assert len({cell['checkpoint_sha256'] for cell in report['results']}) == 1
        for target in GENERATORS:
            assert (run / 'genimage_evaluation' / target / 'predictions.csv').is_file()
        print('Single-source train -> seven held-out targets, consistent checkpoint, provenance, predictions and macro aggregation: OK', flush=True)


if __name__ == '__main__':
    main()
