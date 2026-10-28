"""Export presentation/paper accuracy tables; optionally follow the live batch."""

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from aig.genimage_reporting import MODEL_LABELS, write_accuracy_tables
from aig.io import atomic_write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--batch-dir', required=True)
    parser.add_argument('--watch', action='store_true', help='Refresh until every planned GenImage matrix is complete.')
    parser.add_argument('--interval', type=float, default=10)
    args = parser.parse_args()
    if args.interval < 1:
        parser.error('--interval must be at least one second.')
    batch = Path(args.batch_dir).expanduser()
    batch = batch.resolve() if batch.is_absolute() else (ROOT / batch).resolve()
    if not batch.is_relative_to(ROOT / 'runs') or not (batch / 'plan.json').is_file():
        parser.error('Select an existing batch under this project\'s runs folder.')
    plan = json.loads((batch / 'plan.json').read_text(encoding='utf-8'))
    if 'tiny_genimage' not in plan['datasets']:
        parser.error('This batch does not include Tiny GenImage.')
    if any(model not in MODEL_LABELS for model in plan['models']):
        parser.error('Unknown model in the selected batch.')
    previous = None
    while True:
        path = batch / 'genimage_matrices.json'
        record = json.loads(path.read_text(encoding='utf-8')) if path.exists() else {'matrices': []}
        result = write_accuracy_tables(batch, record, plan['models'], plan['seeds'])
        if result != previous:
            atomic_write_json(batch / 'presentation' / 'export_status.json',
                              dict(result, updated_utc=datetime.now(timezone.utc).isoformat(),
                                   status='completed' if result['complete'] else 'waiting_for_test_results'))
            print(json.dumps(result), flush=True)
            previous = result
        if not args.watch or result['complete']:
            break
        time.sleep(args.interval)


if __name__ == '__main__':
    main()
