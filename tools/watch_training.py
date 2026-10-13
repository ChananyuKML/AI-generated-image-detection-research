"""Read-only dashboard for an already-running full-training batch."""

import argparse
from http.server import ThreadingHTTPServer
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.run_full_training import TrainingQueue, load_live_view, make_handler, pid_alive, read_json


class BatchReader:
    def __init__(self, batch):
        self.batch = batch
        self.gpu_cache, self.gpu_updated = None, 0

    gpu = TrainingQueue.gpu

    def view(self):
        state = read_json(self.batch / 'status.json')
        if not state:
            raise ValueError('The selected batch has no readable status.json.')
        active = next((job for job in state['jobs'] if job['id'] == state.get('active_job')), None)
        return load_live_view(state, self.batch, pid_alive(active.get('pid')) if active else False,
                              manager_alive=pid_alive(state.get('manager_pid')), gpu=self.gpu())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--batch-dir', required=True)
    parser.add_argument('--port', type=int, default=8766)
    args = parser.parse_args()
    batch = Path(args.batch_dir)
    batch = batch.resolve() if batch.is_absolute() else (ROOT / batch).resolve()
    if not batch.is_relative_to(ROOT / 'runs') or not (batch / 'status.json').is_file():
        parser.error('Select an existing batch under this project\'s runs folder.')
    reader = BatchReader(batch)
    server = ThreadingHTTPServer(('127.0.0.1', args.port), make_handler(reader))
    print(f'Read-only live progress: http://127.0.0.1:{args.port}/', flush=True)
    try:
        server.serve_forever()
    finally:
        server.server_close()


if __name__ == '__main__':
    main()
