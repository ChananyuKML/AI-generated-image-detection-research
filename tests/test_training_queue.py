"""Full-split scheduling, process checks, and observational progress checks."""

import json
import os
from pathlib import Path
import tempfile
import unittest
from argparse import Namespace
from unittest.mock import patch

import torch
from torch.utils.data import DataLoader, TensorDataset

from aig.experiment import ROOT
from aig.progress import TrainingProgress
from aig.training import run_epoch
from tools.run_full_training import TrainingQueue, load_live_view, make_jobs, pid_alive, write_json


class TrainingQueueTests(unittest.TestCase):
    def test_full_matrix_orders_matching_encoder_before_kc_models(self):
        jobs = make_jobs()
        self.assertEqual(len(jobs), 15)
        self.assertEqual(len({job['id'] for job in jobs}), 15)
        for name in ('progan', 'tiny_genimage', 'cifake'):
            models = [job['model'] for job in jobs if job['dataset'] == name]
            self.assertEqual(models[0], 'fixed_filter_cnn')
            self.assertLess(models.index('fixed_filter_cnn'), models.index('fixed_filter_kcnet'))
            self.assertLess(models.index('fixed_filter_cnn'), models.index('finetuned_cnn_kcnet'))
        for datasets, models in [(('progan_poc',), ('fixed_filter_cnn',)),
                                 (('progan',), ('fixed_filter_kcnet',))]:
            with self.assertRaises(ValueError):
                make_jobs(datasets, models)
        with self.assertRaises(ValueError):
            make_jobs(seeds=(42, 42))

    def test_pid_check_observes_current_process_without_signaling_it(self):
        self.assertTrue(pid_alive(os.getpid()))
        self.assertFalse(pid_alive(-1))

    def test_live_view_accepts_fresh_trainer_pid_different_from_venv_launcher(self):
        temporary_root = ROOT / 'tmp' / 'tests'
        temporary_root.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=temporary_root) as directory:
            batch = Path(directory)
            job = {'id': 'progan/fixed_filter_cnn/seed_42', 'dataset': 'progan',
                   'model': 'fixed_filter_cnn', 'seed': 42, 'pid': 100,
                   'status': 'training', 'started_utc': '2026-10-13T03:00:00+00:00'}
            state = {'status': 'running', 'jobs': [job], 'active_job': job['id']}
            path = batch / 'experiments' / job['model'] / job['dataset'] / 'seed_42' / 'progress.json'
            progress = {'model': job['model'], 'dataset': job['dataset'], 'seed': 42,
                        'pid': 200, 'epoch': 4, 'updated_utc': '2026-10-13T03:10:00+00:00'}
            write_json(path, progress)
            with patch('tools.run_full_training.pid_alive', return_value=True):
                value = load_live_view(state, batch, child_alive=True)
            self.assertEqual(value['process_pid'], 200)
            self.assertEqual(value['progress']['epoch'], 4)
            self.assertTrue(value['child_alive'])
            progress['updated_utc'] = '2026-10-12T03:10:00+00:00'
            write_json(path, progress)
            self.assertIsNone(load_live_view(state, batch, child_alive=True)['progress'])

    def queue_in(self, directory):
        args = Namespace(batch_dir=directory, datasets=['progan'], models=['fixed_filter_cnn'], seeds=[42],
                         epochs=100, patience=2, batch_size=32, workers=0, port=8765)
        queue = TrainingQueue(args)
        queue.source = ROOT
        queue.state['datasets'] = {'progan': {'signature': {'kind': 'manifest_sha256', 'sha256': 'test'}}}
        return queue

    def test_completed_run_is_reused_and_mismatched_configuration_is_rejected(self):
        temporary_root = (ROOT / 'runs' / 'queue_tests').resolve()
        self.assertTrue(temporary_root.is_relative_to(ROOT / 'runs'))
        temporary_root.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=temporary_root) as directory:
            queue = self.queue_in(directory)
            job = queue.state['jobs'][0]
            run = queue.run_dir(job)
            run.mkdir(parents=True)
            config = queue.expected_config(job)
            write_json(run / 'config.json', config)
            write_json(run / 'summary.json', {'config': config, 'completed_epochs': 17, 'stopped_early': True})
            (run / 'best.pt').write_bytes(b'checkpoint')
            (run / 'last.pt').write_bytes(b'checkpoint')
            self.assertIsNone(queue.train_arguments(job))
            write_json(run / 'config.json', dict(config, seed=43))
            with self.assertRaises(ValueError):
                queue.train_arguments(job)
            with self.assertRaises(RuntimeError):
                self.queue_in(directory)

    def test_interrupted_stopping_boundary_recovers_completed_summary(self):
        temporary_root = (ROOT / 'runs' / 'queue_tests').resolve()
        self.assertTrue(temporary_root.is_relative_to(ROOT / 'runs'))
        temporary_root.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=temporary_root) as directory:
            queue = self.queue_in(directory)
            job = queue.state['jobs'][0]
            run = queue.run_dir(job)
            run.mkdir(parents=True)
            config = queue.expected_config(job)
            write_json(run / 'config.json', config)
            torch.save({'config': config, 'epoch': 4, 'best_epoch': 2, 'best_val_loss': .6,
                        'history': [{'val_loss': loss} for loss in [.7, .6, .65, .66]]}, run / 'last.pt')
            (run / 'best.pt').write_bytes(b'checkpoint')
            self.assertIsNone(queue.train_arguments(job))
            summary = json.loads((run / 'summary.json').read_text())
            self.assertEqual(summary['completed_epochs'], 4)
            self.assertEqual(summary['stop_reason'], 'early_stopping')

    def test_progress_has_no_effect_on_training_weights_or_rng(self):
        temporary_root = ROOT / 'tmp' / 'tests'
        temporary_root.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=temporary_root) as directory:
            torch.manual_seed(9)
            dataset = TensorDataset(torch.randn(6, 2), torch.tensor([0, 1, 0, 1, 0, 1]))
            model = torch.nn.Linear(2, 1)
            copied = torch.nn.Linear(2, 1)
            copied.load_state_dict(model.state_dict())
            loader = DataLoader(dataset, batch_size=2, shuffle=False)
            rng = torch.get_rng_state()
            expected = run_epoch(model, loader, torch.nn.BCEWithLogitsLoss(), torch.device('cpu'),
                                 torch.optim.AdamW(model.parameters(), lr=.001))
            expected_rng = torch.get_rng_state()
            torch.set_rng_state(rng)
            progress = TrainingProgress(directory, 'test', 'progan', 42, 100, interval=0)
            progress.start_epoch(1)
            actual = run_epoch(copied, loader, torch.nn.BCEWithLogitsLoss(), torch.device('cpu'),
                               torch.optim.AdamW(copied.parameters(), lr=.001), progress)
            self.assertEqual(expected, actual)
            self.assertTrue(torch.equal(expected_rng, torch.get_rng_state()))
            for a, b in zip(model.parameters(), copied.parameters()):
                torch.testing.assert_close(a, b, rtol=0, atol=0)
            report = json.loads((Path(directory) / 'progress.json').read_text())
            self.assertEqual((report['phase'], report['batch'], report['batches']), ('training', 3, 3))
            self.assertEqual(report['samples'], 6)
            self.assertTrue(0 <= report['current_accuracy'] <= 1)


if __name__ == '__main__':
    unittest.main()
