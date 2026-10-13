"""Regression checks for live-reader conflicts without interrupting training."""

import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from aig.experiment import ROOT
from aig.io import atomic_write_json
from aig.progress import TrainingProgress


class ProgressIOTests(unittest.TestCase):
    def setUp(self):
        self.temporary_root = ROOT / 'tmp' / 'tests'
        self.temporary_root.mkdir(parents=True, exist_ok=True)

    def test_atomic_writer_retries_a_short_reader_lock(self):
        with tempfile.TemporaryDirectory(dir=self.temporary_root) as directory:
            path = Path(directory) / 'progress.json'
            atomic_write_json(path, {'epoch': 1})
            original = Path.replace
            calls = []
            def replace(source, destination):
                calls.append(source)
                if len(calls) < 3:
                    raise PermissionError('reader still holds the previous file')
                return original(source, destination)
            with patch.object(Path, 'replace', replace), patch('aig.io.time.sleep'):
                atomic_write_json(path, {'epoch': 2})
            self.assertEqual(json.loads(path.read_text()), {'epoch': 2})
            self.assertEqual(len(calls), 3)

    def test_failed_replacement_preserves_the_previous_record(self):
        with tempfile.TemporaryDirectory(dir=self.temporary_root) as directory:
            path = Path(directory) / 'progress.json'
            atomic_write_json(path, {'epoch': 1})
            with patch.object(Path, 'replace', side_effect=PermissionError('locked')), patch('aig.io.time.sleep'):
                with self.assertRaises(PermissionError):
                    atomic_write_json(path, {'epoch': 2}, attempts=2)
            self.assertEqual(json.loads(path.read_text()), {'epoch': 1})

    def test_progress_io_failure_does_not_abort_a_training_epoch(self):
        with tempfile.TemporaryDirectory(dir=self.temporary_root) as directory:
            report = TrainingProgress(directory, 'fixed_filter_cnn', 'progan', 42, 100)
            with patch('aig.progress.write_json', side_effect=PermissionError('viewer file lock')), patch('builtins.print'):
                report.start_epoch(11)
                report.start_phase('training', 175)
                report.batch('training', 175, 175, 5600, .25, .89)
                report.finish_epoch({'epoch': 11}, 7, .42, 4, 10)
            self.assertEqual(report.state['completed_epochs'], 11)
            self.assertEqual(report.state['status'], 'running')


if __name__ == '__main__':
    unittest.main()
