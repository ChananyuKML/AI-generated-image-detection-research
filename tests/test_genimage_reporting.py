"""Paper-table percentages, orientation, and incomplete-result accounting."""

from pathlib import Path
import tempfile
import unittest

from aig.genimage_protocol import GENERATORS, matrix_summary
from aig.genimage_reporting import normalize_matrices, write_accuracy_tables


def matrix(cells):
    return dict(matrix_summary(cells), model='learned_filter_cnn', seed=42)


class GenImageReportingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.test_root = Path(__file__).resolve().parents[1] / 'tmp' / 'reporting_tests'
        cls.test_root.mkdir(parents=True, exist_ok=True)

    def test_column_means_average_sources_and_render_percent(self):
        cells = [dict(training_generator=source, testing_generator=target,
                      accuracy=(10 * i + j) / 100)
                 for i, source in enumerate(GENERATORS) for j, target in enumerate(GENERATORS)]
        record = {'matrices': [matrix(cells)]}
        result = normalize_matrices(record, ['learned_filter_cnn'], [42])[0]
        self.assertAlmostEqual(result['column_means']['midjourney'], .30)
        self.assertAlmostEqual(result['row_means']['midjourney'], .03)
        self.assertAlmostEqual(result['macro_accuracy'], .33)
        with tempfile.TemporaryDirectory(dir=self.test_root) as directory:
            exported = write_accuracy_tables(directory, record, ['learned_filter_cnn'], [42])
            text = Path(exported['markdown']).read_text(encoding='utf-8')
            self.assertIn('| Learned-filter CNN | 30.00 | 31.00 | 32.00 | 33.00 | 34.00 | 35.00 | 36.00 | 33.00 | 49/49 |', text)
            self.assertIn('| Midjourney | 0.00 | 1.00 | 2.00 | 3.00 | 4.00 | 5.00 | 6.00 | 3.00 |', text)
            self.assertTrue(exported['complete'])

    def test_missing_results_are_pending_and_do_not_change_denominators(self):
        cells = [dict(training_generator='midjourney', testing_generator=target, accuracy=.9)
                 for target in GENERATORS]
        result = normalize_matrices({'matrices': [matrix(cells)]}, ['learned_filter_cnn', 'fixed_filter_cnn'], [42])
        self.assertIsNone(result[0]['macro_accuracy'])
        self.assertIsNone(result[0]['column_means']['midjourney'])
        self.assertAlmostEqual(result[0]['row_means']['midjourney'], .9)
        self.assertEqual(result[1]['completed_cells'], 0)
        with tempfile.TemporaryDirectory(dir=self.test_root) as directory:
            exported = write_accuracy_tables(directory, {'matrices': [matrix(cells)]}, ['learned_filter_cnn'], [42])
            self.assertFalse(exported['complete'])
            self.assertIn('Pending', Path(exported['markdown']).read_text(encoding='utf-8'))
            self.assertIn('--', Path(exported['latex']).read_text(encoding='utf-8'))

    def test_unknown_order_duplicate_identity_and_bad_accuracy_are_rejected(self):
        item = matrix([])
        with self.assertRaises(ValueError):
            normalize_matrices({'matrices': [item, item]}, ['learned_filter_cnn'], [42])
        with self.assertRaises(ValueError):
            normalize_matrices({'matrices': [dict(item, generators=list(reversed(GENERATORS)))]}, ['learned_filter_cnn'], [42])
        item['accuracy'][0][0] = float('nan')
        with self.assertRaises(ValueError):
            normalize_matrices({'matrices': [item]}, ['learned_filter_cnn'], [42])


if __name__ == '__main__':
    unittest.main()
