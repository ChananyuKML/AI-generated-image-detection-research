"""Cross-generator split integrity, queue coverage, and macro-mean definitions."""

import unittest

from aig.genimage_protocol import GENERATORS, DATASETS, assign_protocol_splits, audit_cross_generator_splits, matrix_summary
from tools.run_full_training import make_jobs


def record(identity, generator='sdv5', label=1, split='train'):
    return {'pixel_sha256': identity, 'generator': generator if label else 'real', 'label': label,
            'source_generators': [generator], 'original_split': split, 'original_splits': [split]}


class GenImageProtocolTests(unittest.TestCase):
    def test_supplied_val_is_only_test_and_selection_val_is_from_train(self):
        rows = [record(f'{label}-{i}', label=label) for label in (0, 1) for i in range(20)]
        rows += [record('heldout-' + str(label), label=label, split='val') for label in (0, 1)]
        assigned = assign_protocol_splits(rows)
        self.assertEqual(sum(row['split'] == 'test' for row in assigned), 2)
        self.assertEqual(sum(row['split'] == 'val' for row in assigned), 4)
        self.assertTrue(all(row['original_split'] == 'train' for row in assigned if row['split'] != 'test'))
        audit_cross_generator_splits({'sdv5': assigned})
        self.assertEqual(assigned, assign_protocol_splits(list(reversed(rows))))

    def test_any_source_train_to_any_target_test_overlap_is_rejected(self):
        training = dict(record('overlap'), split='train')
        testing = dict(record('overlap', generator='adm', split='val'), split='test')
        with self.assertRaises(ValueError):
            audit_cross_generator_splits({'sdv5': [training], 'adm': [testing]})

    def test_matrix_means_use_equal_cells_and_keep_partial_final_means_empty(self):
        cells = [{'training_generator': source, 'testing_generator': target,
                  'accuracy': .9 if source == target else .5, 'roc_auc': .8}
                 for source in GENERATORS for target in GENERATORS]
        summary = matrix_summary(cells)
        self.assertTrue(summary['complete'])
        self.assertEqual(summary['completed_cells'], 49)
        self.assertAlmostEqual(summary['diagonal_accuracy'], .9)
        self.assertAlmostEqual(summary['off_diagonal_accuracy'], .5)
        self.assertAlmostEqual(summary['macro_accuracy'], (7*.9 + 42*.5)/49)
        self.assertIsNone(matrix_summary(cells[:7])['macro_accuracy'])
        with self.assertRaises(ValueError):
            matrix_summary(cells + [cells[0]])

    def test_cross_generator_queue_has_35_tiny_detectors_and_245_tests(self):
        jobs = make_jobs(genimage_protocol='cross-generator')
        self.assertEqual(len(jobs), 45)
        tiny = [job for job in jobs if job['dataset'] in DATASETS]
        self.assertEqual(len(tiny), 35)
        self.assertEqual(sum(len(job['evaluation_targets']) for job in tiny), 245)
        self.assertFalse(any(job['dataset'] == 'tiny_genimage' for job in tiny))
        for dataset in DATASETS:
            models = [job['model'] for job in tiny if job['dataset'] == dataset]
            self.assertEqual(models[0], 'fixed_filter_cnn')


if __name__ == '__main__':
    unittest.main()
