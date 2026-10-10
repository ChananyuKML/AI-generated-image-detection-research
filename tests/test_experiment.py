"""Checks for research metrics, strict data loading, and portable paths."""

from pathlib import Path
import tempfile
import unittest

import numpy as np
from PIL import Image

from aig.experiment import BinaryImages, ROOT, binary_metrics, dataset_path, project_path
from utils.custom_transforms import gray_center_224


class ExperimentTests(unittest.TestCase):
    def test_metrics_perfect_reversed_and_tied(self):
        labels = [0, 0, 1, 1]
        perfect = binary_metrics(labels, [.1, .2, .8, .9])
        self.assertEqual(perfect["roc_auc"], 1)
        self.assertEqual(perfect["average_precision"], 1)
        self.assertEqual(perfect["accuracy"], 1)
        reversed_scores = binary_metrics(labels, [.9, .8, .2, .1])
        self.assertEqual(reversed_scores["roc_auc"], 0)
        self.assertAlmostEqual(reversed_scores["average_precision"], (1/3 + 1/2) / 2)
        tied = binary_metrics(labels, [.5] * 4)
        self.assertEqual(tied["roc_auc"], .5)
        self.assertEqual(tied["average_precision"], .5)
        self.assertEqual(tied["confusion_matrix"], {"tn": 2, "fp": 0, "fn": 2, "tp": 0})

    def test_metric_balances_unequal_classes_and_rejects_invalid_data(self):
        metrics = binary_metrics([0, 0, 0, 1], [.1, .1, .1, .1])
        self.assertEqual(metrics["accuracy"], .75)
        self.assertEqual(metrics["balanced_accuracy"], .5)
        for labels, scores in [([1], [.5]), ([0, 1], [0, float("nan")]), ([], [])]:
            with self.assertRaises(ValueError):
                binary_metrics(labels, scores)

    def test_paths_are_anchored_and_dataset_cannot_escape_root(self):
        self.assertEqual(project_path("dataset/prepared"), ROOT / "dataset" / "prepared")
        with self.assertRaises(ValueError):
            dataset_path("dataset/prepared", "../other")

    def test_loader_preserves_labels_and_raises_on_corrupt_images(self):
        temporary_root = ROOT / "tmp" / "tests"
        temporary_root.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=temporary_root) as directory:
            root = Path(directory)
            for name in ("ai", "nature"):
                (root / name).mkdir()
                Image.fromarray(np.full((16, 16, 3), 127, dtype=np.uint8)).save(root / name / "valid.png")
            data = BinaryImages(root, gray_center_224)
            self.assertEqual([label for _, label in data.samples], [0, 1])
            self.assertEqual(tuple(data[0][0].shape), (1, 224, 224))
            (root / "nature" / "valid.png").write_bytes(b"broken")
            with self.assertRaises(OSError):
                data[0]
        with tempfile.TemporaryDirectory(dir=temporary_root) as directory:
            with self.assertRaises(ValueError):
                BinaryImages(directory, gray_center_224)


if __name__ == "__main__":
    unittest.main()
