"""Checks that derived benchmark splits do not leak exact image identities."""

import unittest

from tools.prepare_benchmarks import assign_splits, collapse_records, describe_member, sample_balanced


def record(identity, label=0, split="train", generator="sdv4", category="sample"):
    return {"pixel_sha256": identity, "file_sha256": identity, "source_member": f"{split}/{identity}.png",
            "label": label, "original_split": split, "source_generator": generator,
            "generator": generator if label else "real", "category": category}


class DatasetPreparationTests(unittest.TestCase):
    def test_source_labels_and_generators_are_explicit(self):
        self.assertEqual(describe_member("cifake", "test/REAL/a.jpg")["label"], 0)
        self.assertEqual(describe_member("cifake", "train/FAKE/b.jpg")["label"], 1)
        progan = describe_member("progan", "progan_val/airplane/0_real/a.png")
        self.assertEqual((progan["category"], progan["original_split"]), ("airplane", "val"))
        tiny = describe_member("tiny_genimage", "imagenet_ai_0424_sdv5/train/ai/000.png")
        self.assertEqual((tiny["generator"], tiny["label"]), ("sdv5", 1))

    def test_duplicate_group_keeps_original_test_and_removes_label_conflicts(self):
        records = [record("same", split="train"), record("same", split="test"),
                   record("conflict", label=0), record("conflict", label=1)]
        unique, rejected = collapse_records(records)
        self.assertEqual(len(unique), 1)
        self.assertEqual(unique[0]["original_split"], "test")
        self.assertEqual(len(unique[0]["related_members"]), 2)
        self.assertEqual(len(rejected), 2)
        assigned = assign_splits("cifake", unique)
        self.assertEqual(assigned[0]["split"], "test")

    def test_progan_partition_is_disjoint_reproducible_and_balanced(self):
        records = [record(f"{label}-{index}", label=label, split="val", generator="progan")
                   for label in (0, 1) for index in range(100)]
        unique, _ = collapse_records(records)
        first = assign_splits("progan", unique)
        second = assign_splits("progan", list(reversed(unique)))
        mapping = lambda values: {row["pixel_sha256"]: row["split"] for row in values}
        self.assertEqual(mapping(first), mapping(second))
        balanced = sample_balanced(first)
        for split, expected in [("train", 140), ("val", 30), ("test", 30)]:
            rows = [row for row in balanced if row["split"] == split]
            self.assertEqual(len(rows), expected)
            self.assertEqual(sum(row["label"] for row in rows), expected // 2)
        hashes = [{row["pixel_sha256"] for row in balanced if row["split"] == split}
                  for split in ("train", "val", "test")]
        self.assertFalse(hashes[0] & hashes[1] or hashes[0] & hashes[2] or hashes[1] & hashes[2])

    def test_tiny_supplied_validation_is_split_without_moving_training_groups(self):
        records = [record(f"{label}-{split}-{index}", label=label, split=split, generator="sdv5")
                   for label in (0, 1) for split in ("train", "val") for index in range(10)]
        unique, _ = collapse_records(records)
        assigned = assign_splits("tiny_genimage", unique)
        self.assertTrue(all(row["split"] == "train" for row in assigned if row["original_split"] == "train"))
        self.assertEqual(sum(row["split"] == "val" for row in assigned), 10)
        self.assertEqual(sum(row["split"] == "test" for row in assigned), 10)


if __name__ == "__main__":
    unittest.main()
