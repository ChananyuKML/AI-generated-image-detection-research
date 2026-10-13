"""Patience boundaries, resume policy, legacy history and input validation."""

import unittest

from aig.early_stopping import EarlyStopping
from aig.training import parser, resolve_early_stopping_config


class EarlyStoppingTests(unittest.TestCase):
    def test_ties_count_and_strict_improvement_resets_patience(self):
        monitor = EarlyStopping(patience=2)
        self.assertFalse(monitor.update(.5))
        self.assertFalse(monitor.update(.5))
        self.assertFalse(monitor.update(.4))
        self.assertEqual(monitor.bad_epochs, 0)
        self.assertFalse(monitor.update(.41))
        self.assertTrue(monitor.update(.4))
        self.assertEqual(monitor.bad_epochs, 2)
        disabled = EarlyStopping()
        for _ in range(20):
            self.assertFalse(disabled.update(.5))

    def test_small_improvements_accumulate_against_monitor_anchor(self):
        monitor = EarlyStopping(patience=5, min_delta=.01)
        for value in (.5, .495, .491):
            self.assertFalse(monitor.update(value))
        self.assertEqual(monitor.best_monitor_loss, .5)
        self.assertEqual(monitor.bad_epochs, 2)
        self.assertFalse(monitor.update(.489))
        self.assertEqual(monitor.best_monitor_loss, .489)
        self.assertEqual(monitor.bad_epochs, 0)

    def test_history_restores_legacy_counters_and_checks_saved_state(self):
        history = [{"val_loss": value} for value in (.5, .4, .42, .41)]
        settings = {"patience": 3, "min_delta": 0.}
        resumed = EarlyStopping.from_history(settings, history)
        self.assertEqual(resumed.bad_epochs, 2)
        restored = EarlyStopping.from_history(settings, history, resumed.state_dict())
        self.assertTrue(restored.update(.4))
        self.assertTrue(resumed.update(.4))
        self.assertEqual(restored.state_dict(), resumed.state_dict())
        with self.assertRaisesRegex(ValueError, "history"):
            EarlyStopping.from_history(settings, history, {"best_monitor_loss": .4, "bad_epochs": 0})

    def test_cli_restores_settings_and_allows_explicit_disable(self):
        saved = {"patience": 10, "min_delta": .001}
        self.assertEqual(resolve_early_stopping_config(parser().parse_args([])), {"patience": 0, "min_delta": 0.})
        self.assertEqual(resolve_early_stopping_config(parser().parse_args([]), saved), saved)
        self.assertEqual(resolve_early_stopping_config(parser().parse_args(["--patience", "0"]), saved), {"patience": 0, "min_delta": .001})
        self.assertEqual(resolve_early_stopping_config(parser().parse_args(["--min-delta", "0"]), saved), {"patience": 10, "min_delta": 0.})

    def test_invalid_settings_and_nonfinite_losses_are_rejected(self):
        for options in ({"patience": -1}, {"patience": True}, {"patience": 1.5},
                        {"min_delta": -1.}, {"min_delta": float("nan")}, {"min_delta": float("inf")}):
            with self.subTest(options=options), self.assertRaises(ValueError):
                EarlyStopping(**options)
        for value in (float("nan"), float("inf"), -float("inf")):
            with self.subTest(value=value), self.assertRaises(FloatingPointError):
                EarlyStopping().update(value)


if __name__ == "__main__":
    unittest.main()
