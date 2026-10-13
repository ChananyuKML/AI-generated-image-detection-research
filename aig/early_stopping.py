"""Validation-loss patience; independent of absolute-best checkpoint selection."""

import math


class EarlyStopping:
    def __init__(self, patience=0, min_delta=0.0):
        if isinstance(patience, bool) or not isinstance(patience, int) or patience < 0:
            raise ValueError("patience must be a nonnegative integer; 0 disables early stopping")
        if not math.isfinite(min_delta) or min_delta < 0:
            raise ValueError("min_delta must be finite and nonnegative")
        self.patience = patience
        self.min_delta = min_delta
        self.best_monitor_loss = float("inf")
        self.bad_epochs = 0

    @property
    def should_stop(self):
        return self.patience > 0 and self.bad_epochs >= self.patience

    def update(self, validation_loss):
        if not math.isfinite(validation_loss):
            raise FloatingPointError("Early stopping requires finite validation loss")
        if validation_loss < self.best_monitor_loss - self.min_delta:
            self.best_monitor_loss = validation_loss
            self.bad_epochs = 0
        else:
            self.bad_epochs += 1
        return self.should_stop

    def state_dict(self):
        return {"best_monitor_loss": self.best_monitor_loss, "bad_epochs": self.bad_epochs}

    @classmethod
    def from_history(cls, settings, history, saved_state=None):
        monitor = cls(**settings)
        for row in history:
            monitor.update(row["val_loss"])
        if saved_state is not None and saved_state != monitor.state_dict():
            raise ValueError("Early-stopping state does not match validation history")
        return monitor
