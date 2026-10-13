"""Lightweight batch telemetry; never changes model or random-number state."""

from datetime import datetime, timezone
import os
from pathlib import Path
import time

from .experiment import write_json


class TrainingProgress:
    def __init__(self, directory, model, dataset, seed, epochs, start_epoch=0, interval=2):
        self.path = Path(directory) / "progress.json"
        self.interval = interval
        self.last_write = float("-inf")
        self.last_warning = float("-inf")
        self.phase_started = time.monotonic()
        self.state = {"model": model, "dataset": dataset, "seed": seed,
                      "requested_epochs": epochs, "completed_epochs": start_epoch,
                      "epoch": start_epoch, "pid": os.getpid(), "status": "running",
                      "phase": "starting", "batch": 0, "batches": 0}
        self.emit(force=True)

    def emit(self, force=False, **values):
        self.state.update(values)
        now = time.monotonic()
        if force or now - self.last_write >= self.interval:
            self.state["updated_utc"] = datetime.now(timezone.utc).isoformat()
            try:
                write_json(self.path, self.state)
            except OSError as error:
                if now - self.last_warning > 60:
                    print(f"Progress update unavailable; training continues: {error}", flush=True)
                    self.last_warning = now
            self.last_write = time.monotonic()

    def start_epoch(self, epoch):
        self.emit(force=True, epoch=epoch, phase="starting_epoch", batch=0, batches=0)

    def start_phase(self, phase, batches):
        self.phase_started = time.monotonic()
        self.emit(force=True, phase=phase, batch=0, batches=batches, samples=0,
                  current_loss=None, current_accuracy=None, phase_eta_seconds=None)

    def batch(self, phase, batch, batches, samples, loss, accuracy):
        if phase != self.state["phase"]:
            self.phase_started = time.monotonic()
        elapsed = time.monotonic() - self.phase_started
        self.emit(force=batch == batches, phase=phase, batch=batch, batches=batches,
                  samples=samples, current_loss=loss, current_accuracy=accuracy,
                  phase_seconds=elapsed, samples_per_second=samples / max(elapsed, .001),
                  phase_eta_seconds=elapsed * (batches - batch) / max(batch, 1))

    def finish_epoch(self, row, best_epoch, best_loss, bad_epochs, patience):
        self.emit(force=True, phase="epoch_complete", completed_epochs=row["epoch"],
                  last_epoch=row, best_epoch=best_epoch, best_val_loss=best_loss,
                  bad_epochs=bad_epochs, patience=patience)

    def finish(self, completed_epochs, stopped_early):
        self.emit(force=True, status="completed", phase="completed",
                  completed_epochs=completed_epochs, stopped_early=stopped_early)
