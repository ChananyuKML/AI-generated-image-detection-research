"""Exercise real train/resume/evaluate CLIs on tiny temporary synthetic data."""

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

import numpy as np
from PIL import Image
import torch

ROOT = Path(__file__).resolve().parents[1]
KC_MODELS = ("fixed_filter_kcnet", "finetuned_cnn_kcnet")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--models", nargs="+", choices=("learned_filter_cnn", "fixed_filter_cnn", "dual_cnn_128", *KC_MODELS),
                        default=["learned_filter_cnn", "fixed_filter_cnn", "dual_cnn_128", *KC_MODELS])
    args = parser.parse_args()
    temporary_root = (ROOT / "tmp" / "workflow_checks").resolve()
    if not temporary_root.is_relative_to(ROOT):
        raise ValueError("Temporary files must stay inside the project.")
    temporary_root.mkdir(parents=True, exist_ok=True)
    environment = os.environ.copy()
    environment.update(OMP_NUM_THREADS="2", MKL_NUM_THREADS="2", PYTHONUTF8="1")
    with tempfile.TemporaryDirectory(dir=temporary_root) as directory:
        temporary = Path(directory)
        dataset_root = temporary / "data" / "smoke"
        generator = np.random.default_rng(12)
        for split in ("train", "val", "test"):
            for name in ("ai", "nature"):
                folder = dataset_root / split / name
                folder.mkdir(parents=True)
                for index in range(2):
                    Image.fromarray(generator.integers(0, 256, (32, 32, 3), dtype=np.uint8)).save(folder / f"{index}.png")

        def invoke(script, arguments, success=True):
            result = subprocess.run([sys.executable, str(ROOT / script), *arguments],
                                    cwd=temporary, env=environment, text=True,
                                    stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
            print(result.stdout, flush=True)
            if success and result.returncode:
                raise RuntimeError(f"{script} failed: {result.returncode}")
            if not success and not result.returncode:
                raise AssertionError("Expected overwrite/configuration guard to refuse the run.")

        encoder = temporary / "encoder" / "best.pt"
        if any(model in KC_MODELS for model in args.models):
            invoke("train.py", ["--model", "fixed_filter_cnn", "--dataset", "smoke",
                                "--data-root", str(temporary / "data"), "--run-dir", str(encoder.parent),
                                "--batch-size", "2", "--device", args.device, "--epochs", "1"])
        for model in args.models:
            run_dir = temporary / model
            common = ["--model", model, "--dataset", "smoke", "--data-root", str(temporary / "data"),
                      "--run-dir", str(run_dir), "--batch-size", "2", "--seed", "42", "--device", args.device]
            fresh = [*common, "--encoder-checkpoint", str(encoder), "--kc-units", "64", "--projection-seed", "17"] if model in KC_MODELS else common
            if model in KC_MODELS:
                invoke("train.py", [*common, "--epochs", "1"], success=False)
            invoke("train.py", [*fresh, "--epochs", "1"])
            first = torch.load(run_dir / "last.pt", map_location="cpu", weights_only=True)
            invoke("train.py", [*fresh, "--epochs", "1"], success=False)
            if model in KC_MODELS:
                # Prove resume/evaluation do not require the initialization file.
                encoder.rename(encoder.with_name("source_hidden.pt"))
            invoke("train.py", [*common, "--epochs", "2", "--resume", str(run_dir / "last.pt")])
            resumed = torch.load(run_dir / "last.pt", map_location="cpu", weights_only=True)
            assert resumed["epoch"] == 2 and len(resumed["history"]) == 2
            assert torch.equal(first["rng"]["loader"], resumed["rng"]["loader"]) is False
            if model != "dual_cnn_128":
                initial_gx = torch.tensor([[[[0., 0., 0.], [0., -1., 1.], [0., 0., 0.]]]])
                changed = not torch.equal(resumed["model_state"]["frontend.Gx"], initial_gx)
                assert changed == (model == "learned_filter_cnn")
            invoke("evaluate.py", ["--checkpoint", str(run_dir / "best.pt"), "--dataset", "smoke",
                                   "--data-root", str(temporary / "data"), "--device", args.device, "--batch-size", "2"])
            metrics = json.loads((run_dir / "evaluation" / "smoke" / "test" / "metrics.json").read_text())
            assert metrics["images"] == 4 and metrics["split"] == "test"
            assert 0 <= metrics["roc_auc"] <= 1 and metrics["threshold"] == .5
            if model in KC_MODELS:
                assert resumed["config"]["model_config"]["kc_units"] == 64
                assert resumed["config"]["model_config"]["projection_seed"] == 17
                for name, tensor in first["model_state"].items():
                    trainable = name.startswith("readout.") or (model == "finetuned_cnn_kcnet" and name.startswith("cnn_branch."))
                    if not trainable:
                        torch.testing.assert_close(tensor, resumed["model_state"][name], rtol=0, atol=0)
                if model == "finetuned_cnn_kcnet":
                    assert any(not torch.equal(tensor, resumed["model_state"][name])
                               for name, tensor in first["model_state"].items() if name.startswith("cnn_branch."))
                invoke("train.py", [*common, "--epochs", "3", "--projection-seed", "18",
                                    "--resume", str(run_dir / "last.pt")], success=False)
                encoder.with_name("source_hidden.pt").rename(encoder)
            continuous_dir = temporary / (model + "_continuous")
            uninterrupted = [*fresh, "--run-dir", str(continuous_dir), "--epochs", "2"]
            invoke("train.py", uninterrupted)
            continuous = torch.load(continuous_dir / "last.pt", map_location="cpu", weights_only=True)
            for name, tensor in resumed["model_state"].items():
                torch.testing.assert_close(tensor, continuous["model_state"][name], rtol=0, atol=0)
            invoke("train.py", [*common, "--seed", "43", "--epochs", "3",
                                "--resume", str(run_dir / "last.pt")], success=False)
            print(f"{model}: train, overwrite guard, resume equivalence, kernel behavior, held-out evaluation, mismatch guard OK", flush=True)
        # A large synthetic min_delta makes the stopping boundary deterministic.
        # Remove new metadata to exercise enabling early stopping on a legacy run.
        stopping_model = args.models[0]
        early_dir = temporary / "early_stopping"
        early_common = ["--model", stopping_model, "--dataset", "smoke", "--data-root", str(temporary / "data"),
                        "--run-dir", str(early_dir), "--batch-size", "2", "--seed", "42", "--device", args.device]
        early_fresh = [*early_common, "--encoder-checkpoint", str(encoder)] if stopping_model in KC_MODELS else early_common
        invoke("train.py", [*early_fresh, "--epochs", "1"])
        legacy = torch.load(early_dir / "last.pt", map_location="cpu", weights_only=True)
        legacy["config"].pop("early_stopping")
        legacy.pop("early_stopping_state")
        torch.save(legacy, early_dir / "last.pt")
        resume = [*early_common, "--resume", str(early_dir / "last.pt")]
        invoke("train.py", [*resume, "--epochs", "10", "--patience", "2", "--min-delta", "1"])
        stopped = torch.load(early_dir / "last.pt", map_location="cpu", weights_only=True)
        summary = json.loads((early_dir / "summary.json").read_text())
        assert stopped["epoch"] == summary["completed_epochs"] == 3
        assert summary["requested_epochs"] == 10 and summary["stopped_early"] is True
        assert stopped["early_stopping_state"]["bad_epochs"] == 2
        minimum = min(stopped["history"], key=lambda row: row["val_loss"])
        best = torch.load(early_dir / "best.pt", map_location="cpu", weights_only=True)
        assert best["epoch"] == stopped["best_epoch"] == minimum["epoch"]
        assert best["best_val_loss"] == minimum["val_loss"]
        invoke("train.py", [*resume, "--epochs", "10"], success=False)
        invoke("train.py", [*resume, "--epochs", "10", "--patience", "3"])
        assert torch.load(early_dir / "last.pt", map_location="cpu", weights_only=True)["epoch"] == 4
        invoke("train.py", [*resume, "--epochs", "6", "--patience", "0"])
        summary = json.loads((early_dir / "summary.json").read_text())
        assert summary["completed_epochs"] == summary["requested_epochs"] == 6
        assert summary["stopped_early"] is False and summary["stop_reason"] == "epoch_target"
        print("early stopping: legacy resume, exact stop epoch, absolute-best selection, carried patience, exhausted guard and explicit disable OK", flush=True)
    print("Workflow checks passed; temporary images and checkpoints removed.", flush=True)


if __name__ == "__main__":
    main()
