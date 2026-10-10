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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--models", nargs="+", choices=("learned_filter_cnn", "fixed_filter_cnn", "dual_cnn_128"),
                        default=["learned_filter_cnn", "fixed_filter_cnn", "dual_cnn_128"])
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

        for model in args.models:
            run_dir = temporary / model
            common = ["--model", model, "--dataset", "smoke", "--data-root", str(temporary / "data"),
                      "--run-dir", str(run_dir), "--batch-size", "2", "--seed", "42", "--device", args.device]
            invoke("train.py", [*common, "--epochs", "1"])
            first = torch.load(run_dir / "last.pt", map_location="cpu", weights_only=True)
            invoke("train.py", [*common, "--epochs", "1"], success=False)
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
            continuous_dir = temporary / (model + "_continuous")
            uninterrupted = [*common, "--run-dir", str(continuous_dir), "--epochs", "2"]
            invoke("train.py", uninterrupted)
            continuous = torch.load(continuous_dir / "last.pt", map_location="cpu", weights_only=True)
            for name, tensor in resumed["model_state"].items():
                torch.testing.assert_close(tensor, continuous["model_state"][name], rtol=0, atol=0)
            invoke("train.py", [*common, "--seed", "43", "--epochs", "3",
                                "--resume", str(run_dir / "last.pt")], success=False)
            print(f"{model}: train, overwrite guard, resume equivalence, kernel behavior, held-out evaluation, mismatch guard OK", flush=True)
    print("Workflow checks passed; temporary images and checkpoints removed.", flush=True)


if __name__ == "__main__":
    main()
