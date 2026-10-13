"""Cross-platform launcher; no virtual-environment activation is necessary."""

import argparse
import math
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
PYTHON = ROOT / ".venv" / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
DATASETS = ("cifake", "progan", "tiny_genimage")


def invoke(arguments, base_python=False):
    python = Path(sys.executable) if base_python else PYTHON
    if not python.is_file():
        raise FileNotFoundError("Project .venv is missing. Run: python scripts/run.py setup --compute cu128 (or cpu)")
    environment = os.environ.copy()
    environment.update(PYTHONUTF8="1", MPLBACKEND="Agg", MPLCONFIGDIR=str(ROOT / "tmp" / "matplotlib"))
    print("Running: " + subprocess.list2cmdline([str(python), *arguments]), flush=True)
    subprocess.run([str(python), *arguments], cwd=ROOT, env=environment, check=True)


def prepare(names, workers, prepare_only=False):
    if not prepare_only:
        invoke(["tools/download_benchmarks.py", "--datasets", *names])
    invoke(["tools/prepare_benchmarks.py", "--datasets", *names, "--workers", str(workers)])


def main(argv=None):
    arguments = list(sys.argv[1:] if argv is None else argv)
    if not arguments or arguments[0] in ("-h", "--help"):
        print(__doc__ + "\nCommands: setup, data, train, evaluate, test, pipeline\n"
              "Examples:\n  python scripts/run.py setup --compute cu128\n"
              "  python scripts/run.py data --datasets progan\n"
              "  python scripts/run.py train --dataset progan_poc --device cuda\n"
              "  python scripts/run.py pipeline --dataset progan_poc --compute cu128\n"
              "Use <command> --help for options.")
        return
    command, tail = arguments[0], arguments[1:]
    if command == "setup":
        invoke(["scripts/setup_env.py", *tail], base_python=True)
    elif command in ("train", "evaluate"):
        invoke(["train.py" if command == "train" else "evaluate.py", *tail])
    elif command == "test":
        invoke(["-m", "unittest", "discover", "-s", "tests", "-v", *tail])
    elif command == "data":
        parser = argparse.ArgumentParser(description="Download and prepare reproducible binary datasets")
        parser.add_argument("--datasets", nargs="+", choices=DATASETS, default=list(DATASETS))
        parser.add_argument("--workers", type=int, default=8)
        parser.add_argument("--prepare-only", action="store_true", help="Use transferred archives and their JSON metadata")
        args = parser.parse_args(tail)
        if args.workers < 1:
            parser.error("Preparation workers must be positive.")
        prepare(args.datasets, args.workers, args.prepare_only)
    elif command == "pipeline":
        parser = argparse.ArgumentParser(description="Set up, download, prepare, train, and evaluate one POC dataset")
        parser.add_argument("--dataset", choices=("cifake_poc", "progan_poc", "tiny_genimage_poc"), default="progan_poc")
        parser.add_argument("--compute", choices=("auto", "cpu", "cu126", "cu128", "cu130"), default="auto")
        parser.add_argument("--model", choices=("learned_filter_cnn", "fixed_filter_cnn", "dual_cnn_128", "fixed_filter_kcnet", "finetuned_cnn_kcnet"), default="learned_filter_cnn")
        parser.add_argument("--encoder-checkpoint", help="Validation-selected fixed_filter_cnn best.pt for a fresh KC run")
        parser.add_argument("--epochs", type=int, default=20)
        parser.add_argument("--batch-size", type=int, default=32)
        parser.add_argument("--seed", type=int, default=42)
        parser.add_argument("--workers", type=int, default=0)
        parser.add_argument("--patience", type=int, help="Validation BCE patience; 0 disables")
        parser.add_argument("--min-delta", type=float, help="Minimum validation BCE decrease that resets patience")
        parser.add_argument("--skip-setup", action="store_true")
        parser.add_argument("--skip-data", action="store_true")
        args = parser.parse_args(tail)
        if args.epochs < 1 or args.batch_size < 1 or args.workers < 0 or not 0 <= args.seed < 2**32:
            parser.error("Use positive epochs/batch size, nonnegative workers, and seed in [0, 2**32).")
        if (args.model in ("fixed_filter_kcnet", "finetuned_cnn_kcnet")) != bool(args.encoder_checkpoint):
            parser.error("--encoder-checkpoint is required only with a KC model.")
        if ((args.patience is not None and args.patience < 0)
                or (args.min_delta is not None and (not math.isfinite(args.min_delta) or args.min_delta < 0))):
            parser.error("Use nonnegative patience and finite nonnegative min_delta.")
        if not args.skip_setup:
            invoke(["scripts/setup_env.py", "--compute", args.compute], base_python=True)
        if not args.skip_data:
            prepare([args.dataset.removesuffix("_poc")], 8)
        device = "cpu" if args.compute == "cpu" else "auto" if args.compute == "auto" else "cuda"
        common = ["--dataset", args.dataset, "--model", args.model, "--epochs", str(args.epochs),
                  "--batch-size", str(args.batch_size), "--seed", str(args.seed),
                  "--workers", str(args.workers), "--device", device]
        if args.encoder_checkpoint:
            common += ["--encoder-checkpoint", args.encoder_checkpoint]
        if args.patience is not None:
            common += ["--patience", str(args.patience)]
        if args.min_delta is not None:
            common += ["--min-delta", str(args.min_delta)]
        invoke(["train.py", *common, "--check-only"])
        invoke(["train.py", *common])
        checkpoint = f"runs/{args.model}/{args.dataset}/seed_{args.seed}/best.pt"
        invoke(["evaluate.py", "--checkpoint", checkpoint, "--batch-size", str(args.batch_size),
                "--workers", str(args.workers), "--device", device])
    else:
        raise ValueError(f"Unknown command '{command}'. Use --help.")


if __name__ == "__main__":
    try:
        main()
    except subprocess.CalledProcessError as error:
        sys.exit(error.returncode)
    except (FileNotFoundError, ValueError) as error:
        sys.exit(str(error))
