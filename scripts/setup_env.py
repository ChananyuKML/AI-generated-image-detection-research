"""Create the project's virtual environment with the selected PyTorch build."""

import argparse
import os
from pathlib import Path
import shutil
import subprocess
import sys
import venv

ROOT = Path(__file__).resolve().parents[1]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--compute", choices=("auto", "cpu", "cu126", "cu128", "cu130"), default="auto")
    parser.add_argument("--dry-run", action="store_true", help="Show install commands without changing the environment")
    args = parser.parse_args(argv)
    if not (3, 11) <= sys.version_info[:2] <= (3, 13):
        parser.error("Use Python 3.11, 3.12, or 3.13 (3.13 was tested on the original Windows PC).")
    compute = args.compute
    if compute == "auto":
        compute = "cpu"
        executable = shutil.which("nvidia-smi")
        if executable:
            probe = subprocess.run([executable, "--query-gpu=name", "--format=csv,noheader"],
                                   capture_output=True, text=True, timeout=20)
            if probe.returncode == 0 and probe.stdout.strip():
                compute = "cu128"
    if sys.platform == "darwin" and compute != "cpu":
        parser.error("CUDA wheels are for Windows/Linux NVIDIA PCs. Select --compute cpu on macOS.")
    environment = os.environ.copy()
    environment["PYTHONUTF8"] = "1"
    environment["TMPDIR"] = str(ROOT / "tmp" / "install")
    python = ROOT / ".venv" / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
    torch_command = [str(python), "-m", "pip", "install", "torch==2.9.1", "torchvision==0.24.1"]
    if sys.platform != "darwin":
        torch_command += ["--index-url", f"https://download.pytorch.org/whl/{compute}"]
        # Explicit local versions ensure switching compute modes replaces existing wheels.
        torch_command[4:6] = [f"torch==2.9.1+{compute}", f"torchvision==0.24.1+{compute}"]
    commands = [[str(python), "-m", "pip", "install", "--upgrade", "pip"],
                torch_command,
                [str(python), "-m", "pip", "install", "-r", str(ROOT / "requirements.txt")],
                [str(python), "-m", "pip", "check"]]
    print(f"Project: {ROOT}\nBase Python: {sys.executable}\nCompute: {compute}", flush=True)
    for command in commands:
        print(subprocess.list2cmdline(command), flush=True)
    if args.dry_run:
        return
    (ROOT / "tmp" / "install").mkdir(parents=True, exist_ok=True)
    if not python.exists():
        previous_temp = os.environ.get("TMPDIR")
        os.environ["TMPDIR"] = environment["TMPDIR"]
        try:
            venv.EnvBuilder(with_pip=True).create(ROOT / ".venv")
        finally:
            if previous_temp is None:
                os.environ.pop("TMPDIR", None)
            else:
                os.environ["TMPDIR"] = previous_temp
    # Detect a copied/moved incompatible environment before installing anything.
    subprocess.run([str(python), "-c", "import sys; assert (3,11) <= sys.version_info[:2] <= (3,13)"],
                   cwd=ROOT, env=environment, check=True)
    for command in commands:
        subprocess.run(command, cwd=ROOT, env=environment, check=True)
    diagnostic = (
        "import torch; from aig.experiment import build_model; "
        "print('PyTorch:', torch.__version__, 'CUDA:', torch.cuda.is_available()); "
        "model,_=build_model('learned_filter_cnn'); "
        "assert torch.isfinite(model(torch.rand(1,1,224,224))).all(); "
        + ("assert torch.cuda.is_available(), 'CUDA unavailable: check NVIDIA driver or use --compute cpu'; "
           "print('GPU:', torch.cuda.get_device_name(0)); "
           "assert torch.isfinite(model.cuda()(torch.rand(1,1,224,224,device='cuda'))).all(); "
           if compute != "cpu" else "")
        + "print('Environment ready.')"
    )
    subprocess.run([str(python), "-c", diagnostic], cwd=ROOT, env=environment, check=True)


if __name__ == "__main__":
    main()
