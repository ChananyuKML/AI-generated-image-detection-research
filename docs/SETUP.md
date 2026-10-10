# Environment setup on another PC

## Windows

Install Git and Python 3.11, 3.12, or 3.13. Open PowerShell in the cloned repository.
The tested PC uses Python 3.13.5, NVIDIA RTX 5060 Ti, and PyTorch 2.9.1 + CUDA 12.8.

```powershell
python scripts/run.py setup --compute cu128
python scripts/run.py data --datasets progan
python scripts/run.py train --dataset progan_poc --device cuda --check-only
python scripts/run.py train --dataset progan_poc --device cuda --epochs 20
```

If `python` selects an unsupported interpreter, use `py -3.13` in place of `python`.
The launcher uses the project's `.venv` automatically after setup. It works from
any working directory when invoked with the full path to `scripts/run.py`.

The optional PowerShell shortcut uses the same trainer:

```powershell
powershell -ExecutionPolicy Bypass -File .\train_benchmark.ps1 -Dataset progan_poc -Network learned_filter_cnn -Epochs 20
```

## Linux

Install Git and Python 3.11–3.13 with its venv support (on Ubuntu this may require
the matching `python3-venv` package). Open a terminal in the cloned repository.

```sh
python3 scripts/run.py setup --compute cu128
python3 scripts/run.py data --datasets progan
python3 scripts/run.py train --dataset progan_poc --device cuda --check-only
python3 scripts/run.py train --dataset progan_poc --device cuda --epochs 20 --workers 4
```

The launcher runs `.venv/bin/python` on Linux and `.venv/Scripts/python.exe` on
Windows. Do not copy `.venv` between PCs; recreate it on the destination.

## Compute choices

| Setup option | Intended PC |
|---|---|
| `--compute auto` | Chooses cu128 when `nvidia-smi` reports a GPU, otherwise CPU |
| `--compute cu128` | Tested NVIDIA configuration; recommended for RTX 50 series |
| `--compute cu126` / `cu130` | Alternative official builds for other driver/hardware configurations |
| `--compute cpu` | CPU-only setup and CI |

Setup uses the official [PyTorch 2.9.1 installation commands](https://pytorch.org/get-started/previous-versions/#v291)
with torchvision 0.24.1. CUDA availability is checked after installation. The wheel
supplies its CUDA runtime; a compatible NVIDIA driver is required. Install a local
CUDA toolkit only if another part of your work needs it.

Auto detection cannot guarantee driver or GPU compatibility. Select a build
explicitly if setup fails. This repository's training CLI supports CPU and NVIDIA
CUDA. macOS can use the CPU setup; Apple MPS and AMD ROCm training were not validated.

## Dependency records

`requirements.txt` pins the torch/torchvision pair and gives bounded ranges for
direct dependencies so pip can select wheels for the destination Python/OS.
`requirements/windows-cu128.lock.txt` preserves the exact original Windows/Python
3.13 environment. Use the portable setup for normal installation.

To record the exact versions used by a new experiment:

```powershell
.\.venv\Scripts\python.exe -m pip freeze > installed-requirements.txt
```

On Linux, use `.venv/bin/python -m pip freeze > installed-requirements.txt`.
Keep this record with the experiment when comparing results across machines.

## Troubleshooting

- **CUDA unavailable:** run `nvidia-smi`; check the installed driver and chosen
  PyTorch build. Use `--compute cpu` and `--device cpu` for a CPU-only workflow.
- **GPU out of memory:** reduce `--batch-size 32` to 16 or 8. These CNNs still have
  approximately 12.9 million parameters because of the flattened classifier.
- **Windows worker errors:** keep `--workers 0` initially; raise it after a successful run.
- **Missing data:** run the data stage or point `--data-root` at a transferred dataset root.
- **Run directory exists:** resume from `last.pt`, change the seed, or give a new `--run-dir`.
- **Interrupted download:** repeat the data command; `.zip.part` downloads resume
  when the source supports ranges. A completed ZIP must match the recorded snapshot.
- **Source size/checksum changed:** the download stops to preserve reproducibility.
  Review the release before updating the size/hash constants in the downloader.

Run `python scripts/run.py setup --compute cu128 --dry-run` to inspect the install
commands without modifying the environment.
