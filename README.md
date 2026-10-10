# AIG-DEC: gradient-based AI-generated image detection

Binary research code for real (`nature = 0`) versus AI-generated (`ai = 1`) images.
The current experiment learns two 3x3 filters, initialized from the existing FFD
implementation. Their responses become **two channels: magnitude and angle**,
followed by the original three-block CNN and 128/64/1 classifier.

This project provides a portable Windows/Linux workflow for environment setup,
dataset download/preparation, training, checkpoint resume, and held-out evaluation.
The default experiment uses ProGAN POC, 20 epochs, AdamW, learning rate 1e-4,
batch size 32, input size 224, and seed 42. These are pilot defaults.

## Start on another PC

Install Git and Python **3.11–3.13** first. For NVIDIA training, install a compatible
NVIDIA driver. Python 3.13 + PyTorch 2.9.1 CUDA 12.8 was tested on the original PC.

The proposed GitHub repository name is `ChananyuKML/aig-dec`; create/upload it using
[docs/GITHUB.md](docs/GITHUB.md) before using this clone command. Substitute the
chosen repository name if different.

```sh
git clone https://github.com/ChananyuKML/aig-dec.git
cd aig-dec
python scripts/run.py pipeline --dataset progan_poc --compute cu128 --epochs 20
```

On Linux, use `python3` if `python` is unavailable. This single pipeline command
creates `.venv`, installs dependencies, downloads the selected source archive,
prepares its splits, checks the model/data, trains, and evaluates `best.pt` on test.
For a CPU-only PC, change `--compute cu128` to `--compute cpu` (training is slower).
No environment activation is required.

## Run individual stages

```sh
python scripts/run.py setup --compute cu128
python scripts/run.py data --datasets progan
python scripts/run.py train --dataset progan_poc --device cuda --check-only
python scripts/run.py train --dataset progan_poc --device cuda --epochs 20
python scripts/run.py evaluate --checkpoint runs/learned_filter_cnn/progan_poc/seed_42/best.pt --device cuda
```

Download/prepare all three sources:

```sh
python scripts/run.py data --datasets cifake progan tiny_genimage
```

The POC subsets require downloading their complete source archives first.
The three archives total approximately **8.65 GiB**, plus extracted images and
preparation manifests. Allow at least 25 GiB free for all three, preferably more.
See [dataset counts and protocols](docs/DATASETS.md) before choosing an experiment.

## Supported models

| `--model` | Input and filters | Purpose |
|---|---|---|
| `learned_filter_cnn` | Raw grayscale; two trainable FFD-initialized filters | New method |
| `fixed_filter_cnn` | Same frontend with frozen filters | Controlled ablation |
| `dual_cnn_128` | Existing fixed magnitude/angle transform | Original implementation baseline |

The frozen frontend shares the new method's numerical safeguards. The original
baseline uses the original transform. Use the first two for the closest comparison
of filter trainability. See [docs/METHOD.md](docs/METHOD.md).

## Results and resume

Each run writes into `runs/<model>/<dataset>/seed_<seed>/`:

- `best.pt`: checkpoint selected by minimum validation BCE.
- `last.pt`: completed-epoch checkpoint with optimizer and random states for resume.
- `config.json`, `environment.json`, `history.csv`, `summary.json`: experiment records.
- `evaluation/<dataset>/test/metrics.json` and `predictions.csv`: held-out results.

```sh
python scripts/run.py train --dataset progan_poc --device cuda --epochs 40 --resume runs/learned_filter_cnn/progan_poc/seed_42/last.pt
```

`--epochs` is the total target epoch count. A fresh run refuses to overwrite an
existing run directory. Change `--seed` or `--run-dir` for another experiment.
Copy the entire run folder when moving a resumable experiment to another PC.

## Project layout

```text
aig/                    supported training/evaluation/data-loading workflow
networks/               model implementations, including learned_filters.py
utils/                  original transforms and experimental helpers
scripts/                portable setup, launcher, upload helper
tools/                  reproducible dataset download/preparation and smoke check
tests/                  dataset-free correctness checks
configs/                small source metadata safe to version in Git
docs/                   setup, datasets, method, experiments, GitHub, migration
legacy/                 preserved older training/prototype/evaluation entry points
dataset/                local datasets and derived manifests (ignored)
runs/                   local experiment outputs (ignored)
local_archive/          preserved PDFs, old figures/cluster scripts (ignored)
```

The PowerShell entry points `train_benchmark.ps1` and `train_local.ps1` are retained
as wrappers around the current trainer. Older Python entry points were moved to
`legacy/`; their original options/checkpoints are described in
[docs/MIGRATION.md](docs/MIGRATION.md).

## Checks and documentation

```sh
python scripts/run.py test
python scripts/run.py train --dataset cifake_poc --device cuda --check-only
```

For the synthetic end-to-end check (train, resume, evaluate, and overwrite guards),
run `.venv/bin/python tools/check_workflow.py --device cpu` on Linux or
`.\.venv\Scripts\python.exe tools/check_workflow.py --device cuda` on Windows.
Temporary test checkpoints are removed afterward; this does not train a benchmark run.

[SETUP](docs/SETUP.md) · [DATASETS](docs/DATASETS.md) · [METHOD](docs/METHOD.md) ·
[EXPERIMENTS](docs/EXPERIMENTS.md) · [GITHUB](docs/GITHUB.md) · [MIGRATION](docs/MIGRATION.md)

Datasets, environments, model weights, private notes, and outputs are excluded from
Git. No code license has been assigned by this preparation; choose one as the
research owner before a public release. Dataset licenses are independent of the code.
