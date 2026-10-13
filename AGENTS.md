# Project guidance for Codex

## Research context

Repository: `ChananyuKML/AI-generated-image-detection-research`.
This is research on binary detection of real versus AI-generated images. The current
experiment replaces fixed gradient filters with trainable filters initialized from
the existing FFD implementation. The owner's current focus is binary classification;
prototype (`_proto`) experiments are deferred.

The current training scope is the full prepared `progan`, `tiny_genimage`, and
`cifake` datasets with all five supported models. Do not schedule POC variants for
the active comparison. Synthetic data remains appropriate for code-only tests.

Follow the current user's request when it changes this scope. Preserve unrelated
work and inspect `git status` before editing. The owner currently handles GitHub
uploads manually; prepare local changes for pushing unless asked to publish.

## Update log

Read [docs/UPDATE_LOG.md](docs/UPDATE_LOG.md) before starting repository work.
It records the latest implementation changes, completed checks, measured results,
and unfinished experiments. Treat run status as a dated snapshot; check current
logs/checkpoints and the live process before reporting that training is running
or complete. A saved `status: running` alone does not prove a process is active.

After meaningful code, dataset, experiment, or workflow changes, append a dated
entry to that log. Include what changed, what was actually checked, results with
dataset/model/seed and completed versus requested epochs, and remaining work or
machine-specific limits. Give an explicit timezone for progress snapshots.
Preserve previous entries and keep generated datasets, checkpoints, manifests,
and full run outputs outside Git; record their relative local paths instead.

## Where to work

| Location | Purpose |
|---|---|
| `train.py`, `evaluate.py` | Supported binary command-line entry points |
| `aig/training.py`, `aig/evaluation.py` | Training, resume, and held-out evaluation |
| `aig/experiment.py` | Model selection, strict image loading, paths, metrics |
| `networks/learned_filters.py` | New frontend and learned/frozen CNN models |
| `networks/hog_cnn.py`, `utils/custom_transforms.py` | Existing classifier and transforms |
| `scripts/run.py`, `scripts/setup_env.py` | Portable launcher and environment setup |
| `tools/download_benchmarks.py`, `tools/prepare_benchmarks.py` | Source verification and dataset preparation |
| `tests/`, `tools/check_workflow.py` | Unit tests and synthetic workflow checks |
| `tools/run_full_training.py`, `tools/training_dashboard.html` | Full-only resumable queue and live process dashboard |
| `aig/progress.py` | Batch telemetry that must not change model weights or RNG state |
| `docs/UPDATE_LOG.md` | Dated implementation, validation, and experiment updates |
| `legacy/` | Preserved older experiments; not the supported workflow |

Use `docs/METHOD.md` for architecture/filter changes, `docs/DATASETS.md` for dataset
work, `docs/EXPERIMENTS.md` for run/resume behavior, and `docs/MIGRATION.md` for legacy
compatibility. Read only the documentation relevant to the task. `docs/VALIDATION.md`
records checks on the original PC, not proof of the destination PC's environment.

## Model and experiment conventions

- Keep **two channels: magnitude and one angle channel**. Do not silently change
  to magnitude plus sine/cosine, add GroupNorm, or redesign the classifier.
- The original supported models are `learned_filter_cnn`, `fixed_filter_cnn`, and `dual_cnn_128`.
  The first two receive raw grayscale with `gray_center_224`; the original baseline
  receives fixed gradient maps with `raw_grad_224`. Applying a gradient transform
  before the learned frontend would filter the input twice.
- `fixed_filter_kcnet` is a separate supported head experiment: frozen pretrained
  fixed-filter frontend/CNN, 4x4 pooling, per-image L2 normalization, fixed sparse
  random projection and a BCE-trained readout. Read `docs/METHOD.md` and
  `docs/EXPERIMENTS.md` for its additional architecture and pretraining changes.
  Fresh runs require a validation-selected `--encoder-checkpoint` matching the
  dataset signature; resume/evaluation use embedded weights and projection indices.
- `finetuned_cnn_kcnet` uses the same KC architecture and encoder initialization,
  but trains the CNN and readout. Keep FFD filters and random projection fixed.
  For a matched comparison, initialize from the same original fixed-filter encoder,
  use the same fresh head/projection seeds and match the frozen KC run's settings.
- Labels are `nature = 0` and `ai = 1`. Models output a single logit; training uses
  BCEWithLogitsLoss. Classification uses AI probability strictly greater than 0.5.
- The learned/frozen pair shares the same CNN, FFD initialization, normalization,
  and numerical safeguards. Only filter trainability differs. Keep comparisons
  controlled and document any additional change.
- Current kernels match the original code. Its vertical kernel convention and
  `atan2(Gy, Gx)` differ from the paper's written FFD/arctan expressions. Do not
  silently change these when refactoring; explain a scientific-method change.
- Preserve finite forward/backward behavior at zero and tiny gradient responses,
  per-image/channel normalization, and checkpointed numerical thresholds.
- Pilot defaults: 224px input, AdamW, learning rate 1e-4, batch size 32, 20 epochs,
  training seed 42, and workers 0. These are not claimed published-paper settings.
- Select checkpoints by validation BCE. Never select a checkpoint or tune a
  threshold using test labels. Keep test held out for evaluation.
- Optional `--patience`/`--min-delta` early stopping also uses validation BCE.
  Preserve monitor state across resume, including reconstruction from older
  checkpoints. `best.pt` remains the absolute-best BCE checkpoint; summaries must
  report actual completed epochs separately from the maximum epoch target.

## Setup and commands on another PC

For the active full-only comparison, follow [docs/FULL_TRAINING.md](docs/FULL_TRAINING.md).
The queue trains a matching fixed-filter encoder before each dataset's KC models,
snapshots the source, verifies manifests, and evaluates held-out test only after
training finishes. Verify child PID/liveness and telemetry before reporting progress.

Use Python 3.11–3.13. Recreate `.venv` on the destination; do not copy it across PCs.
Run these from the repository root (`python3` may be needed on Linux):

```sh
python scripts/run.py setup --compute auto
python scripts/run.py data --datasets progan
python scripts/run.py train --dataset progan_poc --device auto --check-only
python scripts/run.py train --dataset progan_poc --device auto --epochs 20
python scripts/run.py evaluate --checkpoint runs/learned_filter_cnn/progan_poc/seed_42/best.pt --device auto
```

Setup supports CPU and NVIDIA CUDA builds. `--compute cu128` is the original tested
GPU configuration; `--compute cpu` is the CPU option. Auto detection does not prove
driver compatibility. Check the actual destination environment before training.
The launcher selects project Python without requiring environment activation.

For code changes, use:

```sh
python scripts/run.py test
```

For changes to training, checkpointing, resume, or evaluation, also run the relevant
synthetic workflow check with project Python:

```powershell
# Windows
.\.venv\Scripts\python.exe tools/check_workflow.py --device cpu --models learned_filter_cnn
```

```sh
# Linux
.venv/bin/python tools/check_workflow.py --device cpu --models learned_filter_cnn
```

Use `--device cuda` when checking CUDA behavior on a supported GPU. Omit `--models`
to cover all five supported models. Documentation-only changes need content/link
and Git diff checks, not model training. Run benchmark training when it is part of
the user's request; do not start long runs just to validate an unrelated code edit.

## Data and run handling

- Dataset layout: `dataset/prepared/<name>/{train,val,test}/{nature,ai}/`. Both classes
  must exist in each used split. PNG/JPG/JPEG files can be nested. Use `--data-root`
  for an external data root; keep code portable and avoid personal absolute paths.
- Available sources: CIFAKE, ProGAN/CNNDetection validation archive, and community
  Tiny GenImage. POC variants still require their full source archive download.
  Existing prepared variants can instead be copied with their manifests.
- Preparation uses seed 42, source SHA-256 checks, decoded-pixel duplicate grouping,
  balanced exports, and provenance manifests. Near duplicates were not audited.
  Do not remove integrity checks to accept an unexpected source release.
- CIFAKE is native 32px. ProGAN uses a custom split of a validation archive. Tiny
  GenImage is a community subset missing SD1.4. Report these as pilot derivatives;
  do not claim they reproduce the complete official benchmark protocols.
- Related full/POC/per-generator variants overlap; they are not independent sources.
  Maintain split separation and provenance when adding or combining datasets.
- Runs live in `runs/<model>/<dataset>/seed_<seed>/`. Preserve overwrite protection.
  `best.pt` is validation-selected; `last.pt` contains optimizer and random states.
  `--epochs` is the total target when resuming. Transfer the complete run folder.
- Keep experiment configuration, environment, manifests, history, and metrics.
  Report real measurements; synthetic smoke-test accuracy is not research evidence.

## Repository hygiene and compatibility

- Keep `.venv/`, datasets, generated manifests, `runs/`, weights, `output/`, `tmp/`,
  and `local_archive/` out of Git. Preserve these local artifacts during cleanup.
- Preserve the supported Windows/Linux launcher and existing PowerShell wrappers.
  Update relevant documentation when command behavior changes.
- Current structured `.pt` checkpoints differ from legacy raw `.pth` state dicts.
  Do not mix loaders or rename model state keys without a compatibility strategy.
- Prototype, anomaly, and older transformer/loader paths have unresolved issues.
  Work on them when requested, without routing the supported binary path through
  them or assuming every archived entry point works unchanged.
- Keep dataset licenses separate from the code. No code license was selected;
  do not invent a license or research claims for a public release.
- Finish with what changed, what was checked, and any remaining machine-specific
  limitation. Do not describe Linux/CPU-wheel installation or GitHub CI as executed
  merely because scripts and workflow files were created.
