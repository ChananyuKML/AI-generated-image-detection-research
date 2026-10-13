# Research update log

Read this log together with [AGENTS.md](../AGENTS.md) before changing the project.
Append dated entries after meaningful implementation, dataset, validation, or
experiment work. Progress entries describe the recorded time, not live state.
Verify the current logs, checkpoints, and process before reporting a run's status.
All results below are local measurements from seed 42, not published benchmark
reproductions. Raw datasets, manifests, weights, and full run outputs stay outside Git.

## 2026-10-12 — KCNet-inspired model and ProGAN POC experiments

### Implementation and method

- Researched FlyHash, KCNet, and FlyNet and chose a KCNet-inspired CNN feature head
  for binary real/AI detection. This adapts KCNet; it does not reproduce its
  ridge-regression readout or published benchmark protocol.
- Added `networks/kcnet.py` and the supported models `fixed_filter_kcnet` and
  `finetuned_cnn_kcnet`. The first freezes the pretrained CNN and trains only the
  readout; the second trains the CNN and readout. Both keep the FFD filters and
  random projection fixed.
- Kept the original magnitude plus one `atan2` angle channel, FFD kernel convention,
  grayscale input, and numerical safeguards. The KC path is:

  ```text
  grayscale 224x224 -> fixed FFD magnitude/angle -> CNN 32/64/128
    -> 4x4 average pooling -> 2048 features -> per-image L2 normalization
    -> 4096 fixed random units, 7 distinct inputs each
    -> subtract per-image mean response -> ReLU -> binary linear readout
  ```

- Default KC models contain 97,075 neural parameters. The frozen variant trains
  4,097; CNN fine-tuning trains 97,057. Random connections are checkpointed indices,
  with projection seed 42. ReLU activity is input-dependent, not a forced top-5% code.
- Fresh KC runs require a validation-selected `fixed_filter_cnn` encoder from the
  same dataset/signature. Encoder provenance, weights, projection indices,
  architecture, and numerical thresholds are embedded for resume/evaluation.
- Updated training, evaluation, launcher, PowerShell wrappers, tests, documentation,
  and the CPU CI workflow for the two additional supported models.

See [METHOD.md](METHOD.md) and [EXPERIMENTS.md](EXPERIMENTS.md) for exact conventions
and commands. The design record is local at `runs/designs/fly_head_v1/`.

### Completed POC measurements

ProGAN POC contains 2,800 training, 600 validation, and 600 test images. Runs use
224px input, AdamW at 1e-4, batch 32, workers 0, and training seed 42. Checkpoints
are selected by minimum validation BCE; AI probability must be strictly greater
than 0.5. The KC variants start from the same fixed-filter encoder selected at epoch 12.

| Model / stage | Completed / requested epochs | Best epoch | Test accuracy | ROC-AUC | Test BCE |
|---|---:|---:|---:|---:|---:|
| `learned_filter_cnn` | 20 / 20 | 10 | 64.17% | 0.7057 | 0.63483 |
| `fixed_filter_cnn` | 20 / 20 | 12 | 77.50% | 0.8680 | 0.46293 |
| `fixed_filter_kcnet` | 20 / 20 | 20 | 70.67% | 0.7613 | 0.62847 |
| `finetuned_cnn_kcnet`, initial run | 20 / 20 | 19 | 86.17% | 0.9371 | 0.33795 |

Runs are local at `runs/<model>/progan_poc/seed_42/`. KC training is additional
training after encoder pretraining and also changes pooling and the classifier;
comparison with the original CNN does not isolate the head alone. The two KC
20-epoch runs match encoder initialization, fresh readout/projection seeds, and
optimizer settings, with CNN trainability as their difference.

### Early stopping and extended fine-tuning

- Added `aig/early_stopping.py`, `--patience`, and `--min-delta`, monitored only on
  validation BCE. Ties count against patience; small decreases accumulate against
  the last significant improvement. `best.pt` independently remains the absolute
  minimum-BCE checkpoint.
- Resume preserves monitor state and reconstructs it from older checkpoint history.
  Exhausted patience rejects more training unless explicitly changed or disabled.
  Summaries distinguish completed epochs from the requested maximum.
- Preserved the original 20-epoch KC fine-tuning folder at
  `runs/finetuned_cnn_kcnet/progan_poc/seed_42_epoch_20_snapshot/`.
- Resumed `last.pt` after epoch 20 with a total target of 100, patience 10, and
  min_delta 0. Training stopped at epoch 57; the selected checkpoint is epoch 47.
  Validation BCE was **0.25691**; on 600 test images, accuracy was **88.00%**,
  ROC-AUC **0.9505**, and BCE **0.29047**. This is a continuation of the same run.

### Completed validation and environment

- All **30 unit tests** passed, including the CUDA CNN-gradient check.
- Synthetic CPU train/resume/evaluate checks passed for all five supported models,
  including exact epoch-boundary resume equivalence and KC restore without the
  original encoder file.
- Early-stopping workflow checks covered the exact stopping epoch, legacy counter
  reconstruction, carried patience, absolute-best selection, exhausted-patience
  rejection, and explicit disabling.
- CUDA ProGAN preflight and the updated Windows wrapper preflight passed.
- Current execution machine: Windows, Python 3.13.9, PyTorch 2.9.1+cu128,
  torchvision 0.24.1+cu128, NVIDIA RTX 3050 Ti Laptop GPU with 4 GiB VRAM.
  This differs from the historical machine in [VALIDATION.md](VALIDATION.md).
- CPU execution used the installed CUDA wheel. Linux execution, a CPU-only wheel
  installation, and GitHub Actions were not executed on this machine.

Detailed check logs and source snapshots remain in `tmp/logs/` and the local run
folders. Synthetic accuracy is workflow evidence, not research performance.

## 2026-10-13 — All sources prepared; full prepared-dataset batch

### Data and protocol

Downloaded and verified all three pinned source archives by byte count and SHA-256.
Preparation decoded the images, grouped exact pixel duplicates before splitting,
balanced exports, and wrote provenance manifests with preparation seed 42.
Full, POC, and all seven Tiny GenImage per-generator variants are prepared.

| Full prepared dataset | Train | Validation | Test |
|---|---:|---:|---:|
| `progan` | 5,600 | 1,200 | 1,200 |
| `cifake` | 87,664 | 9,740 | 19,924 |
| `tiny_genimage` | 27,998 | 3,500 | 3,500 |

CIFAKE is native 32px and resized to 224px. ProGAN uses a custom split of the
released validation archive. Tiny GenImage is the seven-generator community
subset and excludes SD1.4. These are cleaned pilot derivatives, not the complete
official benchmark protocols. Full/POC/per-generator views overlap; they are not
independent sources. Near duplicates were not audited. See [DATASETS.md](DATASETS.md).

The full batch trains `fixed_filter_cnn` and then `finetuned_cnn_kcnet` separately
on each of the three full prepared variants: six training stages. Each has seed
42, AdamW at 1e-4, batch 32, workers 0, 224px input, a 100-epoch maximum, patience
10, and min_delta 0. Each KC stage copies that dataset's validation-selected encoder
and starts with a fresh readout and projection seed 42. The two stages have separate
epoch budgets. Test evaluation follows training; test labels do not select checkpoints.

### Completed full ProGAN results

| Model | Completed / requested epochs | Best epoch | Validation BCE | Test accuracy | ROC-AUC | Test BCE |
|---|---:|---:|---:|---:|---:|---:|
| `fixed_filter_cnn` | 17 / 100 | 7 | 0.42650 | 80.17% | 0.8938 | 0.42141 |
| `finetuned_cnn_kcnet` | 43 / 100 | 33 | 0.14575 | 93.92% | 0.9857 | 0.15122 |

Both stopped through validation-BCE patience and were evaluated on 1,200 test images.
The KC result includes its additional fine-tuning stage after encoder pretraining.
POC and full results use different, overlapping test views and are not independent
cross-dataset evidence. Only one training seed has been measured here.

### Training recovery and progress snapshot — 09:55, Asia/Bangkok (UTC+7)

The previous queue and training process were confirmed no longer running. The saved
queue status incorrectly still said `running`; the cause of interruption was not
established. CIFAKE had completed 14 fixed-filter epochs, with its best validation
BCE **0.24456** at epoch **7** and seven carried bad epochs. No CIFAKE test result
or KC fine-tuning result was available at this snapshot; Tiny GenImage training
had not started.

Restarted the queue in a hidden local process after confirming the old processes
had exited. Completed ProGAN stages are reused. CIFAKE resumes from its own
`last.pt`, restoring optimizer/random states and patience rather than restarting
the experiment. Tiny GenImage training remains queued. Confirm the current stage
from live artifacts before treating this snapshot as current progress.

The queue, logs, source snapshot/patch, requirements, dataset manifests, and evolving
results are local under `runs/full_datasets_20261012/`. Its helper `run_all.py` is
an ignored local orchestration artifact, not a command shipped by the source repo.
Source-repo training/resume commands remain in [EXPERIMENTS.md](EXPERIMENTS.md).

## 2026-10-13 — Update-log guidance and GitHub preparation

- Added this tracked update log and linked it from the README.
- Updated `AGENTS.md` to require reading the log before repository work and appending
  dated factual updates after meaningful changes. Live process verification is
  required before describing saved run status as active training.
- Prepared the KC models, early-stopping implementation, tests, launcher/wrapper/CI
  changes, and related documentation for a local source-only commit on `main`.
  Datasets, checkpoints, environments, generated manifests, and full run outputs
  remain ignored and must be transferred separately.
- Verified the configured GitHub `origin` main reference matched the local base
  `199870d` before commit preparation. The owner still performs the actual GitHub push.

### Preparation checks

- Re-ran all **30 unit tests** successfully, including CUDA checks. The local log
  is `tmp/logs/github_update_unit_tests_20261013.log`.
- Parsed 41 Python source files and checked 29 local Markdown links successfully.
- Verified nine model/workflow/test implementation files exactly match the source
  snapshot used for the previously completed all-five-model CPU workflow checks.
  That workflow was not repeated for this documentation update.
- Confirmed completed full ProGAN metrics use the minimum-validation-BCE checkpoint.
- Reviewed 61 source-only files; generated artifacts remain ignored. Git whitespace
  checks passed. No Linux execution or GitHub CI execution is claimed.
- Confirmed CIFAKE resumed after epoch 14 with seven carried bad epochs. The
  background queue survived its launcher and the resumed training log was verified.

The intended local commit is `Add KCNet models, resumable early stopping, and research update log`.
Review the local commit and preview the configured destination from the repository
root before the owner performs the push:

```sh
git log -1 --oneline
git show --stat HEAD
python scripts/publish_github.py
git push origin main
```

Only the final command publishes. The preparation itself does not push to GitHub.
