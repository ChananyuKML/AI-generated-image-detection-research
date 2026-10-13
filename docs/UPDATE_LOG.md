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

## 2026-10-13 — Full-only comparison with live progress on RTX 5060 Ti

The owner requested full training with progress visible during training and
explicitly excluded POC dataset experiments. Prepared a 15-run queue: all five
supported models on the full `progan`, `tiny_genimage`, and `cifake` splits, seed 42.
Settings are 224px, AdamW at 1e-4, batch 32, workers 4, full precision, a 100-epoch
maximum and validation-BCE patience 10/min_delta 0. KC projection seed is 42;
each KC variant starts from that dataset's same validation-selected fixed encoder.

- Added `aig/progress.py` and observational batch reporting to the existing trainer.
  It writes phase/batch/loss/accuracy, throughput, phase ETA and completed-epoch
  telemetry without changing model weights, optimizer behavior, or RNG state.
- Added `tools/run_full_training.py`: full-only inventory/signature verification,
  source/package snapshots, encoder dependency ordering, completed-run reuse,
  checkpoint recovery, stopping-boundary recovery, manager lock, and automatic
  held-out evaluation/result CSV. Windows sleep inhibition is requested while active.
- Added a local live dashboard showing actual child-process liveness/PID, queue,
  current batch, completed-epoch curves, GPU/VRAM, logs, and completed test metrics.
  It binds to 127.0.0.1 and reports disconnection rather than treating saved status
  as proof of an active process. Phase ETA is not a whole-queue completion forecast.
- All **35 unit tests** passed, including CUDA checks and new scheduling, recovery,
  process-observation, and telemetry weight/RNG-equivalence checks. The CPU
  learned-filter workflow also passed train/resume/evaluate, exact epoch-boundary
  equivalence, and the existing early-stopping recovery/guard checks.
- Verified the actual full image inventories against CSV/JSON manifests:
  ProGAN 8,000 images; Tiny GenImage 34,998; CIFAKE 117,328. No POC splits are queued.
- Python syntax, dashboard JavaScript syntax, and Git whitespace checks passed.
- This PC is Windows/Python 3.13.5, torch 2.9.1+cu128, NVIDIA RTX 5060 Ti with
  16 GiB VRAM. CUDA allocation succeeded. This is a new local comparison;
  historical results in earlier entries belong to the other execution machine.

Preparation verified at **10:53:50 Asia/Bangkok (UTC+7)**. Training has not yet
started at this preparation snapshot. Intended local output is
`runs/full_comparison_20261013/`, with its live view at http://127.0.0.1:8765/.
Source/script changes are local; the owner still handles GitHub publishing.

### Launch, Windows reporting recovery, and live snapshot — 11:08:31 Asia/Bangkok (UTC+7)

Started the full-only queue on this PC at approximately **10:55**, with the source
snapshot in `runs/full_comparison_20261013/source/`. The first stage trained
`fixed_filter_cnn` on full ProGAN and saved ten completed epochs. Windows venv
launchers expose a different PID from the actual trainer; added a fresh-telemetry
PID check and a read-only viewer at http://127.0.0.1:8766/ without interrupting the
initial training process.

At the epoch-11 reporting boundary, a Windows reader-sharing conflict prevented
replacing `progress.json` and the initial trainer exited. The epoch-10 `last.pt`,
validation-best checkpoint, optimizer/random states and history were preserved.
Fixed JSON replacement with bounded retry and made optional progress IO failures
nonfatal to training. Added regression tests; all **39 unit tests passed**.

Stopped only the verified failed queue manager, preserved the original source as
`source_v1/` with `source_files_v1.json`, and created a new source snapshot for the
reporting repair. Original source diffs, package records, failure status and logs
remain in the batch folder. No dataset/model/optimizer settings were changed.

The queue resumed from epoch 10 with three carried bad epochs. At **11:08:31**,
the actual manager and trainer were verified running, epoch 11 had completed,
and epoch 12 was validating. The selected checkpoint was still epoch 7 with
validation BCE **0.42013**. No new local full-run test evaluation had completed
at this snapshot; the other 14 model/dataset stages remained queued. This is live
training progress, not a test-set result or a claim that all 15 runs are finished.

Check `status.json`, current child liveness, `progress.json` and the live log before
reporting a newer status. The dashboard and automatic queue continue in background;
source changes remain local and no GitHub push was performed.

### First completed full run and automatic queue advance — 11:12:46 Asia/Bangkok (UTC+7)

Full ProGAN `fixed_filter_cnn`, seed 42, completed **17 / 100 requested epochs**
through validation patience. The selected checkpoint is epoch **7**, validation
BCE **0.42013**. Held-out evaluation on **1,200 test images** measured accuracy
**80.17%** and ROC-AUC **0.8904** on this RTX 5060 Ti execution. The results are in
`runs/full_comparison_20261013/results.csv` and the run's evaluation folder.

The queue automatically started full ProGAN `learned_filter_cnn`. At this recorded
time its actual child and manager were alive, two epochs had completed, and epoch
3 was training. One of 15 runs had completed, one was active, and 13 were queued.
The read-only live view at http://127.0.0.1:8766/ was verified in the in-app browser
and left open. Curves, batch telemetry and the completed-run table were visible.
The original source snapshot was independently hash-verified after preservation.

The full queue continues in background. These values are a dated snapshot; use
the live process view and current artifacts for newer progress. No claim is made
that all full comparisons are finished or will finish by a particular deadline.

## 2026-10-13 — Approved GenImage-style 7×7 Tiny testing

The owner requested following GenImage testing instructions and explicitly selected
the **full 7×7 training-generator × testing-generator matrix for each model**.
Checked the official repository and paper section 3.2: the published experiment
uses eight generators/64 cells per method. This Tiny release contains seven and
lacks SD1.4, so the implemented protocol is a clearly labelled subset adaptation.

- Created seven `tiny_protocol_<generator>` datasets from the verified full Tiny
  source archive/index. All supplied val images are reserved as final test: 500
  real plus 500 AI per generator, 7,000 total. Selection validation comes only
  from original training groups, with deterministic 10% allocation/class balancing.
  No POC sampling caps are used. Original image bytes and source provenance remain.
- Grouped the shared ADM/Midjourney real training duplicate globally. Audited
  train/selection-val/test identities across all source/target pairs: no exact
  cross-role overlap. Near duplicates remain unaudited. Existing prepared exports
  and run outputs were preserved; these new signatures prevent incompatible reuse.
- Added single-checkpoint/seven-target evaluation, predictions, per-target metrics,
  source-row summaries, 7×7 accuracy/AUC matrices, column means and Table-4-style
  equal-cell macro accuracy. Diagonal and 42-cell off-diagonal means are separate;
  incomplete final averages remain blank. Test labels do not tune selection/threshold.
- The revised schedule has 35 Tiny source detectors and 245 test cells, plus the
  existing five ProGAN and five CIFAKE stages: **45 training runs for seed 42**.
  Model architecture, optimizer, batch 32/workers 4, 100-epoch maximum, validation
  patience 10, and matching source-specific KC encoder initialization are unchanged.
- All **43 unit tests** passed. A synthetic code-only workflow trained one source
  checkpoint and successfully evaluated all seven target fixtures, verifying one
  checkpoint hash, predictions, provenance and macro aggregation. Synthetic scores
  are not research results. Protocol Python/dashboard JS and Git diff checks passed.

At **12:27:46 Asia/Bangkok (UTC+7)**, ProGAN `finetuned_cnn_kcnet` was still running
in epoch 14 with 13 completed epochs. The approved switch coordinator was verified
alive, waiting for the Tiny boundary. It preserves this ProGAN work, then replaces
the upcoming pooled Tiny stage with the new per-generator plan/source revision.
The current 15-stage process is not claimed to have already switched at this snapshot.
Its control record is `runs/full_comparison_20261013/genimage_protocol_switch.json`.

The live view has a selectable matrix panel. New Tiny protocol runs and final
matrix scores have not completed at this recorded time. See
[GENIMAGE_TESTING.md](GENIMAGE_TESTING.md) for preparation, execution, averages,
limitations, and transition details. Source changes remain local; no GitHub push.
