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

## 2026-10-13 — Presentation accuracy tables by GenImage generator

The owner requested collecting per-generator test accuracy in the layout of the
research paper. Inspected and rendered its results page: Table III has model rows
and test-generator columns, with each column averaging the training sources. Its
Table II instead uses SD1.4 as the single training source; that source is absent
from Tiny GenImage and is not substituted silently.

- Added `aig/genimage_reporting.py` and `tools/export_genimage_tables.py`. They
  export percentage tables with two decimals to the ignored batch's
  `presentation/genimage_accuracy_tables.md` and `.tex`: a model-by-test-generator
  summary, every model's full 7×7 matrix, row averages, diagonal and off-diagonal
  averages. Column means require seven cells and the overall mean requires 49.
  Missing scores stay Pending; no published or synthetic scores are inserted.
- The full queue exports these tables after source evaluation. A read-only
  `--watch` exporter also follows the existing active manager without restarting
  training. It reads the saved matrix JSON, performs no inference, and exits once
  all planned Tiny matrices are complete. Existing CSV values remain fractions.
- Eight relevant unit tests passed (three reporting tests and five protocol tests),
  covering source/target orientation, percentage conversion, zero versus missing,
  incomplete averages, and invalid/duplicate matrix rejection. Initial tests hit
  sandbox permissions in the system temporary directory; moved test fixtures into
  the project's ignored `tmp/reporting_tests/` and reran successfully. Python syntax
  and Git whitespace checks passed. No benchmark was run just to test formatting.

At **12:46:40 Asia/Bangkok (UTC+7)**, the protocol switch was complete and the
actual manager/child were verified alive through the local API. The active source
run was `tiny_protocol_midjourney/fixed_filter_cnn`, seed 42, with two completed
epochs and epoch 3 active. There were **0/245 measured Tiny matrix cells** at this
snapshot. All table values were therefore Pending. The table observer's actual
Python process was verified alive (PID 40436; Windows launcher PID 110172), with
an empty error log. See its logs and `presentation/export_status.json` in
`runs/full_comparison_20261013/`. Training continues independently. These are dated
observations; inspect current artifacts/processes for a newer completion claim.

Architecture, training settings, dataset splits and checkpoints were unchanged.
Source/exporter changes are prepared locally for the owner's manual GitHub push.

## 2026-10-13 — English research progress presentation

The owner requested presentation slides while the training queue continues and
selected **10–15 minutes, English**. Created a 16-slide editable PowerPoint:
13 main slides with approximately 13 minutes of speaking notes, plus three backup
slides covering exact Tiny splits, KC architecture, and references. The final
local artifact is `output/presentations/research_progress_20261014_final.pptx`.
Build assets, the immutable data snapshot, drafts and validation records are under
ignored `tmp/presentation_build_20261013/`. Generated artifacts remain outside Git.

- Covers the original paper baseline, the trainable 18-coefficient/two-channel
  frontend, five models, current dataset splits, the seven-generator testing
  protocol, training settings, a validation learning curve, completed ProGAN tests,
  measured Midjourney-source tests, interpretation and next research steps.
- Keeps prior full-GenImage paper results separate from this local subset run.
  Describes the atan2/vertical-kernel implementation difference and the KC head's
  additional architecture/pretraining budget. Unfinished scores stay Pending.
- Data snapshot: **14:40 Asia/Bangkok (UTC+7), 13 October 2026**. The live API
  verified actual manager/child liveness at capture. Nine of 45 stages were
  complete, including all five ProGAN runs and four Midjourney detectors; 28 of
  245 Tiny test cells were recorded. Midjourney `finetuned_cnn_kcnet` was active.
  The deck is a static dated snapshot, not an automatically updating result file.
- Used the bundled Artifact Tool runtime with native editable tables, one native
  learning-curve chart and an embedded chart-data workbook. Chart values are
  rounded to six decimals for spreadsheet precision; source histories retain full
  precision. All 16 slides have speaker notes with sources and main-slide timing.
- All 16 final slides rendered successfully. Individually reviewed the slides,
  fixed a chart-axis range and header wrapping, then checked the final renders.
  Package integrity, layout/font checks (zero warnings), first-party import, and
  editable chart/workbook validation passed. Independently compared every displayed
  ProGAN/Midjourney accuracy and both 25-epoch chart series against the saved source
  snapshot. Native PowerPoint application opening was not performed.

Saved and checked at **14:54 Asia/Bangkok (UTC+7)**. No training process, model,
batch size, optimizer setting, dataset split or checkpoint was changed. Source
documentation changes remain local for the owner's manual GitHub push.

## 2026-10-13 — Progress deck matched to enabled Canva reference pages

The owner requested Canva styling from the previous progress-report deck and
specified the enabled pages: **1, 3, 36–54, 68–73, 83–85, 90–91, 95–96**.
Inspected all 34 selected page previews and worked with a separate reference copy;
the original 186-page Canva deck was not edited. The owner identified the diagrams
as draw.io screenshots, and the enabled page-36 overview image is reused on the
local method slide.

- Created the local 16-slide revision
  `output/presentations/research_progress_20261014_canva_style_v2.pptx`.
  It follows the white canvas, black section headings, teal topic headings,
  gray table headers/first columns, and green best-result emphasis from the
  enabled method and results pages. The cover uses the authentic source artwork
  and KU logo, with editable new text. The original font family was unavailable
  through the connector; the local reconstruction uses Arial and does not claim
  an exact font match.
- Preserved the existing 14:40 Asia/Bangkok (UTC+7), 13 October 2026 data snapshot.
  All native table values and chart series match the previous deck exactly.
  Ten editable tables, one native chart with embedded data workbook, and all
  16 speaker-note slides remain. Pending scores stay Pending.
- Rendered all 16 exported slides and individually reviewed them, including
  a repaired results-caption overlap and the clearer separate KU logo.
  After repairs, re-rendered all slides, compared unchanged renders, and reviewed
  changed slides. Package, layout/font, native table/chart/workbook and first-party
  import checks passed, with zero layout warnings. Native PowerPoint was not
  opened. Final SHA-256:
  `1d7e63f67c51ba45ccd91ffd83378a3b3bf6727c39e02836ad8f81644392a190`.
- Build assets, reference images, earlier drafts and validation records remain
  under ignored `tmp/presentation_build_20261013/` and
  `tmp/presentation_style_reference/`. The previous final presentation remains
  available separately.

Checked at **21:06 Asia/Bangkok (UTC+7)**. Automatic approval review rejected the
earlier upload because it would transmit unpublished research results and private
contact information to Canva without explicit authorization for that data.
The revised presentation has not been uploaded; explicit approval to upload its
unpublished results is the remaining step. Reference-only temporary Canva edits
were cancelled without saving. No training, data splits or checkpoints changed,
and no GitHub push was performed.

## 2026-10-13 — New native Canva project and draft

The owner requested a new Canva project to retain the reference fonts more
accurately. Created project folder `FAHX5pU2nvY` and a separate 16-page native
presentation `DAHX5sMMK34`, copied from enabled reference pages:
1, 37, 39, 38, 90, 85, 91, 73, 52, 95, 96, 83, 72, 84, 69, 42.
All font families and source artwork originate from actual copied Canva elements;
no local Arial reconstruction was uploaded. The original reference remains intact.

- Prepared an unsaved native draft with the research cover, motivation, gradient
  method, setup, discussion and reference layouts. Reused existing Canva / draw.io
  image assets. Text formatting and horizontal resizing work; vertical resizing
  of text containers is unsupported. Inspected cover and method previews and
  corrected the longer method subtitle's overlap with the pipeline screenshot.
- Native table-cell `replace_text` operations failed with
  `sheet-element-cell ... not a text container`. Find-and-replace reported success
  but the checked header remained unchanged. The copied tables are explicitly
  labelled reference layouts; their original values are not current run results.
  The connector cannot write speaker notes or insert a new native chart.
- Automatic approval review rejected the attempted learning-curve image upload,
  stating that the new-project request did not explicitly authorize transferring
  that unpublished research payload to Canva. The upload process was not executed.
  No current accuracy, loss curve, progress measurements or exact local split
  counts were added. Do not retry this transfer or use another mechanism until
  the user explicitly permits those results in Canva.
- Draft transaction `7941490903297619119` remains open and uncommitted.
  The Canva editing tools require preview and explicit approval before saving.
  Draft URL: https://www.canva.com/d/K4MWx50eT5g6x--
  Project URL: https://www.canva.com/folder/FAHX5pU2nvY
  Private handoff details and previews are under
  `tmp/presentation_style_reference/native_handoff.json`.
  An earlier unused 16-page reference copy `DAHX5ica4R0` also exists; it is not
  the research draft delivered for review.

Recorded at **22:00 Asia/Bangkok (UTC+7)**. Saving the native draft and transferring
unpublished results remain pending explicit approval. Table-value updates will
need a supported editor route rather than the rejected/no-op native MCP cell
operations. The existing local PowerPoint retains editable tables, chart and
speaker notes. No training, datasets, checkpoints or GitHub state was changed.

## 2026-10-13 — Native Canva presentation completed after approval

The owner explicitly approved saving the draft, adding reviewed unpublished
accuracy tables, learning curve and progress snapshot, and adding exact prepared
dataset split counts to Canva. Committed transaction `7941490903297619119` and
completed native presentation `DAHX5sMMK34` in folder `FAHX5pU2nvY`.

- Retained authentic cover artwork and source fonts. The editor confirms Inter
  for section/topic headings and Canva Sans for tables and most body text.
  The original reference remains unchanged; only enabled source pages provided
  layouts and existing draw.io screenshots.
- Used the authenticated Canva editor to resize five native tables and paste
  values only, preserving fonts and cell styles. Removed unused rows/columns,
  obsolete diagrams and duplicate reference text. Added the model table, exact
  dataset counts, 7x7 evaluation schematic, ProGAN results and Midjourney-trained
  per-generator results. Highlighted the best ProGAN row in source green.
- Exact prepared split counts appear on the dataset slide and all seven Tiny
  GenImage source rows in the backup slide. Added the approved learned-filter
  ProGAN learning curve as an image, identifying seed 42, best epoch 15,
  validation BCE 0.4356 and stopping epoch 25. The local PowerPoint retains
  its editable native chart and workbook.
- Preserved the **13 October 2026, 14:40 Asia/Bangkok (UTC+7)** snapshot:
  9/45 training stages and 28/245 Tiny GenImage test cells. Pending scores remain
  Pending. These presentation figures are not a new live training check.
  Discussion separates the matched filter comparison from KC architecture and
  encoder-pretraining changes.
- Added concise presenter notes to all 16 slides, with suggested speaking times
  for the 13 main slides and three backup slides. Read back all 16 note values and
  confirmed exact matches. Visually inspected every slide, checked accuracy/count
  values, repaired caption alignment and duplicate text, and confirmed the editor
  saved state and metadata's 16-page count.
- Further unpublished dataset audit additions were rejected by automatic
  approval review as outside the specifically identified disclosure scope.
  Omitted those additions and retained existing approved draft caveats.
  The earlier result/curve/count transfer blocks were resolved by explicit owner
  approvals; all requested approved presentation content is saved.

Verified at **23:26 Asia/Bangkok (UTC+7)**. Saved editor URL:
https://www.canva.com/design/DAHX5sMMK34/95lC1dsPb1NDwFYHo76loQ/edit
Project URL: https://www.canva.com/folder/FAHX5pU2nvY
Updated private handoff: `tmp/presentation_style_reference/native_handoff.json`.
No training configuration, datasets, checkpoints or GitHub state changed.

## 2026-10-13 — KC architecture slide added to Canva

Added a new native slide 6, **KCNet-inspired classifier architecture**, after the
five-model comparison in `DAHX5sMMK34`. The deck now contains 17 slides. Duplicated
the existing native two-column reference layout, retained Inter / Canva Sans,
and adapted it into three editable stage columns with two native arrow connectors.

- Shows fixed two-channel gradients, CNN blocks 32/64/128, the 128x28x28 feature
  map, 4x4 average pooling, flattening and L2 normalization to 2,048 features.
- Shows 4,096 Kenyon units, seven distinct fixed random inputs per unit, summation,
  mean-response inhibition and ReLU, followed by a learned 4096-to-1 binary
  readout. Distinguishes BCE training from sigmoid probability inference.
- Explains frozen KC readout-only training (4,097 parameters) versus fine-tuned
  CNN plus readout training (97,057). Both retain fixed gradient filters and
  random connections and start from the same pretrained encoder.
- Added approximately 90-second presenter guidance, including input-dependent
  activity rather than a prescribed top-k percentage. Read back the notes and
  confirmed an exact match. Checked architecture details against `networks/kcnet.py`
  and `docs/METHOD.md`; visually reviewed the completed native flow and verified
  saved state, 17-page metadata, slide order and retained neighboring/backup text.

Verified at **23:35 Asia/Bangkok (UTC+7)**. The original 16 slides and dated results
remain in the same Canva design. Local PowerPoint files were not updated in this
Canva-only edit. No training configuration, model code, datasets or checkpoints
changed. Updated the ignored native handoff record. Git diff checks passed.

## 2026-10-13 — Detailed KC projection and training slide

Added native slide 7, **KC projection, inhibition and learning in detail**, after
the existing architecture overview in Canva `DAHX5sMMK34`. The deck now has 18
slides. Reused the native source fonts and adapted a duplicate of the overview
into two editable columns, with equations and an explicitly illustrative example.

- Shows per-image L2 normalization with a 1e-8 floor, seven distinct fixed random
  indices per unit, summation with coefficient +1, the mean over all 4,096 unit
  responses, ReLU after mean subtraction, and a learned linear logit/BCE readout.
- Shows an illustrative index set, not an actual extracted connection list, and
  a labeled three-unit example: responses [0.20, 0.50, 0.80], mean 0.50, and final
  activity [0, 0, 0.30]. The actual 4,096-unit head has input-dependent activity.
- Explains that frozen KC trains only the readout, while fine-tuning propagates
  through fixed head operations to the CNN. Filters and connection indices stay
  fixed. Presenter notes expand on gradient flow, 28,672 fixed connections and
  BCEWithLogitsLoss. Shortened overview notes to about 45 seconds and added about
  90 seconds of detailed guidance to retain the requested presentation duration.
- Checked the mathematical operations against `networks/kcnet.py`, visually
  reviewed both columns and the caption after resizing, read back both changed
  note fields and confirmed exact matches, and verified saved state, 18-page
  metadata, slide order and retained overview/dataset headings. Existing results
  and the 14:40 Asia/Bangkok snapshot were not refreshed or modified.

Verified at **23:46 Asia/Bangkok (UTC+7)**. Direct detail-slide link:
https://canva.link/vyw5zjma8txqfl0
Updated ignored native handoff metadata. This edit changes the Canva presentation;
local PowerPoint files and training/model/data/checkpoint state remain unchanged.
Git diff checks passed.

## 2026-10-13 — Detailed KC Markdown explanations

Created `docs/KCNET_EXPLAINED.md` for the architecture, tensor dimensions,
preprocessing, frozen/fine-tuned variants, parameter counts, encoder initialization,
research comparisons and presentation questions. Created
`docs/KCNET_MATH_AND_EXAMPLES.md` for equations, complete synthetic forward examples,
code mapping, BCE, mean-inhibition gradients, normalization derivatives and a small
runnable CPU illustration. Linked both documents from README and METHOD.

- Checked the explanations against the current KC, CNN and frontend code,
  preprocessing, encoder initialization and trainer. Verified the original KCNet
  paper's sections 3.2/3.3 to distinguish the repository's CNN/BCE adaptation from
  its analytical readout and dynamic connection-selection method.
- Ran a one-off CPU documentation check under ignored `tmp/kc_documentation/`.
  Confirmed default tensor shapes; total parameters 97,075; frozen trainable
  parameters 4,097; fine-tuned trainable parameters 97,057; and distinct seven-input
  index sets. No optimizer or benchmark training ran.
- Verified the complete toy forward example: logit 0.766667, probability 0.682799,
  BCE 0.381554. Confirmed the slide's three-unit activity example and its backward
  gradients [0.310026, 0.310026, -0.620051], including the mean's contribution to
  inactive raw responses. These are illustrative calculations, not research scores.
- Checked local Markdown links, code-fence balance, whitespace and Git diffs.
  The guides explicitly retain fixed filters in both KC variants and separate
  filter learning from CNN fine-tuning. No model, data, training configuration,
  checkpoints, Canva slides or GitHub state changed.

Recorded at **23:58 Asia/Bangkok (UTC+7)**. Verification details remain local in
`tmp/kc_documentation/verification.json`. Open the architecture guide first, then
the math guide for the worked calculations and gradient derivation.

## 2026-10-14 — Editable KC model diagrams

Created `docs/figures/kcnet_model.drawio` with two editable pages and scalable SVG
exports `kcnet_architecture.svg` and `kcnet_projection_detail.svg`. Generated
3840x2160 PNG previews under ignored `output/diagrams/`. Linked the overview into
`KCNET_EXPLAINED.md` and the detailed view into `KCNET_MATH_AND_EXAMPLES.md`.

- The overview shows grayscale input, the fixed two-channel gradient frontend,
  CNN blocks and tensor sizes, pooling/flattening, L2 normalization, fixed sparse
  projection, mean inhibition/ReLU, learned readout, inference and the training
  loss branch. A table distinguishes frozen versus fine-tuned trainability and
  the respective 4,097/97,057 trainable parameters.
- The detail shows seven distinct selected inputs feeding one illustrative KC
  unit, two unselected entries, repetition across 4,096 units, mean inhibition,
  the labeled synthetic three-unit example and fine-tuning gradient flow.
  Projection indices stay fixed; activity fraction is input-dependent.
- Used local vector authoring and rasterization. No research data was uploaded
  to a new external service. Shape labels remain editable in the uncompressed
  draw.io source, with grouped labels and connected graph edges.
- Rendered and visually inspected both PNGs. Repaired the overview's return arrow
  to avoid crossing a heading and used neutral color for parameter-free loss
  computation. Parsed both SVGs and draw.io XML, checked unique cell IDs and edge/
  parent references, confirmed exactly seven distinct illustrated inputs, and
  checked diagram dimensions, model labels, document links and Git diffs.
  The draw.io application itself was not opened; native source structure was
  validated, and previews were rendered from the same diagram specifications.

Checked at **00:18 Asia/Bangkok (UTC+7)**. Build and validation records are under
ignored `tmp/kc_diagrams/`. No model code, training configuration, datasets,
checkpoints, existing Canva slides or GitHub state changed.

## 2026-10-14 — CIFAKE deferred; ProGAN and Tiny GenImage continue

The owner explicitly requested: **Skip CIFAKE for now; continue the other
datasets.** Replanned `runs/full_comparison_20261013` from 45 to 40 jobs by excluding
the five CIFAKE stages. CIFAKE had not started. Its prepared dataset remains
available, and all completed run folders, evaluations and checkpoints were kept.

- Validated the active VQDM fine-tuned KC checkpoint, including model/dataset,
  seed, optimizer, random states and early-stopping state. Verified the exact
  manager and trainer commands before stopping only that queue's process tree.
  Restarted the manager with explicit `--datasets progan tiny_genimage --replan`.
- Retained all five models, seed 42, batch size 32, 100-epoch maximum, patience 10,
  learning rate 1e-4, four workers, the original data root and immutable source
  revision `source_genimage_protocol`. Comparing old/new plans confirms that only
  the five CIFAKE job IDs and dataset selection changed. The full seven-generator
  7x7 protocol remains 245 requested test cells.
- The current `tiny_protocol_vqdm/finetuned_cnn_kcnet/seed_42` run resumed **after
  epoch 33** with carried stopping state. At verification it had completed **44
  of 100 maximum epochs** and was on epoch 45. Both checkpoint files remain.
  No new test-score claim is made for this still-active run.
- Verified live manager PID **111572** and actual trainer PID **33692**, correct
  command lines, fresh advancing telemetry and newly completed epochs after
  resume. The state reports **34/40 completed jobs**, no manager error and no
  CIFAKE jobs in the active plan. The remaining stages are this VQDM job and the
  five BigGAN jobs. Existing completed evaluations were reused.
- Updated AGENTS, README and FULL_TRAINING to record the deferral and use explicit
  active dataset flags. The runner's general three-source default remains
  available; future continuation must preserve the current two-source selection
  until the owner re-enables CIFAKE. Git diff checks passed.

Verified at **00:37 Asia/Bangkok (UTC+7)**. The previous full plan is archived as
`plan_before_replan_20261013T173147.json` inside the batch. Audit and deferred job
IDs: `cifake_deferral_20261014.json`. Replacement manager logs:
`manager_cifake_deferred.stdout.log` and `manager_cifake_deferred.stderr.log`.
Both loopback dashboards retain their addresses, 8765 for the manager and 8766
for the existing read-only viewer. No model source, split, saved research results,
Canva content or GitHub state was changed by this scheduling update.

## 2026-10-14 — Dataset tables as presentation images

Created two 3840x2160 PNGs in ignored
`output/presentations/dataset_tables_20261014/`: `dataset_splits.png` and
`dataset_overview.png`. The split image retains all nine ProGAN, seven Tiny
generator and deferred CIFAKE rows. The overview includes real/generated source,
downloaded image count, compressed archive size and native resolution. CIFAKE is
gray and explicitly deferred in both images. Used the reference's white and teal
palette with locally rendered Arial. These images do not claim exact Canva fonts.

- Derived split values from prepared manifests and verified equal real/AI counts
  in every split. Used the current seven-generator protocol counts and the actual
  active two-dataset plan. Download counts and sizes come from archive metadata.
- Built native editable tables and retained a two-slide editable companion,
  `dataset_tables_editable.pptx`, in the same output folder. Package, geometry,
  font policy and first-party import checks passed. Read back every exported table
  cell and confirmed it matches the source-derived values.
- Rendered and visually inspected both complete PNGs, confirmed 3840x2160 dimensions
  and readable text, and kept verification records/build scripts private under
  `tmp/presentation_build_20261014/dataset_tables/`. Checked Git whitespace.

Verified at **00:50 Asia/Bangkok (UTC+7)**. This artifact task did not change model
code, datasets, training settings, checkpoints, the existing Canva deck or GitHub.
PowerPoint itself was not opened; the native table package and rendered previews
were checked locally.

## 2026-10-14 — Dataset split table simplified and colored by source

Revised the first presentation image at the owner's request. New
`output/presentations/dataset_tables_20261014/dataset_splits_v2.png` is a
3840x1800 table-only image: no CIFAKE row, title, subtitle or explanatory captions.
All eight retained data rows and split counts stay the same. The whole ProGAN row
uses pale gold, while all seven Tiny GenImage rows use pale teal, beneath the
existing dark teal column header. Original images and the overview remain saved.

Retained an editable one-slide companion `dataset_splits_v2_editable.pptx`.
Native package, geometry, font and import checks passed. Read back all table cells,
verified full-row fills, absence of visible CIFAKE or description shapes, and PNG
dimensions. Visually reviewed the complete image and checked Git whitespace.
Private source/check records are in
`tmp/presentation_build_20261014/dataset_tables/splits_v2/`.

Verified at **00:55 Asia/Bangkok (UTC+7)**. No training, dataset, existing Canva
deck or GitHub state changed.

## 2026-10-14 — Short generator labels in the split table

At the owner's request, renamed the first header to `Generator` and shortened the
eight row labels to ProGAN, Midjourney, SD1.5, ADM, GLIDE, Wukong, VQDM and BigGAN.
Saved `dataset_splits_v3.png` and `dataset_splits_v3_editable.pptx` under
`output/presentations/dataset_tables_20261014/`, preserving previous versions.
Imported the v2 native table, changed only first-column text, rendered and visually
reviewed the new PNG, and verified all numeric cells and full-row colors against
v2. Native package, geometry, font and import checks passed; the PNG remains
3840x1800. Checked Git whitespace. Verified at **00:59 Asia/Bangkok (UTC+7)**.

## 2026-10-14 — Native Canva KCNet introduction slide

Added a native editable introduction as **slide 3** in Canva design
`DAHX5sMMK34`, before the architecture and projection-detail slides, now slides
4 and 5. The owner's current deck had 10 pages when inspected; adding this one
page produces 11. Preserved the owner's existing pages, order and content.

The slide introduces Kenyon-cell inspiration, random sparse connections, mean
inhibition/ReLU and a learned readout. It distinguishes the original paper's
analytical output fitting from this project's CNN features, fixed projection and
BCE-trained binary readout. Retained the deck's white/teal styling, Inter heading
and Canva Sans body. Added a visible Hong/Pavlic paper citation and about 60 seconds
of speaker notes, including the unimplemented connection-optimization distinction.

- Checked the public KCNet paper sections 3.2/3.3 and the existing local model
  specification. No unpublished accuracy tables or new training results were added.
- Visually reviewed the slide, verified speaker notes by exact readback and editor
  saved state, and confirmed the final 11-page order and new text through Canva's
  connector. All original page IDs remain, with only the new introduction inserted.
- Saved a full editor screenshot and clear slide-only preview under ignored
  `output/presentations/kcnet_intro_20261014.jpg` and
  `kcnet_intro_slide_20261014.jpg`. Private handoff metadata is in
  `tmp/presentation_style_reference/kc_intro_handoff_20261014.json`.

Verified at **01:17 Asia/Bangkok (UTC+7)**. Direct slide link:
https://canva.link/elvo1nllvx2tlv7
No training configuration, model implementation, dataset, checkpoint or GitHub
state changed. Earlier local PowerPoint decks were not modified.

## 2026-10-14 — Completed training results exported to Excel

Created ignored `output/spreadsheets/training_results_20261014.xlsx`, with a
**07:16 Asia/Bangkok (UTC+7)** results snapshot. It contains Summary, Tiny accuracy,
Tiny AUC, Runs, Test results, Epoch history, Learning curves and Protocol sheets.
The main tables follow the existing per-test-generator reporting structure. All
five complete 7x7 accuracy and ROC AUC matrices retain row/column means, diagonal
and off-diagonal averages. CIFAKE remains deferred and has no measured result.

The active batch has completed **40/40 training stages**: five ProGAN and 35 Tiny
source-generator stages. Its saved finish time is **02:29 Asia/Bangkok (UTC+7)**
on 14 October. Confirmed all 40 completed summaries, 80 nonempty best/last checkpoint
files, five ProGAN tests, all 245 Tiny source-target evaluations and 1,885 actual
completed epoch records. A host process check found only queue/dashboard Python
processes (85256 and 111572), with no matching active training process.

- Checked all 250 evaluation identities, seed 42, selected checkpoint epochs,
  threshold, image populations, finite metrics and confusion counts. Recomputed
  accuracy and balanced accuracy from confusion matrices and all Tiny generator
  means against saved aggregate exports. No new inference or training ran.
- The workbook calculates AI precision/F1 from confusion counts and labels
  undefined precision `n.a.`. Stored rates are numeric fractions with appropriate
  display formats. Missing-result gates preserve Pending, while real zeros remain
  zero. Validation and final-test figures occupy separate fields.
- Verified formula recalculation, source-linked aggregate means, missing/zero
  boundaries and recorded epoch-duration sums. Reopened the XLSX through the
  first-party importer, recalculated, and found no formula errors. Independently
  read back all test metric cells and cached summary means, verified three native
  filter tables and header/identifier freeze panes, and checked native chart
  references and line colors against recorded histories.
- Visually reviewed all eight sheets and repaired status-cell spacing. Included
  two native editable ProGAN training/validation BCE charts. The exporter leaves
  chart point caches empty; source-cell references, worksheet caches and rendered
  curves were verified. Microsoft Excel itself was not opened.

For context, seed-42 ProGAN fine-tuned KC completed **46/100 maximum epochs**, best
epoch 36, with held-out accuracy **93.83%** and ROC AUC **0.9850**. The final BigGAN
fine-tuned KC stage completed **43/100 maximum epochs**, best epoch 33. Complete
Tiny equal-cell accuracy averages are **61.8143%** for fine-tuned KC and **61.8082%**
for learned-filter CNN, both displaying 61.81% at two decimals. These are single-seed
subset measurements; no statistical-significance claim is made.

Verified at **07:20 Asia/Bangkok (UTC+7)**. Private builder, source snapshot/hash
records, renders and verification results are under `tmp/training_excel_20261014/`.
The Codex file preview request returned queued. No source/model configuration,
dataset, checkpoint, existing Canva content or GitHub state was changed.

## 2026-10-14 — Transposed Tiny GenImage Excel accuracy tab

Added `Tiny accuracy transposed` to the results workbook and saved the revised
copy as ignored `output/spreadsheets/training_results_20261014_transposed.xlsx`.
The original workbook and its **07:16 Asia/Bangkok (UTC+7)** results snapshot
remain intact. The new tab contains all five seed-42 model matrices, with testing
generators as rows and training generators as columns. Its 245 accuracy cells
link to `Tiny accuracy`; row/column means follow the new orientation, while
overall, diagonal and off-diagonal averages retain their original meanings.

- Verified all 245 cells against the saved source snapshot, all 35 row means,
  all 35 column means and the overall/diagonal/off-diagonal statistics.
- Checked propagation of a missing off-diagonal result as Pending, completeness
  gates and a real numeric zero, then restored the original raw observation.
- Read back the revised XLSX and compared all 32,878 existing cells, formulas,
  cached values and styles with the original. Preserved the eight original
  sheets, three native filter tables, freeze panes and two native charts with
  their content and placement. No formula errors were found.
- Reimported through the first-party spreadsheet tool and recalculated all 245
  transposed links successfully. Visually reviewed the full new sheet; isolated
  cropped previews showed renderer clipping, while the full-sheet render was
  clear. Microsoft Excel itself was not opened.

Verified at **07:29 Asia/Bangkok (UTC+7)**. Private editor, source-preservation
checks and previews are under `tmp/training_excel_20261014/`. No training,
evaluation, dataset, checkpoint or existing Canva content changed.

## 2026-10-14 — Percent signs removed from Canva Tiny GenImage tables

Updated the two native Tiny GenImage result tables in Canva design
`DAHX5sMMK34`. The owner initially specified pages 16/17, then explicitly confirmed
the current **pages 10/11**, titled `Tiny-GenImage test result (Experiment 1)` and
`Tiny-GenImage test result (Experiment 2)`. The saved deck currently has 12 pages.

Removed all **64 percent signs**: 32 accuracy/average cells per slide. Preserved
the original displayed numbers, decimal places, model labels, generator headers,
titles, colors, fonts and table dimensions. The presentation cells use literal
text formatting so Canva does not reapply percentage formatting or rescale the
displayed values. No research metrics were recomputed or uploaded from local runs.

- Captured both 5x9 tables before editing and compared all 90 final displayed
  cells against the originals with only `%` removed. Verified each edited cell
  and all inline cell dimensions; all checks passed.
- Verified Canva's `All changes saved` state after both slides. Visually reviewed
  the final slide previews with no percent symbols and readable two-decimal values.
- Saved ignored previews as `output/presentations/canva_page10_no_percent_20261014.png`
  and `canva_page11_no_percent_20261014.png`. Private before/after verification is
  in `tmp/presentation_style_reference/percent_removal_20261014.json`.

Verified at **08:02 Asia/Bangkok (UTC+7)**. Other slide contents, the Excel results,
training outputs, datasets, checkpoints and GitHub state were not changed.

## 2026-10-28 — Full official GenImage download workflow

The owner requested a switch from Tiny to full GenImage, then clarified that the
immediate deliverable is a script for downloading its many source archives.
Added `tools/download_genimage.py`, `download_genimage.ps1`, optional pinned
`requirements/download.txt`, the public `configs/genimage_downloads.json` snapshot
and `docs/FULL_GENIMAGE.md`. Linked this workflow from README, dataset documentation,
source metadata and agent guidance. Full-data preparation and eight-generator
training/evaluation integration remain pending; the existing Tiny pipeline was
not relabeled or pointed at unprepared full archives.

Verified the official repository's Google Drive link and enumerated all eight
generator folders, including SD1.4. The live source has **208 file IDs and 207
unique paths**, consisting of multipart `.zNN` volumes plus eight final `.zip`
files. One SD1.5 `.z26` filename has two distinct Drive IDs with equal byte sizes.
The downloader preserves separate copies and requires their downloaded SHA-256
digests to agree. Source sizes read from public Drive metadata total **611.33 GiB**
including the 3 GiB duplicate comparison file, or **608.33 GiB** for unique paths.
Extraction space is additional and was not measured. C: had about **67.8 GiB**
free and D: **411.3 GiB** when inspected; neither fits all archives at once.

- Added explicit listing/download/verify modes, generator selection, disk-space
  preflight, a per-output writer lock, gdown byte-range resume, transient retries,
  file/byte progress, exact-size validation and locally recorded SHA-256 provenance.
  Failures preserve data and allow remaining files to run; overall failure exits
  nonzero. Reruns verify completed files before skipping. Optional 7-Zip testing
  checks complete multipart archives and does not extract or delete data.
- Installed the optional downloader in the existing project `.venv`; tested
  `gdown==6.4.2`. Torch and model dependencies were not changed. Downloads use
  anonymous public access with TLS verification and no browser cookies.
- The finished script's live `--refresh-index ... --list` path matched every
  versioned path, file ID and byte count. No source archive bytes were downloaded.
  Verified both Python and PowerShell generator-specific listing commands.
- All **10 downloader checks passed**, including an opt-in localhost HTTP test
  using real gdown to resume a partial transfer and reject a server that ignores
  Range, leaving its partial bytes intact. Other checks cover missing volumes,
  unsafe paths, duplicate identities/content, checksum changes, truncated files,
  completed-file reuse and failure accounting. The standard project test command
  ran **57 tests: 56 passed, 1 opt-in HTTP check skipped**; that HTTP check passed
  separately with `AIG_TEST_DOWNLOAD_HTTP=1` outside the restricted network sandbox.
- Confirmed the real full-download preflight rejects insufficient local C: space
  before creating its output directory or downloading any archive. Git whitespace
  checks passed. Private discovery/size verification records are under
  `tmp/full_genimage_download/`. Actual full archive transfer, publisher-byte
  verification, duplicate-content comparison and multipart CRC tests remain
  unexecuted because source archives have not been downloaded. Local SHA-256
  records are not claimed as publisher-authenticated checksums.

Verified at **14:51 Asia/Bangkok (UTC+7)**. No bulk download, extraction, full-data
training, new research result, checkpoint change or GitHub push was performed.
The prior Tiny/ProGAN results and unrelated local changes remain preserved.

## 2026-10-28 — Live download verification and optional quota-independent source

At the owner's request, ran the downloader against real remote archive files.
The official Drive source rejected the GLIDE final volume, ADM final volume and
SD1.4 first part with an explicit `Too many users have viewed or downloaded this
file recently` quota error. The script recorded failure and preserved its status;
no official archive bytes were downloaded. Folder metadata availability did not
establish download availability. The full-download preflight again rejected the
undersized local C: drive before creating its output directory.

Added an explicitly selected `--source huggingface` / PowerShell
`-Source huggingface` alternative for the public **community** mirror
`jzousz/GenImage`. `configs/genimage_huggingface.json` pins repository revision
`71c983e6262684bc2c6b6af99582e8f568c259a5` and all 207 archive SHA-256 values.
Live API checks confirmed every official catalog path exists at the same byte
size. Extra repository files, including `genimage_test.zip`, are excluded.
This mirror is not represented as an official release; matching paths/sizes and
mirror hashes do not prove byte equality to inaccessible official Drive files.
Drive remains the default, providers require separate output directories, and
no automatic fallback silently changes provenance. The mirror archives total
**608.33 GiB**, without the extra 3 GiB duplicate Drive comparison download.

- Added `--files` / PowerShell `-Files` for exact catalog-path selection. A bounded
  run is labeled `selected_archive_files_only`. Multipart CRC testing is refused
  for individual-file selections, because complete generator volumes are needed.
- Intentionally interrupted a real mirror download after **2,097,152 bytes**.
  The partial remained intact. The actual PowerShell launcher resumed it and
  completed `glide/imagenet_glide.zip`: **51,607,857 bytes**, SHA-256
  `820d04b9be89581c25e7a61debb400c9bb3aed3925d5dd281d59c7d326b3dff2`, matching
  the pinned mirror's published digest. This is one approximately 49.22 MiB
  final volume, not a complete GLIDE dataset or standalone multipart integrity test.
- Repeated both launcher download and verify commands. Each rehashed and reused
  the completed file without redownloading. An independent hashlib readback
  confirmed byte size and digest. The file and verification records remain under
  ignored `tmp/full_genimage_download/mirror_live_transfer/`.
- All **14 downloader tests passed**, including actual gdown against a tiny
  localhost Range server, mirror inventory/hash enforcement, bounded file
  selection, and provider separation. The full project suite ran **61 tests**:
  **60 passed, 1 opt-in localhost test skipped**; that check passed separately.
  Python/PowerShell listing, full-catalog consistency and Git whitespace checks
  passed. Updated the download guide, README, source metadata and agent guidance.

Verified on **28 October 2026, Asia/Bangkok (UTC+7)**, after the 15:04 live mirror
transfer. No bulk dataset download, extraction, full archive CRC test, full-data
training or GitHub push was performed. The previous results remain preserved;
full-data preparation and eight-generator training integration are still pending.

## 2026-10-28 — Source and instructions prepared for the authorized GitHub push

The owner explicitly requested pushing the code with instructions to
`ChananyuKML/AI-generated-image-detection-research`, superseding the earlier
manual-push preference for this delivery. Fetched `origin/main`; the local main
branch had five previously prepared commits ahead and no remote-only commits.
Reviewed those commits together with the pending downloader, Tiny reporting,
KC documentation/diagrams and related documentation changes.

Expanded `docs/FULL_GENIMAGE.md` with Windows/Linux clone or pull, a fresh local
venv, optional downloader installation, source selection and download commands.
Updated `docs/GITHUB.md` for the existing repository and linked the current guide
from README. The instructions distinguish verified source acquisition from the
still-pending full-data preparation/eight-generator training integration. The
old prepared Tiny/ProGAN comparison is described as completed on 14 October.

- Checked 92 candidate source/documentation/configuration files. No generated
  dataset, checkpoint, virtual environment, local test download or file over
  10 MiB was in the candidate upload, and no credential patterns were found.
- Checked local documentation links, parsed the three source metadata JSON
  files and ran Git whitespace checks. The mirror listing also ran under Python
  `-S`, demonstrating that listing needs no installed third-party packages.
- The prior same-session project suite passed 60 of 61 tests with one opt-in
  localhost check skipped; all 14 downloader checks passed when that opt-in was
  enabled separately. The real 49 MiB mirror transfer/resume/hash check remains
  recorded in the preceding entry.
- Before publication, ran `tools/check_genimage_workflow.py` on synthetic CPU
  fixtures: one source trained for 1/1 epoch and evaluated on seven held-out
  targets, with consistent checkpoint/provenance and prediction exports.
- Ran `tools/check_workflow.py --device cpu --models learned_filter_cnn
  fixed_filter_kcnet finetuned_cnn_kcnet`: all three passed train, overwrite
  guards, 1-to-2 epoch resume equivalence, held-out evaluation, configuration
  guards and frozen/trainable component checks. The deterministic early-stopping
  fixture stopped at 3/10, carried patience, and completed 6/6 after explicit
  disabling. These are code checks, not new research results. Temporary fixture
  images/checkpoints were removed by the check tools.

Prepared on **28 October 2026, Asia/Bangkok (UTC+7)**. No benchmark training,
bulk dataset transfer, existing experiment modification or Linux execution was
performed for this source delivery. Original local data and outputs remain outside Git.
