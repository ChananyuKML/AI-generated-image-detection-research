# GenImage-style testing on the seven-generator Tiny subset

The approved experiment trains a separate detector on each available generator
and evaluates each detector on every available generator. Each of the five model
types produces a **7×7 matrix**: 35 Tiny training runs and 245 held-out test cells.
Together with ProGAN and CIFAKE, the full queue contains 45 training stages.

This follows the cross-generator structure in the official
[GenImage instructions](https://github.com/GenImage-Dataset/GenImage#benchmark) and
[paper section 3.2](https://arxiv.org/html/2306.08571v2#S3.SS2). The original benchmark
uses eight generators; this community Tiny release lacks SD1.4. Results must be
reported as a **seven-generator Tiny subset adaptation**, not a reproduction of
the complete official benchmark. Model architecture and training settings remain
those of this research project.

## Prepare independent selection and testing splits

Use the already verified source archive/index; no new download is required:

```powershell
.\.venv\Scripts\python.exe tools/prepare_genimage_protocol.py
```

On Linux use `.venv/bin/python` instead. Preparation creates these source datasets
under `dataset/prepared/`, using split seed 42:

| Folder | Train | Selection validation | Final test |
|---|---:|---:|---:|
| `tiny_protocol_midjourney` | 3,556 | 400 | 1,000 |
| `tiny_protocol_sdv5` | 3,600 | 394 | 1,000 |
| `tiny_protocol_adm` | 3,588 | 400 | 1,000 |
| `tiny_protocol_glide` | 3,588 | 400 | 1,000 |
| `tiny_protocol_wukong` | 3,600 | 382 | 1,000 |
| `tiny_protocol_vqdm` | 3,600 | 362 | 1,000 |
| `tiny_protocol_biggan` | 3,600 | 392 | 1,000 |

All supplied `val` images remain in **final test**: 500 real and 500 AI images per
generator. Checkpoint-selection validation is reserved from original training
groups, using a 10% deterministic allocation followed by class balancing. This
allocation is our explicit subset adaptation. There are no POC sampling caps.

Original encoded image bytes are preserved and reused through hard links where
possible. Pixel duplicates are grouped globally before role assignment. The one
shared real training image between ADM and Midjourney keeps the same role. Audits
reject overlap between any detector's train/selection-val images and any target's
test images, and reject train/selection-val overlap across generator exports.
Near-duplicate similarity has not been audited. Existing older prepared datasets
and run outputs are preserved separately.

## Train and evaluate

For the complete matrix, start the full queue after preparation:

```powershell
.\.venv\Scripts\python.exe tools/run_full_training.py --genimage-protocol cross-generator --batch-dir runs/full_genimage_comparison
```

For one source/model, train on the source's `tiny_protocol_` folder using the
standard trainer. Both KC variants require a validation-selected fixed-filter
encoder from that same source folder/signature. Then evaluate its best checkpoint:

```powershell
.\.venv\Scripts\python.exe evaluate_genimage.py --checkpoint runs/full_genimage_comparison/experiments/fixed_filter_cnn/tiny_protocol_midjourney/seed_42/best.pt --device cuda --workers 4
```

The evaluator loads one checkpoint and tests all seven held-out generator sets.
It requires the source protocol signature, binary label convention, and
validation-best checkpoint. Test labels never select checkpoints or thresholds.
The threshold stays at AI probability strictly greater than 0.5.

## Results and averages

Each source run writes `genimage_evaluation/<target>/metrics.json` and
`predictions.csv`, plus `matrix_row.json` with seven target results and row means.
The batch exports:

- `genimage_matrices.json`: accuracy/AUC matrices by model/seed, row/column means,
  diagonal accuracy, off-diagonal accuracy, and complete/incomplete flags.
- `genimage_matrix_cells.csv`: all individual source-target measurements.
- `genimage_table4.csv`: average over training sources for each test-generator
  column, and the overall equal-cell accuracy average.
- `results.csv`: each source run's equal-weight mean across its seven targets.

The overall accuracy averages all 49 cells equally, including the diagonal, as in
the full cross-generator table structure. The off-diagonal average separately
summarizes the 42 unseen-generator cells. Final averages remain blank until their
required cells exist; partial observed means are explicitly separate.

The live dashboard shows the generator-specific queue and selectable 7×7 matrices.
During testing it shows which target is being evaluated and its batch progress.

### Presentation and paper tables

The queue also writes `presentation/genimage_accuracy_tables.md` and `.tex` after
each completed source evaluation. The summary follows **Table III in the owner's
research paper**: model rows, test-generator columns, and the equal-cell overall
average. Each test-generator column averages its seven training sources. Accuracy
is displayed as percentages with two decimals. The existing CSV exports retain
fractions in [0, 1]. These files also include all five individual 7×7 matrices,
source-row means, seven-cell diagonal and 42-cell off-diagonal means. A mean is
Pending until all its required cells are measured; missing scores are never zero.

Regenerate the tables from recorded results, without running inference:

```powershell
.\.venv\Scripts\python.exe tools/export_genimage_tables.py --batch-dir runs/full_comparison_20261013
```

Add `--watch` to keep updating an already-running batch whose manager predates
this exporter. It stops after all planned Tiny matrices are complete. This reads
saved results and uses no GPU. On Linux use `.venv/bin/python`. The `.tex` output
contains table fragments to insert into an existing paper, rather than a standalone
document. The paper's Table II used an SD1.4-trained detector; this subset has no
SD1.4, so its individual source rows are labelled by their actual training source.
Published full-dataset baseline numbers are not inserted into this subset table.

## Updating an already-running older queue

`tools/switch_genimage_protocol.py` is a Windows coordinator for the existing
pooled batch. It audits the new data and waits until the current ProGAN stages
finish before replacing the upcoming Tiny stage. It preserves completed runs,
records the old plan/status, verifies the manager identity, and starts the new
queue with `--replan --source-revision source_genimage_protocol`. ProGAN and CIFAKE
retain their existing settings. Training source snapshots remain versioned.

This coordinator must run outside the Windows sandbox to manage the already
authorized local training processes. Its `genimage_protocol_switch.json` records
whether the transition is waiting, active, or failed; verify that record and
actual manager/child liveness before claiming a transition is complete.
