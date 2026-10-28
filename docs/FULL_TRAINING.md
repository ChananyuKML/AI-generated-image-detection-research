# Full comparison and live progress

As of **14 October 2026**, the owner deferred CIFAKE because of training duration.
The active comparison uses **only full prepared sources**: `progan` and seven
`tiny_protocol_<generator>` subsets. All five models train on each source, giving
40 runs for seed 42 and 245 Tiny cross-generator test cells. Each KC head
initializes from the same source dataset/seed's validation-selected fixed-filter
CNN encoder. The active order runs ProGAN first, then Tiny GenImage.

The runner's general default still includes CIFAKE and produces 45 runs. Use
`--datasets progan tiny_genimage` for the active scope. CIFAKE's prepared dataset
and deferred job records remain available; re-enable it when requested.

## Setup and start

Prepare the environment and sources if they are not already available:

```sh
python scripts/run.py setup --compute auto
python scripts/run.py data --datasets progan tiny_genimage
```

Then run project Python on `tools/prepare_genimage_protocol.py` (see
[GENIMAGE_TESTING.md](GENIMAGE_TESTING.md)). The default queue uses
`--genimage-protocol cross-generator`. `--genimage-protocol pooled` selects the
historical mixed-generator experiment, which has 15 runs when all three sources
are selected.

Start from the repository root on Windows:

```powershell
.\.venv\Scripts\python.exe tools/run_full_training.py --datasets progan tiny_genimage --batch-dir runs/full_comparison --epochs 100 --patience 10 --batch-size 32 --workers 4
```

On Linux:

```sh
.venv/bin/python tools/run_full_training.py --datasets progan tiny_genimage --batch-dir runs/full_comparison --epochs 100 --patience 10 --batch-size 32 --workers 4
```

The queue requires a working NVIDIA CUDA device. Training uses full precision,
224px input, AdamW at 1e-4, and seed 42. Model settings and labels are unchanged.
The selected full image inventories must match their preparation manifests. No POC
variant is accepted by the queue.

For a Windows background process that continues after the launcher exits:

```powershell
$projectRoot = (Get-Location).Path
$projectPython = Join-Path $projectRoot '.venv\Scripts\python.exe'
$queueScript = Join-Path $projectRoot 'tools\run_full_training.py'
$batchRoot = Join-Path $projectRoot 'runs\full_comparison'
New-Item -ItemType Directory -Path $batchRoot -Force | Out-Null
$queueArguments = @('-u', ('"' + $queueScript + '"'), '--datasets', 'progan', 'tiny_genimage', '--batch-dir', ('"' + $batchRoot + '"'))
Start-Process -FilePath $projectPython -ArgumentList $queueArguments -WorkingDirectory $projectRoot -WindowStyle Hidden -RedirectStandardOutput (Join-Path $batchRoot 'manager.stdout.log') -RedirectStandardError (Join-Path $batchRoot 'manager.stderr.log')
```

The manager requests Windows sleep inhibition while the training queue is active,
and releases it when the queue completes or fails. Manual shutdown/hibernation can
still interrupt training; completed epoch checkpoints support recovery.

## Live view

Open **http://127.0.0.1:8765/** on the training PC. It refreshes every two seconds:

- Current dataset/model, training child PID and whether it is still alive.
- Epoch and current training/validation batch, running BCE/accuracy, throughput,
  and approximate time remaining for that phase.
- Train/validation curves from completed epochs, validation-best epoch, patience.
- GPU utilization/VRAM, full queue, process log, and completed held-out test metrics.

The phase ETA does not predict the total queue time or early-stopping epoch. The
100-epoch setting is a ceiling; validation patience may stop a run sooner.
If the server disconnects, the page says so rather than showing a saved status as
proof of live training. The service binds only to the local PC's loopback address;
it is not a public website or a remote-machine monitor.

A read-only viewer can also observe an already-running batch without restarting
the trainer. On Windows, for example:

```powershell
.\.venv\Scripts\python.exe tools/watch_training.py --batch-dir runs/full_comparison_20261013 --port 8766
```

Open http://127.0.0.1:8766/ for that viewer. It resolves the actual trainer PID from
fresh telemetry because Windows virtual-environment launcher PIDs may differ.
Both views reject stale progress from a previous process. Short JSON reader-lock
conflicts are retried; a failed optional progress update does not abort training.

## Saved outputs and recovery

### Current batch with CIFAKE deferred

The existing batch is `runs/full_comparison_20261013`. It retains source revision
`source_genimage_protocol`, all five models, seed 42, batch 32, 100-epoch maximum,
patience 10 and four workers. Its 14 October replan excludes the five CIFAKE jobs
and preserves the complete Tiny 7x7 evaluation schedule. The previous plan is
archived in `plan_before_replan_20261013T173147.json`; the deferral audit is
`cifake_deferral_20261014.json` in the batch folder.

After the current manager exits, the matching Windows recovery command is:

```powershell
.\.venv\Scripts\python.exe tools/run_full_training.py --batch-dir runs/full_comparison_20261013 --datasets progan tiny_genimage --source-revision source_genimage_protocol --genimage-protocol cross-generator --seeds 42 --epochs 100 --patience 10 --batch-size 32 --workers 4
```

On Linux, use `.venv/bin/python` with the same arguments. The source revision and
settings must match the copied batch. Completed runs are reused and partial runs
resume from their last completed epoch. Keep the deferral dataset list on restart.

To change that dataset list later, stop the existing manager first and use an
explicit `--replan` with unchanged training settings. Re-enabling CIFAKE requires
the owner's request; its data does not need to be downloaded again.

Under the selected batch folder:

```text
plan.json, status.json              settings and current queue state
source/, source_files.json          immutable source copy and SHA-256 records
source_changes.patch               local diff at snapshot time
installed_requirements.txt         exact installed package versions
logs/                              one combined training/evaluation log per run
experiments/<model>/<dataset>/seed_42/
  best.pt, last.pt                  checkpoints
  progress.json                    live batch telemetry
  config.json, environment.json     run provenance
  history.csv, summary.json         completed epoch history and stopping outcome
  evaluation/<dataset>/test/        metrics and per-image predictions
results.csv                        completed model/dataset result table
```

Repeat the same queue command and batch folder after an interruption. Completed
runs are reused; partial runs resume from their own last completed epoch. The
manager recovers an interrupted stopping/summary boundary, carries patience,
verifies settings/signatures, and refuses to overwrite incompatible runs. A batch
lock and bound port prevent duplicate managers. Confirm the previous manager has
exited before restarting; a failed batch retains logs and checkpoints.

If starting a different experiment, use a different batch folder. For additional
training seeds, use `--seeds 42 43 44` with a new folder. Projection seed remains
42 for both KC variants. Transfer the full batch folder to preserve source records
and resumable checkpoints.

## Interpretation

The full prepared sources retain the custom protocols in [DATASETS.md](DATASETS.md).
KC stages add training after encoder pretraining and change pooling/classifier;
their comparison with an original CNN does not isolate only the head. Test labels
never select checkpoints, stop training, or choose a threshold. The automatic
evaluation uses the minimum-validation-BCE checkpoint and a fixed 0.5 threshold.
