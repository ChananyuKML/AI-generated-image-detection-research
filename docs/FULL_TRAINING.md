# Full comparison and live progress

The active comparison uses **only the full prepared datasets**: `progan`,
`tiny_genimage`, and `cifake`. All five supported models train on each dataset,
giving 15 runs for seed 42. Each KC head initializes from the same dataset/seed's
validation-selected fixed-filter CNN encoder. The default order runs ProGAN first,
then Tiny GenImage, then CIFAKE.

## Setup and start

Prepare the environment and sources if they are not already available:

```sh
python scripts/run.py setup --compute auto
python scripts/run.py data --datasets progan tiny_genimage cifake
```

Start from the repository root on Windows:

```powershell
.\.venv\Scripts\python.exe tools/run_full_training.py --batch-dir runs/full_comparison --epochs 100 --patience 10 --batch-size 32 --workers 4
```

On Linux:

```sh
.venv/bin/python tools/run_full_training.py --batch-dir runs/full_comparison --epochs 100 --patience 10 --batch-size 32 --workers 4
```

The queue requires a working NVIDIA CUDA device. Training uses full precision,
224px input, AdamW at 1e-4, and seed 42. Model settings and labels are unchanged.
The three full image inventories must match their preparation manifests. No POC
variant is accepted by the queue.

For a Windows background process that continues after the launcher exits:

```powershell
$projectRoot = (Get-Location).Path
$projectPython = Join-Path $projectRoot '.venv\Scripts\python.exe'
$queueScript = Join-Path $projectRoot 'tools\run_full_training.py'
$batchRoot = Join-Path $projectRoot 'runs\full_comparison'
New-Item -ItemType Directory -Path $batchRoot -Force | Out-Null
$queueArguments = @('-u', ('"' + $queueScript + '"'), '--batch-dir', ('"' + $batchRoot + '"'))
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

## Saved outputs and recovery

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
