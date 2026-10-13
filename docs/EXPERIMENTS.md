# Train, compare, resume, and evaluate

## One pilot experiment

```sh
python scripts/run.py train --dataset progan_poc --model learned_filter_cnn --device cuda --epochs 20 --batch-size 32 --lr 0.0001 --seed 42 --workers 0
python scripts/run.py evaluate --checkpoint runs/learned_filter_cnn/progan_poc/seed_42/best.pt --device cuda
```

Use the same options with `--model fixed_filter_cnn` for the controlled frozen-filter
comparison and `--model dual_cnn_128` for the original implementation baseline.
On a CPU-only PC, select `--device cpu` after CPU setup.

## Train the frozen encoder's KC head

First obtain a validation-selected `fixed_filter_cnn` checkpoint on the same
dataset. An existing completed run can be reused. Then train the new head:

```sh
python scripts/run.py train --model fixed_filter_kcnet --dataset progan_poc --encoder-checkpoint runs/fixed_filter_cnn/progan_poc/seed_42/best.pt --device cuda --check-only
python scripts/run.py train --model fixed_filter_kcnet --dataset progan_poc --encoder-checkpoint runs/fixed_filter_cnn/progan_poc/seed_42/best.pt --device cuda --epochs 20 --batch-size 32 --lr 0.0001 --seed 42 --workers 0
python scripts/run.py evaluate --checkpoint runs/fixed_filter_kcnet/progan_poc/seed_42/best.pt --device cuda
```

The source must be selected by validation BCE and match the current dataset manifest
signature. It supplies frozen frontend/CNN weights; its old classifier is discarded.
Only 4,097 parameters train at the default head size. This is a pretrained encoder
head experiment with additional pooling/normalization, not the original filter
trainability ablation. See [METHOD](METHOD.md) for the precise architecture.

Head options are `--kc-units` (4096), `--kc-fan-in` (7), `--projection-seed` (42),
`--kc-pool-size` (4), and `--kc-feature-eps` (1e-8). Choose settings using validation,
then evaluate the selected configuration on test. Training and projection seeds are
recorded separately. Changing a training seed does not implicitly change the projection.

Resume using the complete KC run folder, omitting the source encoder argument:

```sh
python scripts/run.py train --model fixed_filter_kcnet --dataset progan_poc --device cuda --epochs 40 --resume runs/fixed_filter_kcnet/progan_poc/seed_42/last.pt
```

Architecture options are restored from the KC checkpoint. Explicit conflicting
options are rejected. Copied encoder weights, projection indices and numerical
settings are embedded, so the source encoder checkpoint is not needed on another PC.
The `pipeline` launcher also supports this model with `--encoder-checkpoint`; its
source encoder must already have been trained. Use individual commands to adjust
the head architecture options.

The Windows benchmark wrapper accepts the same source via
`-Network fixed_filter_kcnet -EncoderCheckpoint <path>`.

## Fine-tune the CNN with the KC head

For a matched comparison against the frozen encoder, start a separate run from
the same original fixed-filter encoder checkpoint:

```sh
python scripts/run.py train --model finetuned_cnn_kcnet --dataset progan_poc --encoder-checkpoint runs/fixed_filter_cnn/progan_poc/seed_42/best.pt --device cuda --epochs 20 --batch-size 32 --lr 0.0001 --seed 42 --projection-seed 42 --workers 0
python scripts/run.py evaluate --checkpoint runs/finetuned_cnn_kcnet/progan_poc/seed_42/best.pt --device cuda
```

The CNN and readout train at the same learning rate. The FFD frontend and random
connections remain fixed. Pooling, normalization, initialization and training
budget match the frozen KC run. Select each checkpoint using validation BCE.
Resume with `--model finetuned_cnn_kcnet --resume <its-last.pt>`, omitting the source
encoder argument. The existing KC head options, pipeline and Windows wrappers
also support this model id.

## All three sources and three seeds

Prepare the three sources once, then use this PowerShell loop:

```powershell
python scripts/run.py data --datasets cifake progan tiny_genimage
foreach ($datasetName in @('progan_poc', 'cifake_poc', 'tiny_genimage_poc')) {
    foreach ($modelName in @('learned_filter_cnn', 'fixed_filter_cnn', 'dual_cnn_128')) {
        foreach ($experimentSeed in @(42, 43, 44)) {
            python scripts/run.py train --dataset $datasetName --model $modelName --seed $experimentSeed --epochs 20 --device cuda
            if ($LASTEXITCODE -ne 0) { throw 'Training failed.' }
            python scripts/run.py evaluate --checkpoint "runs/$modelName/$datasetName/seed_$experimentSeed/best.pt" --device cuda
            if ($LASTEXITCODE -ne 0) { throw 'Evaluation failed.' }
        }
    }
}
```

Equivalent Bash loop:

```sh
set -e
for dataset in progan_poc cifake_poc tiny_genimage_poc; do
  for model in learned_filter_cnn fixed_filter_cnn dual_cnn_128; do
    for seed in 42 43 44; do
      python3 scripts/run.py train --dataset "$dataset" --model "$model" --seed "$seed" --epochs 20 --device cuda
      python3 scripts/run.py evaluate --checkpoint "runs/$model/$dataset/seed_$seed/best.pt" --device cuda
    done
  done
done
```

This launches 27 runs and may take substantial time. Start with one ProGAN POC run.
Compare mean and standard deviation across training seeds using the same splits,
epoch budget, optimizer settings, and hardware where possible.

## Resume after a completed epoch

```sh
python scripts/run.py train --dataset progan_poc --model learned_filter_cnn --seed 42 --epochs 40 --device cuda --resume runs/learned_filter_cnn/progan_poc/seed_42/last.pt
```

The trainer restores model/optimizer states and Python, NumPy, PyTorch, and training
loader random states. Model, dataset, seed, batch size, learning rate, and dataset
signature must match. Epoch count is the total target. An interrupted partial epoch
restarts from the last completed epoch. Copy `best.pt` together with `last.pt`.

Prepared datasets are identified by their manifest CSV hash. For custom folders
without a manifest, the fallback signature uses file names, labels, and byte sizes;
it is not a full content-integrity check. Preserve source images when resuming.
Seeds do not guarantee bitwise equivalence across GPUs, operating systems, PyTorch
versions, or different worker settings.

## Extend a run with early stopping

`--epochs` remains the total target, including previously completed epochs. Enable
validation-BCE patience when extending a completed run:

```sh
python scripts/run.py train --model finetuned_cnn_kcnet --dataset progan_poc --device cuda --epochs 100 --patience 10 --min-delta 0 --resume runs/finetuned_cnn_kcnet/progan_poc/seed_42/last.pt
```

`--patience 10` stops after ten consecutive epochs without a significant validation
BCE decrease. Ties count as non-improvements. `--min-delta` defaults to zero;
positive values require a decrease greater than that value to reset patience.
Small decreases accumulate relative to the last significant improvement. Every
strictly lower validation BCE still updates `best.pt`, even when the decrease is
too small to reset patience. Test labels never affect stopping or checkpoint selection.

Fresh runs disable early stopping by default (`--patience 0`). On resume, omitted
patience/min-delta options restore the saved policy. Counters are checkpointed and
validated against the full validation history. Older checkpoints without these
fields reconstruct counters from history; enabling patience does not reset the
streak at the resume boundary. Explicit changes re-evaluate history using the new
policy. A checkpoint that has already exhausted patience refuses more training
unless the user increases patience or explicitly supplies `--patience 0`.

`summary.json` records actual `completed_epochs`, the maximum `requested_epochs`,
`stopped_early`, the stopping reason and monitor state. `last.pt` saves the stopping
epoch; `best.pt` retains the absolute-best validation checkpoint, which may predate
the extension. Copy the complete run folder before extension when preserving an
earlier experiment's metrics and checkpoint hashes. Evaluate only after the run
finishes or stops.

The pipeline accepts `--patience`/`--min-delta`, and both Windows wrappers accept
`-Patience`/`-MinDelta`. Unspecified wrapper options do not override a saved policy.

## Evaluate another dataset

```sh
python scripts/run.py evaluate --checkpoint runs/learned_filter_cnn/progan_poc/seed_42/best.pt --dataset tiny_genimage_poc --device cuda
```

Evaluation defaults to `test`, reports accuracy, balanced accuracy, ROC-AUC,
average precision, class recalls, confusion counts, and BCE, and writes per-image
AI probabilities. `--split val` is available for validation diagnostics. The fixed
threshold remains 0.5; evaluation does not optimize it on the held-out data.

Do not describe a comparison across related prepared variants as independent
cross-dataset generalization. Consult [DATASETS.md](DATASETS.md) for overlap/protocols.

## Keep experiment records

Store the complete run folder, dataset manifests, exact installed requirements,
Git commit, and any source-release notes with the reported results. Generated
outputs are ignored by Git; transfer or back them up separately.
