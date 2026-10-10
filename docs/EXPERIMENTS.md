# Train, compare, resume, and evaluate

## One pilot experiment

```sh
python scripts/run.py train --dataset progan_poc --model learned_filter_cnn --device cuda --epochs 20 --batch-size 32 --lr 0.0001 --seed 42 --workers 0
python scripts/run.py evaluate --checkpoint runs/learned_filter_cnn/progan_poc/seed_42/best.pt --device cuda
```

Use the same options with `--model fixed_filter_cnn` for the controlled frozen-filter
comparison and `--model dual_cnn_128` for the original implementation baseline.
On a CPU-only PC, select `--device cpu` after CPU setup.

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
