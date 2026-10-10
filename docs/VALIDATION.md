# Local validation record

Validated on 2026-10-10 using Windows, Python 3.13.5, PyTorch 2.9.1+cu128,
torchvision 0.24.1+cu128, and an NVIDIA GeForce RTX 5060 Ti.

## Completed checks

- All **17 unit tests** passed, covering frontend channels, normalization,
  numerical stability, finite gradients, learned/frozen filter behavior,
  checkpoint restoration, duplicate grouping, split allocation, binary metrics,
  strict image loading, and portable paths.
- `pip check` reported no broken dependencies in the original environment.
- CUDA synthetic workflow checks passed for `learned_filter_cnn`,
  `fixed_filter_cnn`, and `dual_cnn_128`: train, save, resume, held-out evaluation,
  overwrite rejection, and resume configuration mismatch rejection.
- The learned-filter synthetic workflow passed with CPU execution as well.
- For each checked workflow, resuming at an epoch boundary produced exactly the
  same model weights as uninterrupted two-epoch training on the same PC.
- Trainable frontend coefficients changed; frozen frontend coefficients stayed
  at their FFD initialization.
- ProGAN POC and CIFAKE POC passed CUDA dataset/model preflight checks. Tiny
  GenImage POC passed the CPU preflight check.
- The cached ProGAN download/preparation command verified its snapshot checksum
  and regenerated its balanced full and POC manifests successfully.
- The PowerShell benchmark wrapper worked when invoked outside the repository.
- CPU and cu128 setup commands were inspected with `--dry-run`.
- A fresh local clone of implementation commit `5aa23ed` contained the complete
  source and loaded the trainer; its setup commands targeted its own `.venv`.
- Tracked Python files passed AST parsing; source JSON and workflow YAML parsed;
  the new workflow/docs passed Git whitespace checks.
- The upload helper previewed `ChananyuKML/aig-dec` without changing a remote
  or pushing. Git tracking and the source ZIP excluded datasets, environments,
  weights, results, temporary files, and local research archives.

## Practical limits

Synthetic checks exercise the real 224x224 models on tiny random-image splits;
they verify the workflow and do not measure benchmark detection performance.
No full benchmark training run was started during repository preparation.

CPU execution used the existing CUDA-enabled PyTorch wheel on Windows. CPU-only
installation, Linux execution, and the GitHub Actions job have not yet been run
on a destination machine. The provided CPU CI job will install its own CPU wheels
and run unit tests plus the learned-filter train/resume/evaluate smoke check.

The original prepared datasets were validated during download/preparation.
Near-duplicate image similarity was not audited. The dataset split protocols and
their research limitations are documented in [DATASETS.md](DATASETS.md).
