# Organization and compatibility

The root now exposes the supported binary trainer (`train.py`) and evaluator
(`evaluate.py`). Model code and transforms remain in their original `networks/`
and `utils/` locations to preserve imports and saved parameter names.

Older root Python entry points were preserved in `legacy/`:

- `legacy/train.py`: previous multi-model trainer and raw `ckpt_<epoch>.pth` format.
- `legacy/test_binary.py`: previous eight-generator GenImage evaluation matrix.
- `legacy/train_binary.py`: separate older binary trainer.
- `legacy/train_proto.py`, `legacy/test_proto.py`: deferred prototype experiments.
- `legacy/test.py`, `legacy/test_anomally.py`: older evaluation experiments.

Run them as modules **from the project root**, for example:

```sh
python -m legacy.train --network dual_cnn_128 --transform raw_grad_224 --loader gen_image --dataset sdv4 --dataroot dataset/genimage --epochs 100 --lr 0.0001 --num_workers 0 --seed 42
```

Do not assume the new `best.pt`/`last.pt` checkpoint wrapper is compatible with
legacy evaluation. Use the current evaluator with current checkpoints. Old raw
state dictionaries use the legacy evaluator and its original folder conventions.

The previous PowerShell launchers now wrap the supported trainer. Existing
`-Network`, `-Dataset`, `-Epochs`, `-BatchSize`, `-LR`, `-NumWorkers`, `-Seed`, and
`-CheckOnly` options remain available. `-Device`, `-RunDir`, and `-Resume` were added.
`train_local.ps1` targets `dataset/genimage` by default; `train_benchmark.ps1`
targets `dataset/prepared`. The latter also accepts future dataset folder names.

Research PDFs, old top-level figures, and machine-specific cluster scripts were
preserved locally under `local_archive/`. Older `output/` artifacts remain local.
They are excluded from the proposed GitHub upload.

The prototype, anomaly, transformer, and old loader paths remain experimental.
Known unresolved issues include prototype label/threshold handling, incomplete
model registries, and older checkpoint/loader assumptions. The supported workflow
does not route through those entry points. Their presence is for research history,
not a claim that every archived experiment is ready to train unchanged.
