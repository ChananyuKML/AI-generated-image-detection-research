# Current two-channel method

The implemented pipeline is:

```text
RGB image -> grayscale -> aspect-preserving resize -> 224x224 center crop
  -> Gx/Gy 3x3 convolutions -> magnitude and atan2 angle -> per-channel min/max
  -> CNN blocks 32 / 64 / 128 -> flatten -> MLP 128 / 64 / 1 -> binary logit
```

`networks/learned_filters.py` contains `LearnedGradientFrontend` and
`LearnableFilterCNN128`. Both filters start from the existing code's FFD kernels:

```text
Gx = [[0,  0, 0],    Gy = [[0, -1, 0],
      [0, -1, 1],          [0,  1, 0],
      [0,  0, 0]]          [0,  0, 0]]
```

The new model learns all 18 filter coefficients through the binary loss. There is
no zero-sum constraint, kernel norm constraint, or extra GroupNorm. This preserves
the existing classifier and keeps the experiment focused on learning the filters.
The coefficients are shared across images, not generated separately per image.

The frontend produces magnitude and a single angle channel. It uses **atan2(Gy,Gx)**
as the existing code does. The paper's written arctan expression and vertical FFD
convention differ from these implementation details; describe the implementation
explicitly when documenting the new experiment.

The magnitude has an epsilon-stabilized square root. Angle inputs near zero are
replaced before atan2 so the backward pass stays finite. Constant feature maps
become zero after normalization. Low-precision frontend calculations use float32.
The implementation and numerical thresholds are saved in the model state.

| Model | Total parameters at 224x224 | Trainable filter coefficients |
|---|---:|---:|
| `learned_filter_cnn` | 12,946,483 | 18 |
| `fixed_filter_cnn` | 12,946,483 | 0 |
| `dual_cnn_128` | 12,946,465 | 0 |

`fixed_filter_cnn` freezes only the frontend; its CNN and classifier still train.
It uses the same safeguards as the new method, so the learned/frozen pair isolates
filter trainability more closely than comparison with the original fixed transform.

Training uses BCEWithLogitsLoss, AdamW (default weight decay 0.01), gradient clipping
at norm 1.0, and checkpoint selection by validation BCE. Classification uses an AI
probability strictly greater than 0.5. The test split does not select checkpoints
or tune thresholds. The supported runner currently uses full precision and one
device; it does not start distributed training or automatically alter the architecture.
