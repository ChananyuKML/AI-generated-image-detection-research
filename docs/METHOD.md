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

Optional early stopping monitors validation BCE. Patience and a minimum decrease
control when training ends; the checkpoint remains selected by the absolute minimum
validation BCE. Stopping state and policy survive resume. See
[EXPERIMENTS](EXPERIMENTS.md#extend-a-run-with-early-stopping).

## KCNet-inspired head experiment

Detailed companion documents: [architecture and presentation questions](KCNET_EXPLAINED.md)
and [equations, worked examples and backpropagation](KCNET_MATH_AND_EXAMPLES.md).

`fixed_filter_kcnet` is a separate head experiment. A fresh run copies only the
frontend and three CNN blocks from validation-selected `fixed_filter_cnn/best.pt`.
The original dense classifier is discarded. Both the copied encoder and its FFD
kernels remain frozen; only the new readout trains.

```text
224px raw grayscale
  -> unchanged frozen magnitude/angle frontend
  -> frozen CNN blocks 32 / 64 / 128 (128 x 28 x 28)
  -> adaptive average pooling to 4 x 4 -> flatten (2048)
  -> per-image L2 normalization, norm floor 1e-8
  -> fixed sparse random projection (4096 units, 7 distinct inputs per unit)
  -> subtract each image's mean unit response -> ReLU
  -> learned Linear(4096, 1) -> binary logit
```

The projection has 28,672 fixed nonzero connections, stored as integer indices
rather than a dense 4096x2048 matrix. Projection seed defaults to 42 and is separate
from the training seed. The active fraction after inhibition/ReLU is input-dependent;
there is no prescribed top-k or 5% activity. The model has 97,075 neural parameters,
of which 4,097 readout parameters train. It additionally stores the connection indices
(224 KiB at the defaults) and numerical buffers.

This adapts [KCNet](https://arxiv.org/abs/2108.07554)'s random projection and inhibition
to CNN image features. Spatial pooling, feature L2 normalization and a binary
BCE-trained readout are explicit adaptations; the paper uses a ridge-regression
readout. It is not a reproduction of published KCNet benchmark protocols.

For a controlled comparison of heads, use the same frozen encoder, pooling and
normalization with linear or MLP readouts. Comparing only against the original full
CNN changes pooling and the classifier as well as trainability. The KC experiment
also has an encoder pretraining stage, so its 20 head epochs are additional training
and do not represent the same total training budget as a fresh baseline run.

Encoder initialization requires a matching dataset name/signature and the same
input/label conventions. Source checkpoint SHA-256, selection and epoch are recorded.
Every KC checkpoint contains its copied frontend/CNN, projection indices, safe
feature-normalization floor, readout and architecture config; resume/evaluation do
not need the original encoder file. It has a distinct model id while retaining the
structured checkpoint container. Existing model state keys and loaders remain valid.

## CNN fine-tuning with the same KC head

`finetuned_cnn_kcnet` shares every architecture setting and state key with
`fixed_filter_kcnet`. The difference is CNN trainability: the three convolutional
blocks and readout train, while FFD kernels, projection indices and numerical
buffers remain fixed. Gradients pass through pooling, safe feature normalization,
the fixed projection and inhibition/ReLU to the CNN. At the defaults, 97,057 of
97,075 neural parameters train.

For the CNN trainability comparison, both runs initialize from the same
validation-selected fixed-filter CNN checkpoint and the same freshly initialized
KC readout/projection. Use the same training/projection seeds, dataset splits,
optimizer, learning rate and epoch target. Starting fine-tuning from a KC head
already trained for 20 epochs would instead add a different warm-up stage.
The model id determines whether the CNN is frozen when restoring a checkpoint;
existing frozen KC checkpoints retain their original behavior.
