"""Fixed gradient frontend and KC readout with frozen or fine-tuned CNN features."""

import math

import torch
from torch import nn
from torch.nn import functional as F

from .hog_cnn import SimpleCNN
from .learned_filters import LearnedGradientFrontend


KCNET_DEFAULTS = {"pool_size": 4, "kc_units": 4096, "fan_in": 7,
                  "projection_seed": 42, "feature_eps": 1e-8}


class FixedFilterKCNet(nn.Module):
    """The readout trains; optional CNN fine-tuning preserves the fixed frontend.

    Pooling retains coarse spatial information. Each Kenyon unit sums a fixed
    random set of input features, then mean inhibition and ReLU produce sparse
    activity. Activity is not forced to a top-k fraction. Connection indices and
    the feature-normalization floor are checkpoint buffers, not learned weights.
    """

    def __init__(self, img_size=224, pool_size=4, kc_units=4096, fan_in=7,
                 projection_seed=42, feature_eps=1e-8, train_cnn=False):
        super().__init__()
        if not isinstance(train_cnn, bool):
            raise ValueError("train_cnn must be a boolean")
        self.train_cnn = train_cnn
        for name, value, minimum in (("img_size", img_size, 8), ("pool_size", pool_size, 1),
                                     ("kc_units", kc_units, 1), ("fan_in", fan_in, 1)):
            if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
                raise ValueError(f"{name} must be an integer >= {minimum}")
        if pool_size > img_size // 8:
            raise ValueError("pool_size cannot exceed the CNN feature-map size")
        if not isinstance(projection_seed, int) or not 0 <= projection_seed < 2**32:
            raise ValueError("projection_seed must be in [0, 2**32)")
        if not math.isfinite(feature_eps) or feature_eps <= 0:
            raise ValueError("feature_eps must be finite and positive")
        epsilon = torch.tensor(feature_eps, dtype=torch.float32)
        if not torch.isfinite(epsilon) or epsilon <= 0:
            raise ValueError("feature_eps must be representable as positive float32")
        self.img_size = img_size
        self.input_features = 128 * pool_size**2
        if fan_in > self.input_features:
            raise ValueError("fan_in cannot exceed the pooled feature count")
        self.cnn_branch = SimpleCNN(in_channels=2, img_size=img_size)
        self.frontend = LearnedGradientFrontend(trainable=False)
        for parameter in self.cnn_branch.parameters():
            parameter.requires_grad_(train_cnn)
        self.pool = nn.AdaptiveAvgPool2d((pool_size, pool_size))
        generator = torch.Generator().manual_seed(projection_seed)
        self.register_buffer("projection_indices", torch.stack([
            torch.randperm(self.input_features, generator=generator)[:fan_in]
            for _ in range(kc_units)
        ]))
        self.register_buffer("_feature_eps", epsilon)
        self.readout = nn.Linear(kc_units, 1)
        self.train(self.training)

    def train(self, mode=True):
        super().train(mode)
        self.frontend.eval()
        if not self.train_cnn:
            self.cnn_branch.eval()
        return self

    def _apply(self, fn, recurse=True):
        epsilon = self._feature_eps.clone()
        super()._apply(fn, recurse=recurse)
        self._feature_eps = epsilon.to(device=self._feature_eps.device, dtype=torch.float32)
        return self

    def project_features(self, features):
        if features.ndim != 2 or features.shape[1] != self.input_features:
            raise ValueError(f"Expected a [B, {self.input_features}] feature vector")
        if features.dtype in (torch.float16, torch.bfloat16):
            features = features.float()
        with torch.autocast(device_type=features.device.type, enabled=False):
            epsilon = self._feature_eps.to(dtype=features.dtype)
            normalized = features / features.norm(p=2, dim=1, keepdim=True).clamp_min(epsilon)
            responses = normalized[:, self.projection_indices].sum(-1)
            return F.relu(responses - responses.mean(dim=1, keepdim=True))

    def encode(self, images):
        with torch.set_grad_enabled(torch.is_grad_enabled() and self.train_cnn):
            maps = self.frontend(images)
            if maps.shape[-2:] != (self.img_size, self.img_size):
                raise ValueError(f"Expected spatial size {self.img_size}x{self.img_size}")
            features = self.cnn_branch.features(maps)
            return self.project_features(self.pool(features).flatten(1))

    def forward(self, images):
        return self.readout(self.encode(images).to(dtype=self.readout.weight.dtype))
