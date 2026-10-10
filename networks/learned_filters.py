"""Two-channel learned gradient features for the paper's CNN backbone.

The frozen and trainable variants share the same feature calculation.  Their
only difference is whether the two filter tensors receive optimizer updates.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .hog_cnn import DualCNN_128


class LearnedGradientFrontend(nn.Module):
    """Map raw grayscale images to normalized magnitude and angle channels.

    Inputs have shape ``[B, 1, H, W]`` and normally contain intensities in
    ``[0, 1]``.  Two unrestricted 3x3 kernels are initialized exactly as the
    existing ``RawGradMagDir`` transform, including its backward vertical
    difference.  Zero padding and independent min-max normalization for each
    image/channel also follow that transform.

    To make filter learning safe, magnitude is ``sqrt(energy + eps**2) - eps``.
    Direction is zero when response magnitude is at most ``angle_eps``; both
    arguments are replaced before evaluating atan2, avoiding its undefined
    derivative at the origin.  The default angle threshold, 1e-3 for [0, 1]
    intensities, bounds the angle derivative above that threshold.  Channels
    whose value range is at most ``normalization_eps`` produce zero features.
    These weak-response rules apply to both the frozen and trainable control
    and deliberately differ from the original transform near flat responses.

    Safety settings are saved as buffers so ordinary ``state_dict`` files can
    be loaded with ``weights_only=True``.  Half/bfloat16 inputs are processed
    in float32 to avoid underflow of squared epsilon values, then converted
    back to the input dtype.
    """

    def __init__(
        self,
        trainable=True,
        magnitude_eps=1e-6,
        angle_eps=1e-3,
        normalization_eps=1e-8,
    ):
        super().__init__()
        for name, value in (
            ("magnitude_eps", magnitude_eps),
            ("angle_eps", angle_eps),
            ("normalization_eps", normalization_eps),
        ):
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")

        self.Gx = nn.Parameter(
            torch.tensor(
                [[0.0, 0.0, 0.0], [0.0, -1.0, 1.0], [0.0, 0.0, 0.0]],
                dtype=torch.float32,
            ).reshape(1, 1, 3, 3),
            requires_grad=trainable,
        )
        self.Gy = nn.Parameter(
            torch.tensor(
                [[0.0, -1.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 0.0]],
                dtype=torch.float32,
            ).reshape(1, 1, 3, 3),
            requires_grad=trainable,
        )
        self.register_buffer(
            "_magnitude_eps", torch.tensor(magnitude_eps, dtype=torch.float32)
        )
        self.register_buffer("_angle_eps", torch.tensor(angle_eps, dtype=torch.float32))
        self.register_buffer(
            "_normalization_eps", torch.tensor(normalization_eps, dtype=torch.float32)
        )

    def _apply(self, fn, recurse=True):
        # Keep safety constants representable even when model.half() is used.
        # They still follow the module's device and remain checkpoint buffers.
        settings = {name: value.clone() for name, value in self._buffers.items()}
        super()._apply(fn, recurse=recurse)
        for name, value in settings.items():
            self._buffers[name] = value.to(
                device=self._buffers[name].device, dtype=torch.float32
            )
        return self

    def forward(self, x):
        if not isinstance(x, torch.Tensor) or not torch.is_floating_point(x):
            raise TypeError("Expected a floating-point grayscale tensor")
        if x.ndim != 4 or x.shape[1] != 1:
            raise ValueError("Expected input shape [B, 1, H, W]")
        if x.shape[0] < 1 or min(x.shape[-2:]) < 1:
            raise ValueError("Batch and spatial dimensions must be nonempty")

        input_dtype = x.dtype
        compute_dtype = (
            torch.float32
            if input_dtype in (torch.float16, torch.bfloat16)
            else input_dtype
        )
        with torch.autocast(device_type=x.device.type, enabled=False):
            x = x.to(dtype=compute_dtype)
            gx = F.conv2d(x, self.Gx.to(dtype=compute_dtype), padding=1)
            gy = F.conv2d(x, self.Gy.to(dtype=compute_dtype), padding=1)
            energy = gx.square() + gy.square()
            magnitude_eps = self._magnitude_eps.to(dtype=compute_dtype)
            angle_eps = self._angle_eps.to(dtype=compute_dtype)
            normalization_eps = self._normalization_eps.to(dtype=compute_dtype)

            magnitude = torch.sqrt(energy + magnitude_eps.square()) - magnitude_eps
            strong_response = energy > angle_eps.square()
            safe_gx = torch.where(strong_response, gx, torch.ones_like(gx))
            safe_gy = torch.where(strong_response, gy, torch.zeros_like(gy))
            angle = torch.atan2(safe_gy, safe_gx)

            features = torch.cat((magnitude, angle), dim=1)
            minimum = features.amin(dim=(-2, -1), keepdim=True)
            span = features.amax(dim=(-2, -1), keepdim=True) - minimum
            normalized = (features - minimum) / (span + normalization_eps)
            normalized = torch.where(
                span > normalization_eps, normalized, torch.zeros_like(normalized)
            )
        return normalized.to(dtype=input_dtype)


class LearnableFilterCNN128(DualCNN_128):
    """Raw grayscale -> two learned feature maps -> the unchanged 128/64 head.

    The CNN and classifier retain the existing ``cnn_branch`` and
    ``classifier`` state-dictionary keys.  ``trainable=False`` freezes only
    the frontend kernels, leaving the classifier and CNN trainable for the
    paired fixed-filter control.  No pretrained weights are required.
    """

    def __init__(
        self,
        trainable=True,
        img_size=224,
        magnitude_eps=1e-6,
        angle_eps=1e-3,
        normalization_eps=1e-8,
    ):
        if not isinstance(img_size, int) or img_size < 8:
            raise ValueError("img_size must be an integer of at least 8")
        super().__init__(mlp_dim=128, in_channels=2, img_size=img_size)
        self.img_size = img_size
        self.frontend = LearnedGradientFrontend(
            trainable=trainable,
            magnitude_eps=magnitude_eps,
            angle_eps=angle_eps,
            normalization_eps=normalization_eps,
        )

    def forward(self, x):
        features = self.frontend(x)
        if features.shape[-2:] != (self.img_size, self.img_size):
            raise ValueError(
                f"Expected spatial size {self.img_size}x{self.img_size}, "
                f"received {features.shape[-2]}x{features.shape[-1]}"
            )
        return super().forward(features)
