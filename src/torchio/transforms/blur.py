"""Blur: Gaussian smoothing augmentation."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
import torch.nn.functional as functional
from einops import rearrange

from ..data.batch import SubjectsBatch
from .parameter_range import to_nonneg_range
from .transform import IntensityTransform


class Blur(IntensityTransform):
    r"""Blur an image using a Gaussian filter.

    The standard deviations $(\sigma_1, \sigma_2, \sigma_3)$ of the
    Gaussian kernel along each spatial axis are independently sampled
    from the given range.  Sigmas are specified in mm and internally
    converted to voxels using the image spacing.

    Args:
        std: Standard deviation of the Gaussian kernel in mm.
            A scalar $x$ means $\sigma_i \sim \mathcal{U}(0, x)$.
            A 2-tuple $(a, b)$ means
            $\sigma_i \sim \mathcal{U}(a, b)$.
            A 6-tuple $(a_1, b_1, a_2, b_2, a_3, b_3)$ means
            $\sigma_i \sim \mathcal{U}(a_i, b_i)$ independently.
            A ``Choice`` or ``Distribution`` may also be passed.
        **kwargs: See [`Transform`][torchio.Transform].

    Examples:
        >>> import torchio as tio
        >>> transform = tio.Blur(std=2.0)
        >>> transform = tio.Blur(std=(0, 4))
    """

    def __init__(
        self,
        *,
        std: float | tuple[float, float] = (0, 2),
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.std = to_nonneg_range(std)

    def make_params(self, batch: SubjectsBatch) -> dict[str, Any]:
        """Sample per-axis standard deviations."""
        return {"std": self.std.sample()}

    def apply_transform(
        self,
        batch: SubjectsBatch,
        params: dict[str, Any],
    ) -> SubjectsBatch:
        """Apply Gaussian smoothing to each selected image."""
        sigmas_mm = params["std"]
        for _name, img_batch in self._get_images(batch).items():
            spacing = np.asarray(img_batch.affines[0].spacing, dtype=np.float64)
            # Convert mm sigmas to voxel sigmas.
            sigmas_vox = [
                s / sp if sp > 0 else 0.0
                for s, sp in zip(sigmas_mm, spacing, strict=True)
            ]
            img_batch.data = _gaussian_smooth(img_batch.data, sigmas_vox)
        return batch


def _gaussian_smooth(data: torch.Tensor, sigmas: list[float]) -> torch.Tensor:
    """Apply separable Gaussian smoothing to a 5D tensor.

    Args:
        data: ``(B, C, I, J, K)`` tensor.
        sigmas: Per-axis sigma in voxels.

    Returns:
        Smoothed tensor.
    """
    result = data.float()
    b, c = result.shape[:2]
    for axis_idx in range(3):
        sigma = sigmas[axis_idx]
        if sigma <= 0:
            continue
        radius = max(int(np.ceil(3 * sigma)), 1)
        kernel_size = 2 * radius + 1
        x = torch.arange(kernel_size, dtype=torch.float32, device=data.device) - radius
        kernel_1d = torch.exp(-0.5 * (x / sigma) ** 2)
        kernel_1d = kernel_1d / kernel_1d.sum()

        k_shape = [1, 1, 1]
        k_shape[axis_idx] = kernel_size
        kernel_3d = kernel_1d.reshape(1, 1, *k_shape).expand(1, 1, -1, -1, -1)

        # Replicate-pad along the target axis.
        pad = [0] * 6
        pad_idx = 2 * (2 - axis_idx)
        pad[pad_idx] = radius
        pad[pad_idx + 1] = radius

        padded = functional.pad(result, pad, mode="replicate")
        result = functional.conv3d(
            rearrange(padded, "b c i j k -> (b c) 1 i j k"),
            kernel_3d,
            padding=0,
        )
        result = rearrange(result, "(b c) 1 i j k -> b c i j k", b=b, c=c)
    return result.to(data.dtype)
