"""Blur: Gaussian smoothing augmentation."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
import torch.nn.functional as functional
from einops import rearrange

from ...data.batch import ImagesBatch
from ...data.batch import SubjectsBatch
from ..parameter_range import to_nonneg_range
from ..transform import IntensityTransform


class Blur(IntensityTransform):
    r"""Blur an image using a Gaussian filter.

    The standard deviations $(\sigma_1, \sigma_2, \sigma_3)$ of the
    Gaussian kernel along each spatial axis are independently sampled
    from the given range.  Sigmas are specified in mm and internally
    converted to voxels using the image spacing.

    Args:
        std: Standard deviation of the Gaussian kernel in mm.
            A scalar $x$ means $\sigma_i = x$ for every axis
            (deterministic).
            A 2-tuple $(a, b)$ means
            $\sigma_i \sim \mathcal{U}(a, b)$.
            A 6-tuple $(a_1, b_1, a_2, b_2, a_3, b_3)$ means
            $\sigma_i \sim \mathcal{U}(a_i, b_i)$ independently.
            A `Choice` or `Distribution` may also be passed.
            The default `std=0` is a no-op (and warns).
        **kwargs: See [`Transform`][torchio.Transform].

    Examples:
        >>> import torchio as tio
        >>> transform = tio.Blur(std=2.0)
        >>> transform = tio.Blur(std=(0, 4))
    """

    def __init__(
        self,
        *,
        std: float | tuple[float, float] = 0.0,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.std = to_nonneg_range(std)
        self._warn_if_noop(is_noop=self.std.is_constant(0.0), hint="std=(0, 2)")

    def make_params(self, batch: SubjectsBatch) -> dict[str, Any]:
        """Sample per-axis standard deviations (per element when batched)."""
        n = self._resolve_n(batch)
        if n is None:
            return {"std": self.std.sample()}
        keep = self._keep_mask(batch, n)
        std = self.std.sample(n)
        if keep is not None:
            std[~keep] = 0.0
        params = {"std": self._serialize_param(std)}
        self._tag_batched(params, batch, n, keep, ["std"])
        return params

    @property
    def supports_per_instance_params(self) -> bool:
        return True

    @property
    def supports_per_instance_p(self) -> bool:
        return True

    def apply_transform(
        self,
        batch: SubjectsBatch,
        params: dict[str, Any],
    ) -> SubjectsBatch:
        """Apply Gaussian smoothing to each selected image."""
        per_instance = self._is_per_instance_params(params)
        for _name, img_batch in self._get_images(batch).items():
            if per_instance:
                img_batch.data = _blur_per_element(img_batch, params["std"])
            else:
                spacing = np.asarray(img_batch.affines[0].spacing, dtype=np.float64)
                sigmas_vox = _sigmas_mm_to_voxels(params["std"], spacing)
                img_batch.data = _gaussian_smooth(img_batch.data, sigmas_vox)
        return batch


def _sigmas_mm_to_voxels(
    sigmas_mm: list[float],
    spacing: np.ndarray,
) -> list[float]:
    """Convert per-axis sigmas from mm to voxels."""
    return [s / sp if sp > 0 else 0.0 for s, sp in zip(sigmas_mm, spacing, strict=True)]


def _blur_per_element(
    img_batch: ImagesBatch,
    sigmas_mm_per_element: list[list[float]],
) -> torch.Tensor:
    """Blur each batch element with its own per-axis sigmas.

    Args:
        img_batch: The image batch to blur.
        sigmas_mm_per_element: One `[si, sj, sk]` (in mm) per element.

    Returns:
        The blurred `(B, C, I, J, K)` tensor.
    """
    data = img_batch.data
    outputs = []
    for index in range(data.shape[0]):
        spacing = np.asarray(img_batch.affines[index].spacing, dtype=np.float64)
        sigmas_vox = _sigmas_mm_to_voxels(sigmas_mm_per_element[index], spacing)
        outputs.append(_gaussian_smooth(data[index : index + 1], sigmas_vox))
    return torch.cat(outputs, dim=0)


def _gaussian_smooth(data: torch.Tensor, sigmas: list[float]) -> torch.Tensor:
    """Apply separable Gaussian smoothing to a 5D tensor.

    Args:
        data: `(B, C, I, J, K)` tensor.
        sigmas: Per-axis sigma in voxels.

    Returns:
        Smoothed tensor.
    """
    if all(s <= 0 for s in sigmas):
        return data
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
