"""Blur: Gaussian smoothing augmentation."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
import torch.nn.functional as functional
from einops import rearrange
from einops import repeat

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
    sigmas_mm = np.asarray(sigmas_mm_per_element, dtype=np.float64)
    spacings = np.asarray(
        [affine.spacing for affine in img_batch.affines],
        dtype=np.float64,
    )
    sigmas_vox = np.divide(
        sigmas_mm,
        spacings,
        out=np.zeros_like(sigmas_mm),
        where=spacings > 0,
    )
    return _gaussian_smooth(data, sigmas_vox)


def _gaussian_smooth(
    data: torch.Tensor,
    sigmas: list[float] | np.ndarray,
) -> torch.Tensor:
    """Apply separable Gaussian smoothing to a 5D tensor.

    Args:
        data: `(B, C, I, J, K)` tensor.
        sigmas: Per-axis sigma in voxels, or per-element per-axis sigmas
            with shape `(B, 3)`. Zero means skip that axis.

    Returns:
        Smoothed tensor.
    """
    sigmas_array = np.asarray(sigmas, dtype=np.float64)
    if np.all(sigmas_array <= 0):
        return data
    if sigmas_array.ndim == 1:
        # Sigmas shared across the batch: build one kernel set and apply it
        # to every element, which is much cheaper than a grouped conv.
        return _gaussian_smooth_shared(data, sigmas_array)
    if np.all(sigmas_array == sigmas_array[0]):
        # Per-element sigmas that happen to be identical collapse to the
        # shared fast path.
        return _gaussian_smooth_shared(data, sigmas_array[0])
    return _gaussian_smooth_per_element(data, sigmas_array)


def _gaussian_smooth_shared(
    data: torch.Tensor,
    sigmas: np.ndarray,
) -> torch.Tensor:
    """Separable Gaussian smoothing with a single shared kernel set.

    Args:
        data: `(B, C, I, J, K)` tensor.
        sigmas: Per-axis sigma in voxels (length 3), shared by every
            batch element.

    Returns:
        Smoothed tensor (input dtype preserved).
    """
    if np.all(sigmas <= 0):
        return data
    result = data.float()
    b, c = result.shape[:2]
    for axis_idx in range(3):
        sigma = float(sigmas[axis_idx])
        if sigma <= 0:
            continue
        radius = max(int(np.ceil(3 * sigma)), 1)
        kernel_size = 2 * radius + 1
        x = torch.arange(kernel_size, dtype=torch.float32, device=data.device) - radius
        kernel_1d = torch.exp(-0.5 * (x / sigma) ** 2)
        kernel_1d = kernel_1d / kernel_1d.sum()

        k_shape = [1, 1, 1]
        k_shape[axis_idx] = kernel_size
        kernel_3d = kernel_1d.reshape(1, 1, *k_shape)

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


def _gaussian_smooth_per_element(
    data: torch.Tensor,
    sigmas: np.ndarray,
) -> torch.Tensor:
    """Apply Gaussian smoothing with per-element sigmas.

    Args:
        data: `(B, C, I, J, K)` tensor.
        sigmas: Per-element per-axis sigmas in voxels with shape `(B, 3)`.

    Returns:
        Smoothed tensor with the same dtype as `data`.
    """
    result = data.float()
    b, c = result.shape[:2]
    no_blur_rows = np.all(sigmas <= 0, axis=1)
    for axis_idx in range(3):
        axis_sigmas = sigmas[:, axis_idx]
        if np.all(axis_sigmas <= 0):
            continue
        kernel_3d, radius = _make_grouped_axis_kernel(
            axis_sigmas,
            axis_idx,
            c,
            data.device,
        )

        # Replicate-pad along the target axis.
        pad = [0] * 6
        pad_idx = 2 * (2 - axis_idx)
        pad[pad_idx] = radius
        pad[pad_idx + 1] = radius

        padded = functional.pad(result, pad, mode="replicate")
        result = functional.conv3d(
            rearrange(padded, "b c i j k -> 1 (b c) i j k"),
            kernel_3d,
            groups=b * c,
            padding=0,
        )
        result = rearrange(result, "1 (b c) i j k -> b c i j k", b=b, c=c)
    result = result.to(data.dtype)
    if no_blur_rows.any():
        no_blur_mask = torch.as_tensor(no_blur_rows, device=data.device)
        result[no_blur_mask] = data[no_blur_mask]
    return result


def _make_grouped_axis_kernel(
    sigmas: np.ndarray,
    axis_idx: int,
    channels: int,
    device: torch.device,
) -> tuple[torch.Tensor, int]:
    """Build one grouped 3D convolution kernel for one spatial axis.

    Args:
        sigmas: Per-element sigma in voxels for the target axis.
        axis_idx: Spatial axis index, from 0 to 2.
        channels: Number of image channels per batch element.
        device: Device on which the kernel should be allocated.

    Returns:
        The `(B * C, 1, kI, kJ, kK)` grouped convolution kernel and the
            maximum radius used to pad the input.
    """
    radii = np.zeros_like(sigmas, dtype=np.int64)
    positive = sigmas > 0
    radii[positive] = np.maximum(np.ceil(3 * sigmas[positive]).astype(np.int64), 1)
    max_radius = int(radii.max())
    kernel_1d = _make_stacked_1d_kernels(sigmas, radii, max_radius, device)
    if axis_idx == 0:
        kernel_3d = rearrange(kernel_1d, "b i -> b 1 i 1 1")
    elif axis_idx == 1:
        kernel_3d = rearrange(kernel_1d, "b j -> b 1 1 j 1")
    else:
        kernel_3d = rearrange(kernel_1d, "b k -> b 1 1 1 k")
    kernel_3d = repeat(
        kernel_3d,
        "b one i j k -> (b c) one i j k",
        c=channels,
    )
    return kernel_3d, max_radius


def _make_stacked_1d_kernels(
    sigmas: np.ndarray,
    radii: np.ndarray,
    max_radius: int,
    device: torch.device,
) -> torch.Tensor:
    """Build centered 1D Gaussian or identity kernels.

    Args:
        sigmas: Per-element sigma in voxels for one axis.
        radii: Per-element kernel radii.
        max_radius: Maximum radius across all elements for the axis.
        device: Device on which the kernels should be allocated.

    Returns:
        A `(B, 2 * max_radius + 1)` tensor of normalized 1D kernels.
    """
    kernel_size = 2 * max_radius + 1
    offsets = torch.arange(kernel_size, dtype=torch.float32, device=device) - max_radius
    sigmas_tensor = torch.as_tensor(sigmas, dtype=torch.float32, device=device)
    radii_tensor = torch.as_tensor(radii, device=device)
    offsets_row = rearrange(offsets, "k -> 1 k")
    sigmas_column = rearrange(sigmas_tensor, "b -> b 1")
    radii_column = rearrange(radii_tensor, "b -> b 1")
    safe_sigmas = torch.where(
        sigmas_column > 0,
        sigmas_column,
        torch.ones_like(sigmas_column),
    )
    kernels = torch.exp(-0.5 * (offsets_row / safe_sigmas) ** 2)
    within_radius = torch.abs(offsets_row) <= radii_column
    kernels = torch.where(within_radius, kernels, torch.zeros_like(kernels))

    delta = torch.zeros_like(kernels)
    delta[:, max_radius] = 1.0
    kernels = torch.where(sigmas_column > 0, kernels, delta)
    return kernels / kernels.sum(dim=1, keepdim=True)
