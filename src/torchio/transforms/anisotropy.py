"""Anisotropy: simulate low-resolution acquisition along one axis."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as functional

from ..data.batch import SubjectsBatch
from ..data.image import LabelMap
from .parameter_range import to_nonneg_range
from .transform import Transform


class Anisotropy(Transform):
    r"""Simulate an anisotropic acquisition.

    Downsample along a randomly chosen axis and then upsample back
    to the original shape, emulating the through-plane blur seen in
    clinical MRI when one axis has coarser resolution.

    This is useful as a data augmentation for super-resolution
    training.

    Args:
        axes: Spatial axes eligible for downsampling.  One is chosen
            at random per application.
        downsampling: Downsampling factor $m > 1$.  A 2-tuple
            $(a, b)$ samples $m \sim \mathcal{U}(a, b)$.
        image_interpolation: Interpolation mode used when upsampling
            scalar images back to the original shape.
        **kwargs: See [`Transform`][torchio.Transform].

    Examples:
        >>> import torchio as tio
        >>> transform = tio.Anisotropy()
        >>> transform = tio.Anisotropy(
        ...     axes=(2,),
        ...     downsampling=4,
        ... )
    """

    def __init__(
        self,
        *,
        axes: tuple[int, ...] = (0, 1, 2),
        downsampling: float | tuple[float, float] = (1.5, 5.0),
        image_interpolation: str = "linear",
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.axes = axes
        self.downsampling = to_nonneg_range(downsampling)
        self.image_interpolation = image_interpolation
        self._validate_downsampling()

    def _validate_downsampling(self) -> None:
        """Ensure the range produces factors >= 1."""
        _lo, hi = self.downsampling._ranges[0]
        if hi < 1.0:
            msg = f"downsampling range upper bound must be >= 1, got {hi}"
            raise ValueError(msg)

    def make_params(self, batch: SubjectsBatch) -> dict[str, Any]:
        """Sample axis and downsampling factor."""
        axis = self.axes[int(torch.randint(len(self.axes), (1,)).item())]
        factor = max(1.0, self.downsampling.sample_1d())
        return {"axis": axis, "factor": factor}

    def apply_transform(
        self,
        batch: SubjectsBatch,
        params: dict[str, Any],
    ) -> SubjectsBatch:
        """Downsample then upsample along the chosen axis."""
        axis = params["axis"]
        factor = params["factor"]
        if factor <= 1.0:
            return batch
        for _name, img_batch in batch.images.items():
            is_label = issubclass(img_batch._image_class, LabelMap)
            img_batch.data = _simulate_anisotropy(
                img_batch.data,
                axis=axis,
                factor=factor,
                mode="nearest" if is_label else self.image_interpolation,
            )
        return batch


def _simulate_anisotropy(
    data: torch.Tensor,
    *,
    axis: int,
    factor: float,
    mode: str,
) -> torch.Tensor:
    """Downsample then upsample one axis of a 5-D tensor.

    Args:
        data: ``(B, C, I, J, K)`` tensor.
        axis: Spatial axis (0, 1, or 2) to degrade.
        factor: Downsampling factor (> 1).
        mode: Interpolation mode for upsampling (``"nearest"`` or
            ``"linear"``).

    Returns:
        Degraded ``(B, C, I, J, K)`` tensor with original shape.
    """
    original_shape = list(data.shape[2:])
    down_shape = list(original_shape)
    down_shape[axis] = max(1, round(original_shape[axis] / factor))

    torch_mode_down = "nearest"
    torch_mode_up = "nearest" if mode == "nearest" else "trilinear"

    # Downsample.
    downsampled = functional.interpolate(
        data.float(),
        size=down_shape,
        mode=torch_mode_down,
    )
    # Upsample back.
    upsampled = functional.interpolate(
        downsampled,
        size=original_shape,
        mode=torch_mode_up,
        align_corners=None if torch_mode_up == "nearest" else True,
    )
    return upsampled.to(data.dtype)
