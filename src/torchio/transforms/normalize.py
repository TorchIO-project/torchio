"""Normalize: linearly map voxel intensities to a target range."""

from __future__ import annotations

import warnings
from collections.abc import Callable
from typing import Any

import torch
from torch import Tensor

from ..data.batch import ImagesBatch
from ..data.batch import SubjectsBatch
from ..data.image import LabelMap
from .parameter_range import Choice
from .parameter_range import ParameterRange
from .transform import IntensityTransform

TypeParameterValue = float | tuple | Choice | torch.distributions.Distribution


def _to_range(
    value: TypeParameterValue,
) -> ParameterRange:
    """Convert a scalar, tuple, Choice, or Distribution to a ParameterRange."""
    if isinstance(value, (torch.distributions.Distribution, Choice)):
        return ParameterRange(value)
    if isinstance(value, (int, float)):
        return ParameterRange(float(value))
    return ParameterRange(tuple(float(v) for v in value))


class Normalize(IntensityTransform):
    r"""Linearly rescale voxel intensities to a target range.

    The transform clips values to an input range, then applies the
    affine map:

    $$v_{\text{out}} = \frac{v - m_{\min}}{m_{\max} - m_{\min}}
    \cdot (n_{\max} - n_{\min}) + n_{\min}$$

    All six numeric parameters are independently randomizable via
    scalar, ``(low, high)`` range, or ``torch.distributions.Distribution``.

    Args:
        out_min: Lower bound of the output range.
        out_max: Upper bound of the output range.
        in_min: Lower bound of the input range. If ``None``, determined
            from *percentile_low* of the (masked) input data.
        in_max: Upper bound of the input range. If ``None``, determined
            from *percentile_high* of the (masked) input data.
        percentile_low: Lower percentile for auto input range.
        percentile_high: Upper percentile for auto input range.
            Use ``(0.5, 99.5)`` for the nn-UNet convention.
        masking_method: Which voxels to include when computing
            percentiles. ``None`` uses all voxels. A ``str`` is
            interpreted as a key to a
            [`LabelMap`][torchio.LabelMap] in the subject. A callable
            receives the image tensor and returns a boolean mask.
        **kwargs: See [`Transform`][torchio.Transform].

    Examples:
        >>> import torchio as tio
        >>> # Rescale to [-1, 1] (default)
        >>> transform = tio.Normalize()
        >>> # CT windowing
        >>> transform = tio.Normalize(
        ...     out_min=0.0, out_max=1.0,
        ...     in_min=-1000.0, in_max=1000.0,
        ... )
        >>> # nn-UNet percentile clipping
        >>> transform = tio.Normalize(
        ...     percentile_low=0.5, percentile_high=99.5,
        ... )
        >>> # Random output range
        >>> transform = tio.Normalize(
        ...     out_min=(-1.0, 0.0), out_max=(0.5, 1.0),
        ... )
    """

    def __init__(
        self,
        *,
        out_min: TypeParameterValue = -1.0,
        out_max: TypeParameterValue = 1.0,
        in_min: TypeParameterValue | None = None,
        in_max: TypeParameterValue | None = None,
        percentile_low: TypeParameterValue = 0.0,
        percentile_high: TypeParameterValue = 100.0,
        masking_method: str | Callable[[Tensor], Tensor] | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.out_min = _to_range(out_min)
        self.out_max = _to_range(out_max)
        self.in_min = _to_range(in_min) if in_min is not None else None
        self.in_max = _to_range(in_max) if in_max is not None else None
        self.percentile_low = _to_range(percentile_low)
        self.percentile_high = _to_range(percentile_high)
        self.masking_method = masking_method

    def make_params(self, batch: SubjectsBatch) -> dict[str, Any]:
        """Sample random parameters and compute the input range.

        Returns:
            Dict with ``out_min``, ``out_max``, ``in_min``, ``in_max``
            (per image name).
        """
        out_min = self.out_min.sample_1d()
        out_max = self.out_max.sample_1d()
        pct_low = self.percentile_low.sample_1d()
        pct_high = self.percentile_high.sample_1d()

        # If explicit in_min/in_max are given, sample them directly.
        if self.in_min is not None and self.in_max is not None:
            return {
                "out_min": out_min,
                "out_max": out_max,
                "in_min": self.in_min.sample_1d(),
                "in_max": self.in_max.sample_1d(),
            }

        # Otherwise, compute per-image input range from percentiles.
        images = self._get_images(batch)
        in_ranges: dict[str, tuple[float, float]] = {}
        for name, img_batch in images.items():
            mask = self._get_mask(img_batch, batch)
            in_min, in_max = _percentile_range(
                img_batch.data[0],
                mask,
                pct_low,
                pct_high,
                name,
            )
            in_ranges[name] = (in_min, in_max)

        return {
            "out_min": out_min,
            "out_max": out_max,
            "in_ranges": in_ranges,
        }

    def apply_transform(
        self,
        batch: SubjectsBatch,
        params: dict[str, Any],
    ) -> SubjectsBatch:
        """Clip and linearly rescale each selected image."""
        out_min = params["out_min"]
        out_max = params["out_max"]
        out_range = out_max - out_min

        for name, img_batch in self._get_images(batch).items():
            if "in_min" in params:
                in_min = params["in_min"]
                in_max = params["in_max"]
            else:
                in_ranges = params.get("in_ranges", {})
                if name not in in_ranges:
                    continue
                in_min, in_max = in_ranges[name]

            in_range = in_max - in_min
            if in_range == 0:
                warnings.warn(
                    f'Cannot rescale "{name}": input range is zero.',
                    RuntimeWarning,
                    stacklevel=2,
                )
                continue

            data = img_batch.data.float()
            data = data.clamp(in_min, in_max)
            data = (data - in_min) / in_range * out_range + out_min
            img_batch.data = data

        return batch

    @property
    def invertible(self) -> bool:
        """Whether this transform can be inverted."""
        return True

    def inverse(self, params: dict[str, Any]) -> _RescaleInverse:
        """Build the inverse transform from recorded parameters."""
        return _RescaleInverse(
            out_min=params["out_min"],
            out_max=params["out_max"],
            in_min=params.get("in_min"),
            in_max=params.get("in_max"),
            in_ranges=params.get("in_ranges"),
            copy=False,
        )

    def _get_mask(
        self,
        img_batch: ImagesBatch,
        batch: SubjectsBatch,
    ) -> Tensor | None:
        """Resolve masking_method to a boolean tensor or None."""
        if self.masking_method is None:
            return None
        if callable(self.masking_method) and not isinstance(self.masking_method, str):
            return self.masking_method(img_batch.data[0]).bool()
        # String key: look up a LabelMap in the batch.
        if isinstance(self.masking_method, str):
            key = self.masking_method
            if key not in batch.images:
                msg = (
                    f'Masking method "{key}" not found in batch images.'
                    f" Available: {list(batch.images.keys())}"
                )
                raise KeyError(msg)
            mask_batch = batch.images[key]
            if not issubclass(mask_batch._image_class, LabelMap):
                msg = f'Masking method "{key}" must refer to a LabelMap.'
                raise TypeError(msg)
            return mask_batch.data[0].bool()
        msg = (
            "masking_method must be None, str, or callable, got"
            f" {type(self.masking_method)}"
        )
        raise TypeError(msg)


class _RescaleInverse(IntensityTransform):
    """Inverse of Normalize for history replay."""

    def __init__(
        self,
        *,
        out_min: float,
        out_max: float,
        in_min: float | None,
        in_max: float | None,
        in_ranges: dict[str, tuple[float, float]] | None,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self._out_min = out_min
        self._out_max = out_max
        self._in_min = in_min
        self._in_max = in_max
        self._in_ranges = in_ranges

    def make_params(self, batch: SubjectsBatch) -> dict[str, Any]:
        """Return empty params; all state is in instance attributes."""
        return {}

    def apply_transform(
        self,
        batch: SubjectsBatch,
        params: dict[str, Any],
    ) -> SubjectsBatch:
        """Reverse the linear rescaling."""
        out_range = self._out_max - self._out_min

        for name, img_batch in self._get_images(batch).items():
            if self._in_min is not None and self._in_max is not None:
                in_min = self._in_min
                in_max = self._in_max
            elif self._in_ranges is not None and name in self._in_ranges:
                in_min, in_max = self._in_ranges[name]
            else:
                continue

            in_range = in_max - in_min
            if in_range == 0 or out_range == 0:
                continue

            data = img_batch.data.float()
            # Reverse: v_original = (v - out_min) / out_range * in_range + in_min
            data = (data - self._out_min) / out_range * in_range + in_min
            img_batch.data = data

        return batch


def _percentile_range(
    tensor: Tensor,
    mask: Tensor | None,
    pct_low: float,
    pct_high: float,
    image_name: str,
) -> tuple[float, float]:
    """Compute the input range from percentiles of (masked) data.

    Args:
        tensor: ``(C, I, J, K)`` image tensor (first sample).
        mask: Optional boolean mask with compatible shape, or ``None``.
        pct_low: Lower percentile (0-100).
        pct_high: Upper percentile (0-100).
        image_name: Used in warning messages.

    Returns:
        ``(in_min, in_max)`` tuple.
    """
    values = tensor[mask.expand_as(tensor)] if mask is not None else tensor.reshape(-1)

    if values.numel() == 0:
        warnings.warn(
            f'Cannot compute percentiles for "{image_name}": mask is empty.'
            " Using full range.",
            RuntimeWarning,
            stacklevel=3,
        )
        values = tensor.reshape(-1)

    low = float(torch.quantile(values.float(), pct_low / 100.0).item())
    high = float(torch.quantile(values.float(), pct_high / 100.0).item())
    return low, high


# Backwards-compatible alias.
RescaleIntensity = Normalize
