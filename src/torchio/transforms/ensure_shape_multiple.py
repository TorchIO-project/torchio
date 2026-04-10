"""EnsureShapeMultiple: pad or crop so spatial dims are divisible by n."""

from __future__ import annotations

import math
from typing import Any

from ..data.batch import SubjectsBatch
from ..data.image import Image
from ..data.subject import Subject
from ..types import TypeThreeInts
from .crop_or_pad import CropOrPad
from .transform import SpatialTransform

#: Accepted target_multiple specifications.
#: ``int`` → same value for all axes.
#: 3-tuple → per-axis values.
TargetMultipleParam = int | TypeThreeInts


def _parse_target_multiple(value: TargetMultipleParam) -> TypeThreeInts:
    """Normalise target_multiple to a 3-tuple of positive ints."""
    if isinstance(value, int):
        if value < 1:
            msg = f"target_multiple must be >= 1, got {value}"
            raise ValueError(msg)
        return (value, value, value)
    values = tuple(value)
    if len(values) != 3:
        msg = f"target_multiple must have 1 or 3 values, got {len(values)}"
        raise ValueError(msg)
    for v in values:
        if v < 1:
            msg = f"All target_multiple values must be >= 1, got {v}"
            raise ValueError(msg)
    return (values[0], values[1], values[2])


def _compute_target_shape(
    current_shape: TypeThreeInts,
    target_multiple: TypeThreeInts,
    method: str,
) -> TypeThreeInts:
    """Compute the target shape so each axis is a multiple of target_multiple."""
    result: list[int] = []
    for size, multiple in zip(current_shape, target_multiple, strict=True):
        if method == "pad":
            target = math.ceil(size / multiple) * multiple
        else:
            target = math.floor(size / multiple) * multiple
        target = max(target, 1)
        result.append(target)
    return (result[0], result[1], result[2])


class EnsureShapeMultiple(SpatialTransform):
    r"""Ensure that all values in the image shape are divisible by $n$.

    Some convolutional neural network architectures need the size of the
    input across all spatial dimensions to be a power of 2.

    For example, a 3D U-Net with 3 downsampling (pooling) operations
    needs all spatial dimensions to be multiples of $2^3 = 8$.

    This transform computes the nearest valid shape and delegates to
    [`CropOrPad`][torchio.CropOrPad] to reach it.

    Args:
        target_multiple: Tuple $(n_i, n_j, n_k)$ so that the output
            size along axis $d$ is a multiple of $n_d$. If a single
            value $n$ is provided, then $n_i = n_j = n_k = n$.
        method: Either ``'pad'`` (default) to pad up to the next
            multiple, or ``'crop'`` to crop down to the previous
            multiple.
        padding_mode: Padding mode forwarded to ``CropOrPad`` when
            ``method='pad'``. One of ``'constant'``, ``'reflect'``,
            ``'replicate'``, or ``'circular'``.
        fill: Fill value when ``padding_mode='constant'``.
        **kwargs: See [`Transform`][torchio.Transform] for additional
            keyword arguments.

    Examples:
        >>> import torchio as tio
        >>> transform = tio.EnsureShapeMultiple(8)
        >>> transform = tio.EnsureShapeMultiple(2**3, method='pad')
        >>> transform = tio.EnsureShapeMultiple(16, method='crop')
        >>> transform = tio.EnsureShapeMultiple((4, 8, 16))
    """

    def __init__(
        self,
        target_multiple: TargetMultipleParam,
        *,
        method: str = "pad",
        padding_mode: str = "constant",
        fill: float = 0,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.target_multiple = _parse_target_multiple(target_multiple)
        if method not in ("crop", "pad"):
            msg = f"method must be 'crop' or 'pad', got {method!r}"
            raise ValueError(msg)
        self.method = method
        self.padding_mode = padding_mode
        self.fill = fill

    def forward(self, data: Any) -> Any:
        """Apply the transform.

        For ``Subject`` and ``Image`` inputs, delegates to ``CropOrPad``
        for lazy operation without loading data from disk.
        """
        if isinstance(data, (Subject, Image)):
            return self._build_crop_or_pad(data).forward(data)
        return super().forward(data)

    def _build_crop_or_pad(self, data: Subject | Image) -> CropOrPad:
        """Build a CropOrPad targeting the nearest valid shape."""
        if isinstance(data, Image):
            current_shape = data.spatial_shape
        else:
            current_shape = data.spatial_shape
        target_shape = _compute_target_shape(
            current_shape,
            self.target_multiple,
            self.method,
        )
        return CropOrPad(
            target_shape=target_shape,
            padding_mode=self.padding_mode,
            fill=self.fill,
            only_crop=self.method == "crop",
            only_pad=self.method == "pad",
            p=self.p,
            copy=self.copy,
            include=self.include,
            exclude=self.exclude,
        )

    def make_params(self, batch: SubjectsBatch) -> dict[str, Any]:
        first_images = next(iter(batch.images.values()))
        data_tensor = first_images.data
        current_shape: TypeThreeInts = (
            data_tensor.shape[-3],
            data_tensor.shape[-2],
            data_tensor.shape[-1],
        )
        target_shape = _compute_target_shape(
            current_shape,
            self.target_multiple,
            self.method,
        )
        return {"target_shape": target_shape}

    def apply_transform(
        self,
        batch: SubjectsBatch,
        params: dict[str, Any],
    ) -> SubjectsBatch:
        target_shape = params["target_shape"]
        crop_or_pad = CropOrPad(
            target_shape=target_shape,
            padding_mode=self.padding_mode,
            fill=self.fill,
            only_crop=self.method == "crop",
            only_pad=self.method == "pad",
            copy=False,
            include=self.include,
            exclude=self.exclude,
        )
        return crop_or_pad.apply_transform(
            batch,
            crop_or_pad.make_params(batch),
        )
