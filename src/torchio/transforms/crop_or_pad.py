"""CropOrPad transform: crop and/or pad to a target shape."""

from __future__ import annotations

import math
from typing import Any
from typing import Literal

from loguru import logger

from ..data.batch import SubjectsBatch
from ..types import TypeSixInts
from ..types import TypeSpacing
from ..types import TypeThreeInts
from .compose import Compose
from .crop import Crop
from .pad import Pad
from .transform import SpatialTransform

#: Accepted target shape specifications.
#: ``int`` → same size for each axis.
#: 3-tuple of int or float → per axis.
TargetShapeParam = int | TypeThreeInts | tuple[float, float, float]

#: Accepted unit values.
Units = Literal["voxels", "mm", "cm"]


def _parse_target_shape(
    target_shape: TargetShapeParam,
) -> tuple[float, float, float]:
    """Normalise target_shape to a 3-tuple of floats."""
    if isinstance(target_shape, (int, float)):
        return (float(target_shape), float(target_shape), float(target_shape))
    values = list(target_shape)
    n = len(values)
    if n == 3:
        return (float(values[0]), float(values[1]), float(values[2]))
    msg = f"target_shape must have 1 or 3 values, got {n}"
    raise ValueError(msg)


def _to_voxels(
    target: tuple[float, float, float],
    units: Units,
    spacing: TypeSpacing,
) -> TypeThreeInts:
    """Convert a target shape from the given units to integer voxels."""
    if units == "voxels":
        return (round(target[0]), round(target[1]), round(target[2]))
    factor = 10.0 if units == "cm" else 1.0
    return (
        round(target[0] * factor / spacing[0]),
        round(target[1] * factor / spacing[1]),
        round(target[2] * factor / spacing[2]),
    )


def _compute_crop_and_pad(
    current_shape: TypeThreeInts,
    target_shape: TypeThreeInts,
    *,
    only_crop: bool,
    only_pad: bool,
) -> tuple[TypeSixInts | None, TypeSixInts | None]:
    """Compute per-side crop and pad amounts to go from current to target.

    Returns:
        ``(padding_six, cropping_six)`` — either may be ``None`` when no
        padding or cropping is needed (or when ``only_crop`` / ``only_pad``
        suppress it).
    """
    pad_values: list[int] = []
    crop_values: list[int] = []
    for cur, tgt in zip(current_shape, target_shape, strict=True):
        diff = tgt - cur
        if diff > 0:
            # Need to pad
            ini = math.ceil(diff / 2)
            fin = math.floor(diff / 2)
            pad_values.extend([ini, fin])
            crop_values.extend([0, 0])
        elif diff < 0:
            # Need to crop
            amount = -diff
            ini = math.ceil(amount / 2)
            fin = math.floor(amount / 2)
            pad_values.extend([0, 0])
            crop_values.extend([ini, fin])
        else:
            pad_values.extend([0, 0])
            crop_values.extend([0, 0])

    has_padding = any(v > 0 for v in pad_values)
    has_cropping = any(v > 0 for v in crop_values)

    padding: TypeSixInts | None = None
    if has_padding and not only_crop:
        padding = (
            pad_values[0],
            pad_values[1],
            pad_values[2],
            pad_values[3],
            pad_values[4],
            pad_values[5],
        )

    cropping: TypeSixInts | None = None
    if has_cropping and not only_pad:
        cropping = (
            crop_values[0],
            crop_values[1],
            crop_values[2],
            crop_values[3],
            crop_values[4],
            crop_values[5],
        )

    return padding, cropping


class CropOrPad(SpatialTransform):
    r"""Crop and/or pad to a target spatial shape.

    If the current spatial size along an axis is larger than the target, that
    axis is cropped symmetrically from both sides. If it is smaller, it is
    padded symmetrically. The affine matrix is updated so that physical
    positions of the voxels are maintained.

    The target shape can be specified in voxels (the default), millimetres,
    or centimetres. When physical units are used, the target is converted to
    voxels at transform time using the image spacing.

    Args:
        target_shape: Desired spatial shape. A single ``int`` broadcasts
            to all three axes. When ``units`` is ``"mm"`` or ``"cm"``,
            values may be floats representing the physical extent along
            each axis.
        units: Coordinate system for ``target_shape``. One of
            ``"voxels"`` (default), ``"mm"``, or ``"cm"``.
        padding_mode: One of ``'constant'``, ``'reflect'``,
            ``'replicate'``, or ``'circular'``. See
            [`torch.nn.functional.pad`](https://pytorch.org/docs/stable/generated/torch.nn.functional.pad.html).
        fill: Fill value when ``padding_mode='constant'``.
        only_crop: If ``True``, padding is never applied. Mutually
            exclusive with ``only_pad``.
        only_pad: If ``True``, cropping is never applied. Mutually
            exclusive with ``only_crop``.
        **kwargs: See [`Transform`][torchio.Transform] for additional
            keyword arguments.

    Examples:
        >>> import torchio as tio
        >>> transform = tio.CropOrPad(target_shape=(120, 80, 180))
        >>> transform = tio.CropOrPad(target_shape=256)
        >>> transform = tio.CropOrPad(target_shape=(150.0, 200.0, 180.0), units='mm')
        >>> transform = tio.CropOrPad(target_shape=(15.0, 20.0, 18.0), units='cm')
        >>> transform = tio.CropOrPad(target_shape=256, only_pad=True)
    """

    def __init__(
        self,
        target_shape: TargetShapeParam,
        *,
        units: Units = "voxels",
        padding_mode: str = "constant",
        fill: float = 0,
        only_crop: bool = False,
        only_pad: bool = False,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        if only_crop and only_pad:
            msg = "only_crop and only_pad cannot both be True"
            raise ValueError(msg)
        if units not in ("voxels", "mm", "cm"):
            msg = f"units must be 'voxels', 'mm', or 'cm', got {units!r}"
            raise ValueError(msg)
        self.target_shape = _parse_target_shape(target_shape)
        self.units = units
        self.padding_mode = padding_mode
        self.fill = fill
        self.only_crop = only_crop
        self.only_pad = only_pad

    def make_params(self, batch: SubjectsBatch) -> dict[str, Any]:
        first_images = next(iter(batch.images.values()))
        spacing = first_images.affines[0].spacing
        target_voxels = _to_voxels(self.target_shape, self.units, spacing)

        data_tensor = first_images.data
        current_shape: TypeThreeInts = (
            data_tensor.shape[-3],
            data_tensor.shape[-2],
            data_tensor.shape[-1],
        )

        if self.units != "voxels":
            logger.debug(
                "CropOrPad target {} {} → {} voxels (spacing {} mm)",
                self.target_shape,
                self.units,
                target_voxels,
                spacing,
            )

        padding, cropping = _compute_crop_and_pad(
            current_shape,
            target_voxels,
            only_crop=self.only_crop,
            only_pad=self.only_pad,
        )

        return {"padding": padding, "cropping": cropping}

    def apply_transform(
        self,
        batch: SubjectsBatch,
        params: dict[str, Any],
    ) -> SubjectsBatch:
        padding: TypeSixInts | None = params["padding"]
        cropping: TypeSixInts | None = params["cropping"]

        transforms: list[SpatialTransform] = []
        if padding is not None:
            transforms.append(
                Pad(
                    padding=padding,
                    padding_mode=self.padding_mode,
                    fill=self.fill,
                    include=self.include,
                    exclude=self.exclude,
                )
            )
        if cropping is not None:
            transforms.append(
                Crop(
                    cropping=cropping,
                    include=self.include,
                    exclude=self.exclude,
                )
            )

        if transforms:
            pipeline = Compose(transforms, copy=False)
            batch = pipeline(batch)

        return batch
