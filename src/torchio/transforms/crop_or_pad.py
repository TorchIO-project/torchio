"""CropOrPad transform: crop and/or pad to a target shape."""

from __future__ import annotations

import copy as _copy
import math
from typing import Any
from typing import Literal

import numpy as np
import torch
from loguru import logger
from torch import Tensor

from ..data.affine import Affine
from ..data.backends import ImageDataBackend
from ..data.batch import SubjectsBatch
from ..data.image import Image
from ..data.subject import Subject
from ..types import SliceIndex
from ..types import TypeAffineMatrix
from ..types import TypeSixInts
from ..types import TypeSpacing
from ..types import TypeTensorShape
from ..types import TypeThreeInts
from .compose import Compose
from .crop import Crop
from .pad import Pad
from .transform import AppliedTransform
from .transform import SpatialTransform

#: Accepted target shape specifications.
#: ``int`` or ``float`` → same size for each axis.
#: 3-tuple → per axis; use ``None`` to leave an axis unchanged.
TargetShapeParam = (
    int | float | tuple[int | float | None, int | float | None, int | float | None]
)

#: Accepted unit values.
Units = Literal["voxels", "mm", "cm"]


def _parse_target_shape(
    target_shape: TargetShapeParam,
) -> tuple[float | None, float | None, float | None]:
    """Normalise target_shape to a 3-tuple of floats or None."""
    if isinstance(target_shape, (int, float)):
        return (float(target_shape), float(target_shape), float(target_shape))
    values = list(target_shape)
    n = len(values)
    if n == 3:
        a, b, c = values
        return (
            None if a is None else float(a),
            None if b is None else float(b),
            None if c is None else float(c),
        )
    msg = f"target_shape must have 1 or 3 values, got {n}"
    raise ValueError(msg)


def _to_voxels(
    target: tuple[float | None, float | None, float | None],
    units: Units,
    spacing: TypeSpacing,
    current_shape: TypeThreeInts,
) -> TypeThreeInts:
    """Convert a target shape from the given units to integer voxels.

    ``None`` entries are replaced with the current size along that axis.
    """
    result: list[int] = []
    for t, sp, cur in zip(target, spacing, current_shape, strict=True):
        if t is None:
            result.append(cur)
        elif units == "voxels":
            result.append(round(t))
        else:
            factor = 10.0 if units == "cm" else 1.0
            result.append(round(t * factor / sp))
    return (result[0], result[1], result[2])


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


class _CroppedBackend:
    """Backend wrapper that defers spatial cropping until data is accessed."""

    __slots__ = ("_shape", "_source", "_spatial_slices")

    def __init__(
        self,
        source: ImageDataBackend,
        spatial_slices: tuple[slice, slice, slice],
        cropped_shape: tuple[int, int, int, int],
    ) -> None:
        self._source = source
        self._spatial_slices = spatial_slices
        self._shape = cropped_shape

    @property
    def shape(self) -> TypeTensorShape:
        return self._shape

    @property
    def affine(self) -> TypeAffineMatrix:
        return self._source.affine

    @property
    def dtype(self) -> np.dtype:
        return self._source.dtype

    def to_tensor(self) -> Tensor:
        slices = (slice(None), *self._spatial_slices)
        array = self._source[slices]
        return torch.tensor(array, dtype=torch.float32)

    def __getitem__(self, slices: SliceIndex) -> np.ndarray:
        tensor = self.to_tensor()
        if not isinstance(slices, tuple):
            slices = (slices,)
        return tensor[slices].numpy()


class _PaddedBackend:
    """Backend wrapper that defers spatial padding until data is accessed."""

    __slots__ = ("_fill", "_padding", "_padding_mode", "_shape", "_source")

    def __init__(
        self,
        source: ImageDataBackend,
        padding: TypeSixInts,
        padded_shape: tuple[int, int, int, int],
        padding_mode: str = "constant",
        fill: float = 0,
    ) -> None:
        self._source = source
        self._padding = padding
        self._shape = padded_shape
        self._padding_mode = padding_mode
        self._fill = fill

    @property
    def shape(self) -> TypeTensorShape:
        return self._shape

    @property
    def affine(self) -> TypeAffineMatrix:
        return self._source.affine

    @property
    def dtype(self) -> np.dtype:
        return self._source.dtype

    def to_tensor(self) -> Tensor:
        base = self._source.to_tensor()
        i0, i1, j0, j1, k0, k1 = self._padding
        pad_arg = (k0, k1, j0, j1, i0, i1)
        return torch.nn.functional.pad(
            base,
            pad_arg,
            mode=self._padding_mode,
            value=self._fill,
        )

    def __getitem__(self, slices: SliceIndex) -> np.ndarray:
        tensor = self.to_tensor()
        if not isinstance(slices, tuple):
            slices = (slices,)
        return tensor[slices].numpy()


def _get_images(
    subject: Subject,
    include: list[str] | None,
    exclude: list[str] | None,
) -> dict[str, Image]:
    """Filter subject images by include/exclude."""
    images = subject.images
    if include is not None:
        images = {k: v for k, v in images.items() if k in include}
    if exclude is not None:
        images = {k: v for k, v in images.items() if k not in exclude}
    return images


def _crop_image_lazy(image: Image, cropping: TypeSixInts) -> Image:
    """Crop an image lazily — data is only loaded when accessed."""
    i0, i1, j0, j1, k0, k1 = cropping
    c, si, sj, sk = image.shape

    i_slice = slice(i0, si - i1 or None)
    j_slice = slice(j0, sj - j1 or None)
    k_slice = slice(k0, sk - k1 or None)

    # Compute new affine
    affine_matrix = image.affine.data.clone()
    start_voxel = torch.tensor(
        [float(i0), float(j0), float(k0)],
        dtype=torch.float64,
    )
    affine_matrix[:3, 3] += affine_matrix[:3, :3] @ start_voxel
    new_affine = Affine(affine_matrix)

    if image.is_loaded:
        new_data = image.data[:, i_slice, j_slice, k_slice]
        return image.new_like(data=new_data, affine=new_affine)

    # Install a cropped backend on a new Image from the same path
    image._ensure_backend()
    if image._backend is not None and image._path is not None:
        cropped_shape = (
            c,
            len(range(*i_slice.indices(si))),
            len(range(*j_slice.indices(sj))),
            len(range(*k_slice.indices(sk))),
        )
        new = type(image)(
            image._path,
            reader=image._reader,
            reader_kwargs=dict(image._reader_kwargs),
            affine=new_affine,
            **dict(image._metadata),
        )
        new._backend = _CroppedBackend(
            image._backend,
            (i_slice, j_slice, k_slice),
            cropped_shape,
        )
        return new

    # No backend (custom reader) → fall back to eager crop
    new_data = image.data[:, i_slice, j_slice, k_slice]
    return image.new_like(data=new_data, affine=new_affine)


def _pad_image_lazy(
    image: Image,
    padding: TypeSixInts,
    padding_mode: str,
    fill: float,
) -> Image:
    """Pad an image lazily — data is only loaded when accessed."""
    i0, i1, j0, j1, k0, k1 = padding
    c, si, sj, sk = image.shape

    # Compute new affine
    affine_matrix = image.affine.data.clone()
    origin_shift = affine_matrix[:3, :3] @ affine_matrix.new_tensor(
        [-float(i0), -float(j0), -float(k0)],
    )
    affine_matrix[:3, 3] += origin_shift
    new_affine = Affine(affine_matrix)

    padded_shape = (c, si + i0 + i1, sj + j0 + j1, sk + k0 + k1)

    if image.is_loaded:
        pad_arg = (k0, k1, j0, j1, i0, i1)
        new_data = torch.nn.functional.pad(
            image.data,
            pad_arg,
            mode=padding_mode,
            value=fill,
        )
        return image.new_like(data=new_data, affine=new_affine)

    # Install a padded backend on a new Image from the same path
    image._ensure_backend()
    if image._backend is not None and image._path is not None:
        new = type(image)(
            image._path,
            reader=image._reader,
            reader_kwargs=dict(image._reader_kwargs),
            affine=new_affine,
            **dict(image._metadata),
        )
        new._backend = _PaddedBackend(
            image._backend,
            padding,
            padded_shape,
            padding_mode=padding_mode,
            fill=fill,
        )
        return new

    # No backend → fall back to eager pad
    pad_arg = (k0, k1, j0, j1, i0, i1)
    new_data = torch.nn.functional.pad(
        image.data,
        pad_arg,
        mode=padding_mode,
        value=fill,
    )
    return image.new_like(data=new_data, affine=new_affine)


class CropOrPad(SpatialTransform):
    r"""Crop and/or pad to a target spatial shape.

    If the current spatial size along an axis is larger than the target, that
    axis is cropped symmetrically from both sides. If it is smaller, it is
    padded symmetrically. The affine matrix is updated so that physical
    positions of the voxels are maintained.

    The target shape can be specified in voxels (the default), millimetres,
    or centimetres. When physical units are used, the target is converted to
    voxels at transform time using the image spacing.

    When the input is a ``Subject`` or ``Image``, the transform operates
    lazily — data is not loaded from disk until it is actually accessed.

    Args:
        target_shape: Desired spatial shape. A single ``int`` broadcasts
            to all three axes. When ``units`` is ``"mm"`` or ``"cm"``,
            values may be floats representing the physical extent along
            each axis. Use ``None`` for an axis to leave it unchanged,
            e.g., ``(256, 256, None)``.
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
        >>> transform = tio.CropOrPad(target_shape=(256, 256, None))  # keep depth
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

    def forward(self, data):  # type: ignore[override]
        """Apply the transform.

        For ``Subject`` and ``Image`` inputs, operates lazily per-image
        without loading data from disk. For batched inputs, falls back
        to the standard ``SubjectsBatch`` path.
        """
        if isinstance(data, (Subject, Image)):
            return self._forward_lazy(data)
        return super().forward(data)

    def _forward_lazy(self, data: Subject | Image) -> Subject | Image:
        is_image = isinstance(data, Image)
        if is_image:
            subject = Subject(tio_default_image=data)
        else:
            assert isinstance(data, Subject)
            subject = data

        if self.copy:
            subject = _copy.deepcopy(subject)

        if torch.rand(1).item() > self.p:
            return subject.tio_default_image if is_image else subject

        # Read shape and affine from header (no data load)
        first_image = next(iter(subject.images.values()))
        spacing = first_image.affine.spacing
        current_shape: TypeThreeInts = first_image.spatial_shape

        target_voxels = _to_voxels(
            self.target_shape,
            self.units,
            spacing,
            current_shape,
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

        images = _get_images(subject, self.include, self.exclude)

        if padding is not None:
            for name, image in images.items():
                subject._images[name] = _pad_image_lazy(
                    image,
                    padding,
                    self.padding_mode,
                    self.fill,
                )
            pad_params = {
                "padding": padding,
                "padding_mode": self.padding_mode,
                "fill": self.fill,
            }
            subject.applied_transforms.append(
                AppliedTransform(name="Pad", params=pad_params),
            )

        if cropping is not None:
            # Re-read images after padding may have replaced them
            images = _get_images(subject, self.include, self.exclude)
            for name, image in images.items():
                subject._images[name] = _crop_image_lazy(image, cropping)
            crop_params = {"cropping": cropping}
            subject.applied_transforms.append(
                AppliedTransform(name="Crop", params=crop_params),
            )

        # Record CropOrPad (skipped during inversion)
        subject.applied_transforms.append(
            AppliedTransform(
                name="CropOrPad",
                params={"padding": padding, "cropping": cropping},
            ),
        )

        return subject.tio_default_image if is_image else subject

    # --- Standard batch path (for SubjectsBatch, Tensor, etc.) ---

    def make_params(self, batch: SubjectsBatch) -> dict[str, Any]:
        first_images = next(iter(batch.images.values()))
        spacing = first_images.affines[0].spacing

        data_tensor = first_images.data
        current_shape: TypeThreeInts = (
            data_tensor.shape[-3],
            data_tensor.shape[-2],
            data_tensor.shape[-1],
        )

        target_voxels = _to_voxels(
            self.target_shape,
            self.units,
            spacing,
            current_shape,
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
