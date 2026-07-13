from __future__ import annotations

import warnings
from collections.abc import Sequence
from typing import Literal

import numpy as np
import torch

from ....data.subject import Subject
from ...spatial_transform import SpatialTransform
from ...transform import TypeSixBounds
from ...transform import TypeTripletInt
from .crop import Crop
from .pad import Pad

TypeLocation = Literal['center', 'random']
TypeUnits = Literal['voxels', 'mm', 'cm']

TypeTargetShapeAxis = int | float | None
TypeTargetShape = (
    int
    | float
    | tuple[TypeTargetShapeAxis, TypeTargetShapeAxis, TypeTargetShapeAxis]
    | Sequence[TypeTargetShapeAxis]
)


class CropOrPad(SpatialTransform):
    """Modify the field of view by cropping or padding to match a target shape.

    This transform modifies the affine matrix associated to the volume so that
    physical positions of the voxels are maintained.

    Args:
        target_shape: Tuple $(W, H, D)$. If a single value $N$ is
            provided, then $W = H = D = N$. If `None`, the shape will
            be computed from the `mask_name` (and the `labels`, if
            `labels` is not `None`).
        padding_mode: Same as `padding_mode` in
            [`Pad`][torchio.transforms.Pad].
        mask_name: If `None`, the centers of the input and output volumes
            will be the same.
            If a string is given, the output volume center will be the center
            of the bounding box of non-zero values in the image named
            `mask_name`.
        labels: If a label map is used to generate the mask, sequence of labels
            to consider.
        only_crop: If `True`, padding will not be applied, only cropping will
            be done. `only_crop` and `only_pad` cannot both be `True`.
        only_pad: If `True`, cropping will not be applied, only padding will
            be done. `only_crop` and `only_pad` cannot both be `True`.
        location: Where to place the crop window when an axis of the input is
            larger than the target. ``'center'`` (default) splits the cropped
            amount evenly between both sides; ``'random'`` picks a uniformly
            random start position per axis using
            [`torch.randint`][torch.randint], so seeding with
            [`torch.manual_seed`][torch.manual_seed] makes the result
            reproducible. Padding is always centered, regardless of this
            parameter. ``location='random'`` cannot be combined with
            ``mask_name``.
        units: Coordinate system for ``target_shape``. One of ``'voxels'``
            (default), ``'mm'``, or ``'cm'``. When ``'mm'`` or ``'cm'`` is
            used, ``target_shape`` may contain floats representing the
            physical extent along each axis; the target is converted to
            voxels at transform time using the image spacing. When
            ``'voxels'`` is used, all values must be positive integers.
            Use ``None`` for an axis to leave it unchanged
            (e.g., ``target_shape=(256, 256, None)``).
        **kwargs: See [`Transform`][torchio.transforms.Transform] for additional
            keyword arguments.

    Examples:
        >>> import torchio as tio
        >>> subject = tio.Subject(
        ...     chest_ct=tio.ScalarImage('subject_a_ct.nii.gz'),
        ...     heart_mask=tio.LabelMap('subject_a_heart_seg.nii.gz'),
        ... )
        >>> subject.chest_ct.shape
        torch.Size([1, 512, 512, 289])
        >>> transform = tio.CropOrPad(
        ...     (120, 80, 180),
        ...     mask_name='heart_mask',
        ... )
        >>> transformed = transform(subject)
        >>> transformed.chest_ct.shape
        torch.Size([1, 120, 80, 180])
        >>> # Random crop window (useful for data augmentation):
        >>> transform = tio.CropOrPad((96, 96, 96), location='random')
        >>> # Target specified in physical units:
        >>> transform = tio.CropOrPad((150.0, 200.0, 180.0), units='mm')
        >>> # Keep the depth axis unchanged:
        >>> transform = tio.CropOrPad((256, 256, None))

    Warning:
        If `target_shape` is `None`, subjects in the dataset
        will probably have different shapes. This is probably fine if you are
        using [patch-based training ](https://docs.torchio.org/patches/index.html).
        If you are using full volumes for training and a batch size larger than
        one, an error will be raised by the [`DataLoader`][torch.utils.data.DataLoader]
        while trying to collate the batches.

    """

    def __init__(
        self,
        target_shape: TypeTargetShape | None = None,
        padding_mode: str | float = 0,
        mask_name: str | None = None,
        labels: Sequence[int] | None = None,
        only_crop: bool = False,
        only_pad: bool = False,
        location: TypeLocation = 'center',
        units: TypeUnits = 'voxels',
        **kwargs,
    ):
        if target_shape is None and mask_name is None:
            message = 'If mask_name is None, a target shape must be passed'
            raise ValueError(message)
        super().__init__(**kwargs)
        if units not in ('voxels', 'mm', 'cm'):
            message = f"units must be 'voxels', 'mm', or 'cm', got {units!r}"
            raise ValueError(message)
        self.units: TypeUnits = units
        if target_shape is None:
            self.target_shape: (
                tuple[TypeTargetShapeAxis, TypeTargetShapeAxis, TypeTargetShapeAxis]
                | None
            ) = None
        else:
            self.target_shape = self._parse_target_shape(target_shape, units)
        self.padding_mode = padding_mode
        if mask_name is not None and not isinstance(mask_name, str):
            message = (
                f'If mask_name is not None, it must be a string, not {type(mask_name)}'
            )
            raise ValueError(message)
        if location not in ('center', 'random'):
            message = f"location must be 'center' or 'random', got {location!r}"
            raise ValueError(message)
        if location == 'random' and mask_name is not None:
            message = (
                "location='random' cannot be combined with mask_name;"
                ' mask centering and random placement are mutually exclusive'
            )
            raise ValueError(message)
        self.location: TypeLocation = location
        if mask_name is None:
            if labels is not None:
                message = (
                    'If mask_name is None, labels should be None,'
                    f' but "{labels}" was passed'
                )
                raise ValueError(message)
            self._compute_crop_or_pad = self._compute_center_crop_or_pad
        else:
            self._compute_crop_or_pad = self._compute_mask_center_crop_or_pad
        self.mask_name = mask_name
        self.labels = labels

        if only_pad and only_crop:
            message = 'only_crop and only_pad cannot both be True'
            raise ValueError(message)
        self.only_crop = only_crop
        self.only_pad = only_pad
        self.args_names = [
            'target_shape',
            'padding_mode',
            'mask_name',
            'labels',
            'only_crop',
            'only_pad',
            'location',
            'units',
        ]

    @staticmethod
    def _parse_target_shape(
        target_shape: TypeTargetShape,
        units: TypeUnits,
    ) -> tuple[TypeTargetShapeAxis, TypeTargetShapeAxis, TypeTargetShapeAxis]:
        """Normalise ``target_shape`` to a 3-tuple of int|float|None.

        Validates that values are positive and, when ``units == 'voxels'``,
        that they are integers (or ``None``).
        """
        if isinstance(target_shape, str):
            message = (
                'target_shape must be an int, float, or a sequence of 3 '
                f'int/float/None values, got string {target_shape!r}'
            )
            raise TypeError(message)
        if isinstance(target_shape, bool):
            message = (
                'target_shape must be an int, float, or a sequence of 3 '
                f'int/float/None values, got bool {target_shape!r}'
            )
            raise ValueError(message)
        values: list[TypeTargetShapeAxis]
        if isinstance(target_shape, (int, float)):
            values = [target_shape, target_shape, target_shape]
        else:
            try:
                values = list(target_shape)
            except TypeError as e:
                message = (
                    'target_shape must be an int, float, or a sequence of 3 '
                    f'int/float/None values, got {target_shape!r}'
                )
                raise ValueError(message) from e
            if len(values) != 3:
                message = (
                    'target_shape must have 3 elements, got '
                    f'{len(values)}: {target_shape!r}'
                )
                raise ValueError(message)
        result: list[TypeTargetShapeAxis] = []
        for v in values:
            if v is None:
                result.append(None)
                continue
            if isinstance(v, bool) or not isinstance(v, (int, float)):
                message = (
                    'Each target_shape element must be int, float, or None,'
                    f' got {v!r} ({type(v).__name__})'
                )
                raise ValueError(message)
            if v <= 0:
                message = f'target_shape values must be positive, got {v!r}'
                raise ValueError(message)
            if units == 'voxels':
                if isinstance(v, float) and not v.is_integer():
                    message = (
                        'target_shape must contain integers when units='
                        f"'voxels', got {v!r}"
                    )
                    raise ValueError(message)
                result.append(int(v))
            else:
                result.append(float(v))
        return (result[0], result[1], result[2])

    def _resolve_target_voxels(
        self,
        subject: Subject,
    ) -> TypeTripletInt | None:
        """Convert ``self.target_shape`` to integer voxels using subject spacing.

        Returns ``None`` when ``self.target_shape is None`` (which signals the
        mask-bbox path to derive the target from the mask). ``None`` entries
        in ``target_shape`` keep the corresponding source-axis size.
        """
        if self.target_shape is None:
            return None
        spacing = subject.spacing
        source_shape = subject.spatial_shape
        out: list[int] = []
        for value, sp, src in zip(
            self.target_shape, spacing, source_shape, strict=True
        ):
            if value is None:
                out.append(int(src))
            elif self.units == 'voxels':
                out.append(int(value))
            else:
                factor = 10.0 if self.units == 'cm' else 1.0
                out.append(int(round(float(value) * factor / float(sp))))
        for n in out:
            if n < 1:
                message = (
                    'Resolved target shape must be positive in all axes,'
                    f' got {tuple(out)} from target_shape={self.target_shape!r}'
                    f' with units={self.units!r} and spacing={spacing!r}'
                )
                raise ValueError(message)
        return (out[0], out[1], out[2])

    @staticmethod
    def _bbox_mask(mask_volume: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Return 6 coordinates of a 3D bounding box from a given mask.

        Taken from [this SO question ](https://stackoverflow.com/questions/31400769/bounding-box-of-numpy-array).

        Args:
            mask_volume: 3D NumPy array.
        """
        i_any = np.any(mask_volume, axis=(1, 2))
        j_any = np.any(mask_volume, axis=(0, 2))
        k_any = np.any(mask_volume, axis=(0, 1))
        i_min, i_max = np.where(i_any)[0][[0, -1]]
        j_min, j_max = np.where(j_any)[0][[0, -1]]
        k_min, k_max = np.where(k_any)[0][[0, -1]]
        bb_min = np.array([i_min, j_min, k_min])
        bb_max = np.array([i_max, j_max, k_max]) + 1
        return bb_min, bb_max

    @staticmethod
    def _get_six_bounds_parameters(
        parameters: np.ndarray,
    ) -> TypeSixBounds:
        r"""Compute bounds parameters for ITK filters.

        Args:
            parameters: Tuple $(w, h, d)$ with the number of voxels to be
                cropped or padded.

        Returns:
            Tuple $(w_{ini}, w_{fin}, h_{ini}, h_{fin}, d_{ini}, d_{fin})$,
            where $n_{ini} = \left \lceil \frac{n}{2} \right \rceil$ and
            $n_{fin} = \left \lfloor \frac{n}{2} \right \rfloor$.

        Examples:
            >>> p = np.array((4, 0, 7))
            >>> CropOrPad._get_six_bounds_parameters(p)
            (2, 2, 0, 0, 4, 3)
        """
        parameters = parameters / 2
        result = []
        for number in parameters:
            ini, fin = int(np.ceil(number)), int(np.floor(number))
            result.extend([ini, fin])
        i1, i2, j1, j2, k1, k2 = result
        return i1, i2, j1, j2, k1, k2

    def _compute_cropping_padding_from_shapes(
        self,
        source_shape: TypeTripletInt,
        target_shape: TypeTripletInt,
    ) -> tuple[TypeSixBounds | None, TypeSixBounds | None]:
        diff_shape = np.array(target_shape) - source_shape

        cropping = -np.minimum(diff_shape, 0)
        if cropping.any():
            if self.location == 'random':
                cropping_params = self._get_random_six_bounds_parameters(
                    cropping,
                )
            else:
                cropping_params = self._get_six_bounds_parameters(cropping)
        else:
            cropping_params = None

        padding = np.maximum(diff_shape, 0)
        if padding.any():
            padding_params = self._get_six_bounds_parameters(padding)
        else:
            padding_params = None

        return padding_params, cropping_params

    @staticmethod
    def _get_random_six_bounds_parameters(
        parameters: np.ndarray,
    ) -> TypeSixBounds:
        """Compute asymmetric per-axis bounds for random cropping.

        For each axis with a non-zero amount ``n`` to remove, sample
        ``ini`` uniformly from ``{0, 1, ..., n}`` using
        [`torch.randint`][torch.randint] and use ``fin = n - ini`` so
        the total amount removed matches ``n``. Axes with ``n == 0``
        contribute ``(0, 0)``.
        """
        result: list[int] = []
        for amount in parameters:
            amount_int = int(amount)
            if amount_int == 0:
                result.extend([0, 0])
            else:
                ini = int(torch.randint(0, amount_int + 1, (1,)).item())
                result.extend([ini, amount_int - ini])
        i1, i2, j1, j2, k1, k2 = result
        return i1, i2, j1, j2, k1, k2

    def _compute_center_crop_or_pad(
        self,
        subject: Subject,
        target_shape: TypeTripletInt | None,
    ) -> tuple[TypeSixBounds | None, TypeSixBounds | None]:
        assert target_shape is not None, (
            'target_shape must be resolved before center crop/pad'
        )
        source_shape = subject.spatial_shape
        parameters = self._compute_cropping_padding_from_shapes(
            source_shape, target_shape
        )
        padding_params, cropping_params = parameters
        return padding_params, cropping_params

    def _compute_mask_center_crop_or_pad(
        self,
        subject: Subject,
        target_shape: TypeTripletInt | None,
    ) -> tuple[TypeSixBounds | None, TypeSixBounds | None]:
        assert self.mask_name is not None
        if self.mask_name not in subject:
            message = (
                f'Mask name "{self.mask_name}"'
                f' not found in subject keys "{tuple(subject.keys())}".'
                ' Using volume center instead'
            )
            warnings.warn(message, RuntimeWarning, stacklevel=2)
            # Fall back: if target_shape is None too, we cannot proceed; but
            # validation in __init__ guarantees mask_name is set together with
            # an optional target_shape. When target_shape is None we use the
            # subject's own spatial shape (no-op).
            fallback_target = (
                target_shape if target_shape is not None else subject.spatial_shape
            )
            return self._compute_center_crop_or_pad(
                subject=subject,
                target_shape=fallback_target,
            )

        mask_image = subject.get_image(self.mask_name)
        mask_data = self.get_mask_from_masking_method(
            self.mask_name,
            subject,
            mask_image.data,
            self.labels,
        ).numpy()

        if not np.any(mask_data):
            message = (
                f'All values found in the mask "{self.mask_name}"'
                ' are zero. Using volume center instead'
            )
            warnings.warn(message, RuntimeWarning, stacklevel=2)
            fallback_target = (
                target_shape if target_shape is not None else subject.spatial_shape
            )
            return self._compute_center_crop_or_pad(
                subject=subject,
                target_shape=fallback_target,
            )

        # Let's assume that the center of first voxel is at coordinate 0.5
        # (which is typically not the case)
        subject_shape = subject.spatial_shape
        bb_min, bb_max = self._bbox_mask(mask_data[0])
        center_mask = np.mean((bb_min, bb_max), axis=0)
        padding = []
        cropping = []

        if target_shape is None:
            resolved_target = bb_max - bb_min
        else:
            resolved_target = target_shape

        for dim in range(3):
            target_dim = resolved_target[dim]
            center_dim = center_mask[dim]
            subject_dim = subject_shape[dim]

            center_on_index = not (center_dim % 1)
            target_even = not (target_dim % 2)

            # Approximation when the center cannot be computed exactly
            # The output will be off by half a voxel, but this is just an
            # implementation detail
            if target_even ^ center_on_index:
                center_dim -= 0.5

            begin = center_dim - target_dim / 2
            if begin >= 0:
                crop_ini = begin
                pad_ini = 0
            else:
                crop_ini = 0
                pad_ini = -begin

            end = center_dim + target_dim / 2
            if end <= subject_dim:
                crop_fin = subject_dim - end
                pad_fin = 0
            else:
                crop_fin = 0
                pad_fin = end - subject_dim

            padding.extend([pad_ini, pad_fin])
            cropping.extend([crop_ini, crop_fin])
        # Conversion for SimpleITK compatibility
        padding_array = np.asarray(padding, dtype=int)
        cropping_array = np.asarray(cropping, dtype=int)
        if padding_array.any():
            padding_values = [int(value) for value in padding_array.tolist()]
            padding_params = (
                padding_values[0],
                padding_values[1],
                padding_values[2],
                padding_values[3],
                padding_values[4],
                padding_values[5],
            )
        else:
            padding_params = None
        if cropping_array.any():
            cropping_values = [int(value) for value in cropping_array.tolist()]
            cropping_params = (
                cropping_values[0],
                cropping_values[1],
                cropping_values[2],
                cropping_values[3],
                cropping_values[4],
                cropping_values[5],
            )
        else:
            cropping_params = None
        return padding_params, cropping_params

    def apply_transform(self, subject: Subject) -> Subject:
        subject.check_consistent_space()
        target_shape = self._resolve_target_voxels(subject)
        padding_params, cropping_params = self._compute_crop_or_pad(
            subject, target_shape
        )
        if padding_params is not None and not self.only_crop:
            pad = Pad(
                padding_params,
                padding_mode=self.padding_mode,
                copy=self.copy,
                include=self.include,
                exclude=self.exclude,
                keep=self.keep,
                parse_input=self.parse_input,
                label_keys=self.label_keys,
            )
            transformed = pad(subject)
            assert isinstance(transformed, Subject)
            subject = transformed
        if cropping_params is not None and not self.only_pad:
            crop = Crop(
                cropping_params,
                copy=self.copy,
                include=self.include,
                exclude=self.exclude,
                keep=self.keep,
                parse_input=self.parse_input,
                label_keys=self.label_keys,
            )
            transformed = crop(subject)
            assert isinstance(transformed, Subject)
            subject = transformed
        return subject
