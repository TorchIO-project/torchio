from __future__ import annotations

import warnings
from collections.abc import Sequence

import numpy as np

from ....data.subject import Subject
from ....utils import parse_spatial_shape
from ...spatial_transform import SpatialTransform
from ...transform import TypeSixBounds
from ...transform import TypeTripletInt
from .crop import Crop
from .pad import Pad


class CropOrPad(SpatialTransform):
    """Modify the field of view by cropping or padding to match a target shape.

    This transform modifies the affine matrix associated to the volume so that
    physical positions of the voxels are maintained.

    Args:
        target_shape: Tuple :math:`(W, H, D)`. If a single value :math:`N` is
            provided, then :math:`W = H = D = N`. If ``None``, the shape will
            be computed from the :attr:`mask_name` (and the :attr:`labels`, if
            :attr:`labels` is not ``None``).
        padding_mode: Same as :attr:`padding_mode` in
            :class:`~torchio.transforms.Pad`.
        mask_name: If ``None``, the centers of the input and output volumes
            will be the same.
            If a string is given, the output volume center will be the center
            of the bounding box of non-zero values in the image named
            :attr:`mask_name`.
        labels: If a label map is used to generate the mask, sequence of labels
            to consider.
        only_crop: If ``True``, padding will not be applied, only cropping will
            be done. ``only_crop`` and ``only_pad`` cannot both be ``True``.
        only_pad: If ``True``, cropping will not be applied, only padding will
            be done. ``only_crop`` and ``only_pad`` cannot both be ``True``.
        **kwargs: See :class:`~torchio.transforms.Transform` for additional
            keyword arguments.

    Example:
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

    .. warning:: If :attr:`target_shape` is ``None``, subjects in the dataset
        will probably have different shapes. This is probably fine if you are
        using `patch-based training <https://docs.torchio.org/patches/index.html>`_.
        If you are using full volumes for training and a batch size larger than
        one, an error will be raised by the :class:`~torch.utils.data.DataLoader`
        while trying to collate the batches.

    .. plot::

        import torchio as tio
        t1 = tio.datasets.Colin27().t1
        crop_pad = tio.CropOrPad((512, 512, 32))
        t1_pad_crop = crop_pad(t1)
        subject = tio.Subject(t1=t1, crop_pad=t1_pad_crop)
        subject.plot()
    """

    def __init__(
        self,
        target_shape: int | TypeTripletInt | None = None,
        padding_mode: str | float = 0,
        mask_name: str | None = None,
        labels: Sequence[int] | None = None,
        only_crop: bool = False,
        only_pad: bool = False,
        **kwargs,
    ):
        if target_shape is None and mask_name is None:
            message = 'If mask_name is None, a target shape must be passed'
            raise ValueError(message)
        super().__init__(**kwargs)
        if target_shape is None:
            self.target_shape = None
        else:
            self.target_shape = parse_spatial_shape(target_shape)
        self.padding_mode = padding_mode
        if mask_name is not None and not isinstance(mask_name, str):
            message = (
                f'If mask_name is not None, it must be a string, not {type(mask_name)}'
            )
            raise ValueError(message)
        if mask_name is None:
            if labels is not None:
                message = (
                    'If mask_name is None, labels should be None,'
                    f' but "{labels}" was passed'
                )
                raise ValueError(message)
            self.compute_crop_or_pad = self._compute_center_crop_or_pad
        else:
            if not isinstance(mask_name, str):
                message = (
                    'If mask_name is not None, it must be a string,'
                    f' not {type(mask_name)}'
                )
                raise ValueError(message)
            self.compute_crop_or_pad = self._compute_mask_center_crop_or_pad
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
        ]

    @staticmethod
    def _bbox_mask(mask_volume: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Return 6 coordinates of a 3D bounding box from a given mask.

        Taken from `this SO question <https://stackoverflow.com/questions/31400769/bounding-box-of-numpy-array>`_.

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
            parameters: Tuple :math:`(w, h, d)` with the number of voxels to be
                cropped or padded.

        Returns:
            Tuple :math:`(w_{ini}, w_{fin}, h_{ini}, h_{fin}, d_{ini}, d_{fin})`,
            where :math:`n_{ini} = \left \lceil \frac{n}{2} \right \rceil` and
            :math:`n_{fin} = \left \lfloor \frac{n}{2} \right \rfloor`.

        Example:
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
    ) -> tuple[TypeSixBounds | None, TypeSixBounds | None]:
        diff_shape = np.array(self.target_shape) - source_shape

        cropping = -np.minimum(diff_shape, 0)
        if cropping.any():
            cropping_params = self._get_six_bounds_parameters(cropping)
        else:
            cropping_params = None

        padding = np.maximum(diff_shape, 0)
        if padding.any():
            padding_params = self._get_six_bounds_parameters(padding)
        else:
            padding_params = None

        return padding_params, cropping_params

    def _compute_center_crop_or_pad(
        self,
        subject: Subject,
    ) -> tuple[TypeSixBounds | None, TypeSixBounds | None]:
        source_shape = subject.spatial_shape
        parameters = self._compute_cropping_padding_from_shapes(source_shape)
        padding_params, cropping_params = parameters
        return padding_params, cropping_params

    def _compute_mask_center_crop_or_pad(
        self,
        subject: Subject,
    ) -> tuple[TypeSixBounds | None, TypeSixBounds | None]:
        if self.mask_name not in subject:
            message = (
                f'Mask name "{self.mask_name}"'
                f' not found in subject keys "{tuple(subject.keys())}".'
                ' Using volume center instead'
            )
            warnings.warn(message, RuntimeWarning, stacklevel=2)
            return self._compute_center_crop_or_pad(subject=subject)

        mask_data = self.get_mask_from_masking_method(
            self.mask_name,
            subject,
            subject[self.mask_name].data,
            self.labels,
        ).numpy()

        if not np.any(mask_data):
            message = (
                f'All values found in the mask "{self.mask_name}"'
                ' are zero. Using volume center instead'
            )
            warnings.warn(message, RuntimeWarning, stacklevel=2)
            return self._compute_center_crop_or_pad(subject=subject)

        # Let's assume that the center of first voxel is at coordinate 0.5
        # (which is typically not the case)
        subject_shape = subject.spatial_shape
        bb_min, bb_max = self._bbox_mask(mask_data[0])
        center_mask = np.mean((bb_min, bb_max), axis=0)
        padding = []
        cropping = []

        if self.target_shape is None:
            target_shape = bb_max - bb_min
        else:
            target_shape = self.target_shape

        for dim in range(3):
            target_dim = target_shape[dim]
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
            padding_params = tuple(padding_array.tolist())
        else:
            padding_params = None
        if cropping_array.any():
            cropping_params = tuple(cropping_array.tolist())
        else:
            cropping_params = None
        return padding_params, cropping_params  # type: ignore[return-value]

    def apply_transform(self, subject: Subject) -> Subject:
        subject.check_consistent_space()
        padding_params, cropping_params = self.compute_crop_or_pad(subject)
        padding_kwargs = {'padding_mode': self.padding_mode}
        if padding_params is not None and not self.only_crop:
            pad = Pad(padding_params, **self.get_base_args(), **padding_kwargs)
            subject = pad(subject)  # type: ignore[assignment]
        if cropping_params is not None and not self.only_pad:
            crop = Crop(cropping_params, **self.get_base_args())
            subject = crop(subject)  # type: ignore[assignment]
        return subject
