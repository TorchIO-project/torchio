import warnings

import numpy as np

from ....data.subject import Subject
from ....types import TypeSpatialShape
from ....utils import to_tuple
from ...spatial_transform import SpatialTransform
from .crop_or_pad import CropOrPad
from .resample import Resample


class Resize(SpatialTransform):
    """Resample images so the output shape matches the given target shape.

    The field of view remains the same.

    .. warning:: In most medical image applications, this transform should not
        be used as it will deform the physical object by scaling anisotropically
        along the different dimensions. The solution to change an image size is
        typically applying :class:`~torchio.transforms.Resample` and
        :class:`~torchio.transforms.CropOrPad`.

    Args:
        target_shape: Tuple :math:`(W, H, D)`. If a single value :math:`N` is
            provided, then :math:`W = H = D = N`. The size of dimensions set to
            -1 will be kept.
        image_interpolation: See :ref:`Interpolation`.
        label_interpolation: See :ref:`Interpolation`.
    """

    def __init__(
        self,
        target_shape: TypeSpatialShape,
        image_interpolation: str = 'linear',
        label_interpolation: str = 'nearest',
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.target_shape = np.asarray(to_tuple(target_shape, length=3))
        self.image_interpolation = self.parse_interpolation(
            image_interpolation,
        )
        self.label_interpolation = self.parse_interpolation(
            label_interpolation,
        )
        self.args_names = [
            'target_shape',
            'image_interpolation',
            'label_interpolation',
        ]

    def apply_transform(self, subject: Subject) -> Subject:
        shape_in = np.asarray(subject.spatial_shape)
        shape_out = self.target_shape
        negative_mask = shape_out == -1
        shape_out[negative_mask] = shape_in[negative_mask]
        spacing_in = np.asarray(subject.spacing)
        spacing_out = shape_in / shape_out * spacing_in
        resample = Resample(
            spacing_out,
            image_interpolation=self.image_interpolation,
            label_interpolation=self.label_interpolation,
            **self.get_base_args(),
        )
        resampled = resample(subject)
        assert isinstance(resampled, Subject)
        # Sometimes, the output shape is one voxel too large
        if not resampled.spatial_shape == tuple(shape_out):
            message = (
                f'Output shape {resampled.spatial_shape}'
                f' != target shape {tuple(shape_out)}. Fixing with CropOrPad'
            )
            warnings.warn(message, RuntimeWarning, stacklevel=2)
            crop_pad = CropOrPad(shape_out, **self.get_base_args())  # type: ignore[arg-type]
            resampled = crop_pad(resampled)
        assert isinstance(resampled, Subject)
        return resampled
