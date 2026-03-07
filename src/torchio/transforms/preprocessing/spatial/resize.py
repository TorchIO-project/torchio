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

    Warning:
        In most medical image applications, this transform should not
        be used as it will deform the physical object by scaling anisotropically
        along the different dimensions. The solution to change an image size is
        typically applying [`Resample`][torchio.transforms.Resample] and
        [`CropOrPad`][torchio.transforms.CropOrPad].

    Args:
        target_shape: Tuple $(W, H, D)$. If a single value $N$ is
            provided, then $W = H = D = N$. The size of dimensions set to
            -1 will be kept.
        image_interpolation: See Interpolation.
        label_interpolation: See Interpolation.
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
            copy=self.copy,
            include=self.include,
            exclude=self.exclude,
            keep=self.keep,
            parse_input=self.parse_input,
            label_keys=self.label_keys,
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
            target_shape_values = [int(value) for value in shape_out.tolist()]
            crop_pad = CropOrPad(
                (
                    target_shape_values[0],
                    target_shape_values[1],
                    target_shape_values[2],
                ),
                copy=self.copy,
                include=self.include,
                exclude=self.exclude,
                keep=self.keep,
                parse_input=self.parse_input,
                label_keys=self.label_keys,
            )
            resampled = crop_pad(resampled)
        assert isinstance(resampled, Subject)
        return resampled
