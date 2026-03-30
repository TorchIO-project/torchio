import warnings

import torch

from ....data.subject import Subject
from ....types import TypeRangeFloat
from ....utils import to_tuple
from ...preprocessing import Resample
from .. import RandomTransform


class RandomAnisotropy(RandomTransform):
    r"""Downsample an image along an axis and upsample to initial space.

    This transform simulates an image that has been acquired using anisotropic
    spacing and resampled back to its original spacing.

    Similar to the work by Billot et al.: [Partial Volume Segmentation of Brain
    MRI Scans of any Resolution and
    Contrast ](https://link.springer.com/chapter/10.1007/978-3-030-59728-3_18).

    Args:
        axes: Axis or tuple of axes along which the image will be downsampled.
        downsampling: Downsampling factor $m \gt 1$. If a tuple
            $(a, b)$ is provided then $m \sim \mathcal{U}(a, b)$.
        image_interpolation: Image interpolation used to upsample the image
            back to its initial spacing. Downsampling is performed using
            nearest neighbor interpolation. See Interpolation for
            supported interpolation types.
        scalars_only: Apply only to instances of [`torchio.ScalarImage`][torchio.ScalarImage].
            This is useful when the segmentation quality needs to be kept,
            as in [Billot et al. ](https://link.springer.com/chapter/10.1007/978-3-030-59728-3_18).
        **kwargs: See [`Transform`][torchio.transforms.Transform] for additional
            keyword arguments.

    Examples:
        >>> import torchio as tio
        >>> transform = tio.RandomAnisotropy(axes=1, downsampling=2)
        >>> transform = tio.RandomAnisotropy(
        ...     axes=(0, 1, 2),
        ...     downsampling=(2, 5),
        ... )   # Multiply spacing of one of the 3 axes by a factor randomly chosen in [2, 5]
        >>> colin = tio.datasets.Colin27()
        >>> transformed = transform(colin)
    """

    def __init__(
        self,
        axes: int | tuple[int, ...] = (0, 1, 2),
        downsampling: TypeRangeFloat = (1.5, 5),
        image_interpolation: str = 'linear',
        scalars_only: bool = True,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.axes = self.parse_axes(axes)
        self.downsampling_range = self._parse_range(
            downsampling,
            'downsampling',
            min_constraint=1,
        )
        parsed_interpolation = self.parse_interpolation(image_interpolation)
        self.image_interpolation = parsed_interpolation
        self.scalars_only = scalars_only

    def get_params(
        self,
        axes: tuple[int, ...],
        downsampling_range: tuple[float, float],
    ) -> tuple[int, float]:
        axis = axes[torch.randint(0, len(axes), (1,))]
        downsampling = self.sample_uniform(*downsampling_range)
        return axis, downsampling

    @staticmethod
    def parse_axes(axes: int | tuple[int, ...]):
        axes_tuple = to_tuple(axes)
        for axis in axes_tuple:
            is_int = isinstance(axis, int)
            if not is_int or axis not in (0, 1, 2):
                raise ValueError('All axes must be 0, 1 or 2')
        return axes_tuple

    def apply_transform(self, subject: Subject) -> Subject:
        is_2d = subject.get_first_image().is_2d()
        axes = self.axes
        if is_2d and 2 in self.axes:
            warnings.warn(
                f'Input image is 2D, but "2" is in axes: {self.axes}',
                RuntimeWarning,
                stacklevel=2,
            )
            axes = tuple(axis for axis in self.axes if axis != 2)
        axis, downsampling = self.get_params(
            axes,
            self.downsampling_range,
        )
        target_spacing = list(subject.spacing)
        target_spacing[axis] *= downsampling

        # NOTE: If copy=False, the underlying image data will be modified in place.
        # We have to obtain the target spatial shape and affine before the transform
        image = subject.get_first_image()
        original_shape = image.spatial_shape
        original_affine = image.affine.copy()
        downsample = Resample(
            target=(
                float(target_spacing[0]),
                float(target_spacing[1]),
                float(target_spacing[2]),
            ),
            image_interpolation='nearest',
            scalars_only=self.scalars_only,
            copy=self.copy,
            include=self.include,
            exclude=self.exclude,
            keep=self.keep,
            parse_input=self.parse_input,
            label_keys=self.label_keys,
        )
        downsampled = downsample(subject)
        upsample = Resample(
            target=(original_shape, original_affine),
            image_interpolation=self.image_interpolation,
            scalars_only=self.scalars_only,
            copy=self.copy,
            include=self.include,
            exclude=self.exclude,
            keep=self.keep,
            parse_input=self.parse_input,
            label_keys=self.label_keys,
        )
        upsampled = upsample(downsampled)
        assert isinstance(upsampled, Subject)
        return upsampled
