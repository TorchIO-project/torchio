from __future__ import annotations

from collections.abc import Sequence
from numbers import Number
from typing import Protocol
from typing import cast

import numpy as np
import SimpleITK as sitk
import torch

from ....constants import INTENSITY
from ....constants import TYPE
from ....data.io import nib_to_sitk
from ....data.subject import Subject
from ....types import TypeRangeFloat
from ....types import TypeSextetFloat
from ....types import TypeTripletFloat
from ....utils import get_major_sitk_version
from ....utils import to_tuple
from ...spatial_transform import SpatialTransform
from .. import RandomTransform

TypeOneToSixFloat = TypeRangeFloat | TypeTripletFloat | TypeSextetFloat


class _CompositeTransform(Protocol):
    def AddTransform(self, transform: sitk.Transform) -> None: ...


class RandomAffine(RandomTransform, SpatialTransform):
    r"""Apply a random affine transformation and resample the image.

    Args:
        scales: Tuple $(a_1, b_1, a_2, b_2, a_3, b_3)$ defining the
            scaling ranges.
            The scaling values along each dimension are $(s_1, s_2, s_3)$,
            where $s_i \sim \mathcal{U}(a_i, b_i)$.
            If two values $(a, b)$ are provided,
            then $s_i \sim \mathcal{U}(a, b)$.
            If only one value $x$ is provided,
            then $s_i \sim \mathcal{U}(1 - x, 1 + x)$.
            If three values $(x_1, x_2, x_3)$ are provided,
            then $s_i \sim \mathcal{U}(1 - x_i, 1 + x_i)$.
            For example, using `scales=(0.5, 0.5)` will zoom out the image,
            making the objects inside look twice as small while preserving
            the physical size and position of the image bounds.
        degrees: Tuple $(a_1, b_1, a_2, b_2, a_3, b_3)$ defining the
            rotation ranges in degrees.
            Rotation angles around each axis are
            $(\theta_1, \theta_2, \theta_3)$,
            where $\theta_i \sim \mathcal{U}(a_i, b_i)$.
            If two values $(a, b)$ are provided,
            then $\theta_i \sim \mathcal{U}(a, b)$.
            If only one value $x$ is provided,
            then $\theta_i \sim \mathcal{U}(-x, x)$.
            If three values $(x_1, x_2, x_3)$ are provided,
            then $\theta_i \sim \mathcal{U}(-x_i, x_i)$.
        translation: Tuple $(a_1, b_1, a_2, b_2, a_3, b_3)$ defining the
            translation ranges in mm.
            Translation along each axis is $(t_1, t_2, t_3)$,
            where $t_i \sim \mathcal{U}(a_i, b_i)$.
            If two values $(a, b)$ are provided,
            then $t_i \sim \mathcal{U}(a, b)$.
            If only one value $x$ is provided,
            then $t_i \sim \mathcal{U}(-x, x)$.
            If three values $(x_1, x_2, x_3)$ are provided,
            then $t_i \sim \mathcal{U}(-x_i, x_i)$.
            For example, if the image is in RAS+ orientation (e.g., after
            applying [`ToCanonical`][torchio.transforms.preprocessing.ToCanonical])
            and the translation is $(10, 20, 30)$, the sample will move
            10 mm to the right, 20 mm to the front, and 30 mm upwards.
            If the image was in, e.g., PIR+ orientation, the sample will move
            10 mm to the back, 20 mm downwards, and 30 mm to the right.
        isotropic: If `True`, only one scaling factor will be sampled for all dimensions,
            i.e. $s_1 = s_2 = s_3$.
            If one value $x$ is provided in `scales`, the scaling factor along all
            dimensions will be $s \sim \mathcal{U}(1 - x, 1 + x)$.
            If two values provided $(a, b)$ in `scales`, the scaling factor along all
            dimensions will be $s \sim \mathcal{U}(a, b)$.
        center: If `'image'`, rotations and scaling will be performed around
            the image center. If `'origin'`, rotations and scaling will be
            performed around the origin in world coordinates.
        default_pad_value: As the image is rotated, some values near the
            borders will be undefined.
            If `'minimum'`, the fill value will be the image minimum.
            If `'mean'`, the fill value is the mean of the border values.
            If `'otsu'`, the fill value is the mean of the values at the
            border that lie under an
            [Otsu threshold ](https://ieeexplore.ieee.org/document/4310076).
            If it is a number, that value will be used.
            This parameter applies to intensity images only.
        default_pad_label: As the label map is rotated, some values near the
            borders will be undefined. This numeric value will be used to fill
            those undefined regions. This parameter applies to label maps only.
        image_interpolation: See Interpolation.
        label_interpolation: See Interpolation.
        check_shape: If `True` an error will be raised if the images are in
            different physical spaces. If `False`, `center` should
            probably not be `'image'` but `'center'`.
        **kwargs: See [`Transform`][torchio.transforms.Transform] for additional
            keyword arguments.

    Examples:
        >>> import torchio as tio
        >>> image = tio.datasets.Colin27().t1
        >>> transform = tio.RandomAffine(
        ...     scales=(0.9, 1.2),
        ...     degrees=15,
        ... )
        >>> transformed = transform(image)

    """

    def __init__(
        self,
        scales: TypeOneToSixFloat = 0.1,
        degrees: TypeOneToSixFloat = 10,
        translation: TypeOneToSixFloat = 0,
        isotropic: bool = False,
        center: str = 'image',
        default_pad_value: str | float = 'minimum',
        default_pad_label: int | float = 0,
        image_interpolation: str = 'linear',
        label_interpolation: str = 'nearest',
        check_shape: bool = True,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.isotropic = isotropic
        _parse_scales_isotropic(scales, isotropic)
        self.scales = self.parse_params(scales, 1, 'scales', min_constraint=0)
        self.degrees = self.parse_params(degrees, 0, 'degrees')
        self.translation = self.parse_params(translation, 0, 'translation')
        if center not in ('image', 'origin'):
            message = f'Center argument must be "image" or "origin", not "{center}"'
            raise ValueError(message)
        self.center = center
        self.default_pad_value = _parse_default_value(default_pad_value)
        if not isinstance(default_pad_label, (int, float)):
            message = 'default_pad_label must be a number, '
            message += f'but it is "{default_pad_label}"'
            raise ValueError(message)
        self.default_pad_label = float(default_pad_label)
        self.image_interpolation = self.parse_interpolation(
            image_interpolation,
        )
        self.label_interpolation = self.parse_interpolation(
            label_interpolation,
        )
        self.check_shape = check_shape

    @staticmethod
    def get_params(
        scales: TypeSextetFloat,
        degrees: TypeSextetFloat,
        translation: TypeSextetFloat,
        isotropic: bool,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        scaling_params = torch.as_tensor(
            RandomTransform.sample_uniform_sextet(scales),
            dtype=torch.float64,
        )
        if isotropic:
            scaling_params.fill_(scaling_params[0])
        rotation_params = torch.as_tensor(
            RandomTransform.sample_uniform_sextet(degrees),
            dtype=torch.float64,
        )
        translation_params = torch.as_tensor(
            RandomTransform.sample_uniform_sextet(translation),
            dtype=torch.float64,
        )
        return scaling_params, rotation_params, translation_params

    def apply_transform(self, subject: Subject) -> Subject:
        scaling_params, rotation_params, translation_params = self.get_params(
            self.scales,
            self.degrees,
            self.translation,
            self.isotropic,
        )
        scaling_values = [float(value) for value in scaling_params.tolist()]
        rotation_values = [float(value) for value in rotation_params.tolist()]
        translation_values = [float(value) for value in translation_params.tolist()]
        transform = Affine(
            scales=(scaling_values[0], scaling_values[1], scaling_values[2]),
            degrees=(rotation_values[0], rotation_values[1], rotation_values[2]),
            translation=(
                translation_values[0],
                translation_values[1],
                translation_values[2],
            ),
            center=self.center,
            default_pad_value=self.default_pad_value,
            default_pad_label=self.default_pad_label,
            image_interpolation=self.image_interpolation,
            label_interpolation=self.label_interpolation,
            check_shape=self.check_shape,
            **self.get_base_args(),
        )
        transformed = transform(subject)
        assert isinstance(transformed, Subject)
        return transformed


class Affine(SpatialTransform):
    r"""Apply affine transformation.

    Args:
        scales: Tuple $(s_1, s_2, s_3)$ defining the
            scaling values along each dimension.
        degrees: Tuple $(\theta_1, \theta_2, \theta_3)$ defining the
            rotation around each axis.
        translation: Tuple $(t_1, t_2, t_3)$ defining the
            translation in mm along each axis.
        center: If `'image'`, rotations and scaling will be performed around
            the image center. If `'origin'`, rotations and scaling will be
            performed around the origin in world coordinates.
        default_pad_value: As the image is rotated, some values near the
            borders will be undefined.
            If `'minimum'`, the fill value will be the image minimum.
            If `'mean'`, the fill value is the mean of the border values.
            If `'otsu'`, the fill value is the mean of the values at the
            border that lie under an
            [Otsu threshold ](https://ieeexplore.ieee.org/document/4310076).
            If it is a number, that value will be used.
            This parameter applies to intensity images only.
        default_pad_label: As the label map is rotated, some values near the
            borders will be undefined. This numeric value will be used to fill
            those undefined regions. This parameter applies to label maps only.
        image_interpolation: See Interpolation.
        label_interpolation: See Interpolation.
        check_shape: If `True` an error will be raised if the images are in
            different physical spaces. If `False`, `center` should
            probably not be `'image'` but `'center'`.
        **kwargs: See [`Transform`][torchio.transforms.Transform] for additional
            keyword arguments.
    """

    def __init__(
        self,
        scales: TypeTripletFloat,
        degrees: TypeTripletFloat,
        translation: TypeTripletFloat,
        center: str = 'image',
        default_pad_value: str | float = 'minimum',
        default_pad_label: int | float = 0,
        image_interpolation: str = 'linear',
        label_interpolation: str = 'nearest',
        check_shape: bool = True,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.scales = self.parse_params(
            scales,
            None,
            'scales',
            make_ranges=False,
            min_constraint=0,
        )
        self.degrees = self.parse_params(
            degrees,
            None,
            'degrees',
            make_ranges=False,
        )
        self.translation = self.parse_params(
            translation,
            None,
            'translation',
            make_ranges=False,
        )
        if center not in ('image', 'origin'):
            message = f'Center argument must be "image" or "origin", not "{center}"'
            raise ValueError(message)
        self.center = center
        self.use_image_center = center == 'image'
        self.default_pad_value = _parse_default_value(default_pad_value)
        if not isinstance(default_pad_label, (int, float)):
            message = 'default_pad_label must be a number, '
            message += f'but it is "{default_pad_label}"'
            raise ValueError(message)
        self.default_pad_label = float(default_pad_label)
        self.image_interpolation = self.parse_interpolation(
            image_interpolation,
        )
        self.label_interpolation = self.parse_interpolation(
            label_interpolation,
        )
        self.invert_transform = False
        self.check_shape = check_shape
        self.args_names = [
            'scales',
            'degrees',
            'translation',
            'center',
            'default_pad_value',
            'default_pad_label',
            'image_interpolation',
            'label_interpolation',
            'check_shape',
        ]

    @staticmethod
    def _get_scaling_transform(
        scaling_params: Sequence[float] | np.ndarray,
        center_lps: TypeTripletFloat | None = None,
    ) -> sitk.ScaleTransform:
        # 1.5 means the objects look 1.5 times larger
        transform = sitk.ScaleTransform(3)
        scaling_params_array = np.array(scaling_params).astype(float)
        transform.SetScale(scaling_params_array)
        if center_lps is not None:
            transform.SetCenter(center_lps)
        return transform

    @staticmethod
    def _get_rotation_transform(
        degrees: Sequence[float] | np.ndarray,
        translation: Sequence[float] | np.ndarray,
        center_lps: TypeTripletFloat | None = None,
    ) -> sitk.Euler3DTransform:
        def ras_to_lps(triplet: Sequence[float] | np.ndarray) -> np.ndarray:
            return np.array((-1, -1, 1), dtype=float) * np.asarray(triplet)

        transform = sitk.Euler3DTransform()
        radians = np.asarray(np.radians(degrees), dtype=float)

        # SimpleITK uses LPS
        radians_lps = ras_to_lps(radians)
        translation_lps = ras_to_lps(translation)

        transform.SetRotation(*radians_lps)
        transform.SetTranslation(translation_lps)
        if center_lps is not None:
            transform.SetCenter(center_lps)
        return transform

    def get_affine_transform(self, image):
        scaling = np.asarray(self.scales).copy()
        rotation = np.asarray(self.degrees).copy()
        translation = np.asarray(self.translation).copy()

        if image.is_2d():
            scaling[2] = 1
            rotation[:-1] = 0

        if self.use_image_center:
            center_lps = image.get_center(lps=True)
        else:
            center_lps = None

        scaling_transform = self._get_scaling_transform(
            scaling,
            center_lps=center_lps,
        )
        rotation_transform = self._get_rotation_transform(
            rotation,
            translation,
            center_lps=center_lps,
        )

        sitk_major_version = get_major_sitk_version()
        if sitk_major_version == 1:
            composite = cast(
                _CompositeTransform,
                sitk.Transform(3, sitk.sitkComposite),
            )
            composite.AddTransform(scaling_transform)
            composite.AddTransform(rotation_transform)
            transform = cast(sitk.Transform, composite)
        elif sitk_major_version == 2:
            transforms = [scaling_transform, rotation_transform]
            transform = sitk.CompositeTransform(transforms)

        # ResampleImageFilter expects the transform from the output space to
        # the input space. Intuitively, the passed arguments should take us
        # from the input space to the output space, so we need to invert the
        # transform.
        # More info at https://github.com/TorchIO-project/torchio/discussions/693
        transform = transform.GetInverse()

        if self.invert_transform:
            transform = transform.GetInverse()

        return transform

    def get_default_pad_value(
        self, tensor: torch.Tensor, sitk_image: sitk.Image
    ) -> float:
        default_value: float
        if self.default_pad_value == 'minimum':
            default_value = tensor.min().item()
        elif self.default_pad_value == 'mean':
            default_value = get_borders_mean(
                sitk_image,
                filter_otsu=False,
            )
        elif self.default_pad_value == 'otsu':
            default_value = get_borders_mean(
                sitk_image,
                filter_otsu=True,
            )
        else:
            assert isinstance(self.default_pad_value, Number)
            default_value = float(self.default_pad_value)
        return default_value

    def apply_transform(self, subject: Subject) -> Subject:
        if self.check_shape:
            subject.check_consistent_spatial_shape()
        default_value: float
        for image in self.get_images(subject):
            transform = self.get_affine_transform(image)
            transformed_tensors = []
            for tensor in image.data:
                sitk_image = nib_to_sitk(
                    tensor[np.newaxis],
                    image.affine,
                    force_3d=True,
                )
                if image[TYPE] != INTENSITY:
                    interpolation = self.label_interpolation
                    default_value = self.default_pad_label
                else:
                    interpolation = self.image_interpolation
                    default_value = self.get_default_pad_value(tensor, sitk_image)
                transformed_tensor = self.apply_affine_transform(
                    sitk_image,
                    transform,
                    interpolation,
                    default_value,
                )
                transformed_tensors.append(transformed_tensor)
            image.set_data(torch.stack(transformed_tensors))
        return subject

    def apply_affine_transform(
        self,
        sitk_image: sitk.Image,
        transform: sitk.Transform,
        interpolation: str,
        default_value: float,
    ) -> torch.Tensor:
        floating = reference = sitk_image

        resampler = sitk.ResampleImageFilter()
        resampler.SetInterpolator(self.get_sitk_interpolator(interpolation))
        resampler.SetReferenceImage(reference)
        resampler.SetDefaultPixelValue(float(default_value))
        resampler.SetOutputPixelType(sitk.sitkFloat32)
        resampler.SetTransform(transform)
        resampled = resampler.Execute(floating)

        np_array = sitk.GetArrayFromImage(resampled)
        np_array = np_array.transpose()  # ITK to NumPy
        tensor = torch.as_tensor(np_array)
        return tensor


def get_borders_mean(image, filter_otsu=True):
    array = sitk.GetArrayViewFromImage(image)
    borders_tuple = (
        array[0, :, :],
        array[-1, :, :],
        array[:, 0, :],
        array[:, -1, :],
        array[:, :, 0],
        array[:, :, -1],
    )
    borders_flat = np.hstack([border.ravel() for border in borders_tuple])
    if not filter_otsu:
        return borders_flat.mean()
    borders_reshaped = borders_flat.reshape(1, 1, -1)
    borders_image = sitk.GetImageFromArray(borders_reshaped)
    otsu = sitk.OtsuThresholdImageFilter()
    otsu.Execute(borders_image)
    threshold = otsu.GetThreshold()
    values = borders_flat[borders_flat < threshold]
    if values.any():
        default_value = values.mean()
    else:
        default_value = borders_flat.mean()
    return default_value


def _parse_scales_isotropic(scales, isotropic):
    scales = to_tuple(scales)
    if isotropic and len(scales) in (3, 6):
        message = (
            'If "isotropic" is True, the value for "scales" must have'
            f' length 1 or 2, but "{scales}" was passed.'
            ' If you want to set isotropic scaling, use a single value or two values as a range'
            ' for the scaling factor. Refer to the documentation for more information.'
        )
        raise ValueError(message)


def _parse_default_value(value: str | float) -> str | float:
    if isinstance(value, Number) or value in ('minimum', 'otsu', 'mean'):
        return value
    message = (
        'Value for default_pad_value must be "minimum", "otsu", "mean" or a number'
    )
    raise ValueError(message)
