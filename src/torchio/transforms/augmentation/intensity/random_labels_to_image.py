from __future__ import annotations

from collections.abc import Sequence
from typing import TypeVar

import torch

from ....data.image import LabelMap
from ....data.image import ScalarImage
from ....data.subject import Subject
from ....types import TypeRangeFloat
from ....utils import check_sequence
from ...intensity_transform import IntensityTransform
from .. import RandomTransform

GaussianParameterT = TypeVar('GaussianParameterT')


class RandomLabelsToImage(RandomTransform, IntensityTransform):
    r"""Randomly generate an image from a segmentation.

    Based on the work by Billot et al.: [A Learning Strategy for Contrast-agnostic MRI Segmentation](http://proceedings.mlr.press/v121/billot20a.html)
    and [Partial Volume Segmentation of Brain MRI Scans of any Resolution and Contrast](https://link.springer.com/chapter/10.1007/978-3-030-59728-3_18).




    Args:
        label_key: String designating the label map in the subject
            that will be used to generate the new image.
        used_labels: Sequence of integers designating the labels used
            to generate the new image. If categorical encoding is used,
            `label_channels` refers to the values of the
            categorical encoding. If one hot encoding or partial-volume
            label maps are used, `label_channels` refers to the
            channels of the label maps.
            Default uses all labels. Missing voxels will be filled with zero
            or with voxels from an already existing volume,
            see `image_key`.
        image_key: String designating the key to which the new volume will be
            saved. If this key corresponds to an already existing volume,
            missing voxels will be filled with the corresponding values
            in the original volume.
        mean: Sequence of means for each label.
            For each value $v$, if a tuple $(a, b)$ is
            provided then $v \sim \mathcal{U}(a, b)$.
            If `None`, `default_mean` range will be used for every
            label.
            If not `None` and `label_channels` is not `None`,
            `mean` and `label_channels` must have the
            same length.
        std: Sequence of standard deviations for each label.
            For each value $v$, if a tuple $(a, b)$ is
            provided then $v \sim \mathcal{U}(a, b)$.
            If `None`, `default_std` range will be used for every
            label.
            If not `None` and `label_channels` is not `None`,
            `std` and `label_channels` must have the
            same length.
        default_mean: Default mean range.
        default_std: Default standard deviation range.
        discretize: If `True`, partial-volume label maps will be discretized.
            Does not have any effects if not using partial-volume label maps.
            Discretization is done taking the class of the highest value per
            voxel in the different partial-volume label maps using
            `torch.argmax()` on the channel dimension (i.e. 0).
        ignore_background: If `True`, input voxels labeled as `0` will not
            be modified.
        **kwargs: See [`Transform`][torchio.transforms.Transform] for additional
            keyword arguments.

    Tip:
        It is recommended to blur the new images in order to simulate
        partial volume effects at the borders of the synthetic structures. See
        [`RandomBlur`][torchio.transforms.augmentation.intensity.random_blur.RandomBlur].

    Examples:
        >>> import torchio as tio
        >>> subject = tio.datasets.ICBM2009CNonlinearSymmetric()
        >>> # Using the default parameters
        >>> transform = tio.RandomLabelsToImage(label_key='tissues')
        >>> # Using custom mean and std
        >>> transform = tio.RandomLabelsToImage(
        ...     label_key='tissues', mean=[0.33, 0.66, 1.], std=[0, 0, 0]
        ... )
        >>> # Discretizing the partial volume maps and blurring the result
        >>> simulation_transform = tio.RandomLabelsToImage(
        ...     label_key='tissues', mean=[0.33, 0.66, 1.], std=[0, 0, 0], discretize=True
        ... )
        >>> blurring_transform = tio.RandomBlur(std=0.3)
        >>> transform = tio.Compose([simulation_transform, blurring_transform])
        >>> transformed = transform(subject)  # subject has a new key 'image_from_labels' with the simulated image
        >>> # Filling holes of the simulated image with the original T1 image
        >>> rescale_transform = tio.RescaleIntensity(
        ...     out_min_max=(0, 1), percentiles=(1, 99))   # Rescale intensity before filling holes
        >>> simulation_transform = tio.RandomLabelsToImage(
        ...     label_key='tissues',
        ...     image_key='t1',
        ...     used_labels=[0, 1]
        ... )
        >>> transform = tio.Compose([rescale_transform, simulation_transform])
        >>> transformed = transform(subject)  # subject's key 't1' has been replaced with the simulated image

    !!! note "See also"
        [`RemapLabels`][torchio.transforms.preprocessing.label.remap_labels.RemapLabels].

    """

    def __init__(
        self,
        label_key: str | None = None,
        used_labels: Sequence[int] | None = None,
        image_key: str = 'image_from_labels',
        mean: Sequence[TypeRangeFloat] | None = None,
        std: Sequence[TypeRangeFloat] | None = None,
        default_mean: TypeRangeFloat = (0.1, 0.9),
        default_std: TypeRangeFloat = (0.01, 0.1),
        discretize: bool = False,
        ignore_background: bool = False,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.label_key = _parse_label_key(label_key)
        self.used_labels = _parse_used_labels(used_labels)
        self.mean_ranges: list[tuple[float, float]] | None
        self.std_ranges: list[tuple[float, float]] | None
        self.mean_ranges, self.std_ranges = self.parse_mean_and_std(mean, std)
        self.default_mean = self.parse_gaussian_parameter(
            default_mean,
            'default_mean',
        )
        self.default_std = self.parse_gaussian_parameter(
            default_std,
            'default_std',
        )
        self.image_key = image_key
        self.discretize = discretize
        self.ignore_background = ignore_background

    def parse_mean_and_std(
        self,
        mean: Sequence[TypeRangeFloat] | None,
        std: Sequence[TypeRangeFloat] | None,
    ) -> tuple[list[tuple[float, float]] | None, list[tuple[float, float]] | None]:
        if mean is not None:
            mean = self.parse_gaussian_parameters(mean, 'mean')
        if std is not None:
            std = self.parse_gaussian_parameters(std, 'std')
        if mean is not None and std is not None:
            message = (
                'If both "mean" and "std" are defined they must have the samelength'
            )
            assert len(mean) == len(std), message
        return mean, std

    def parse_gaussian_parameters(
        self,
        params: Sequence[TypeRangeFloat],
        name: str,
    ) -> list[tuple[float, float]]:
        check_sequence(params, name)
        parsed_params: list[tuple[float, float]] = [
            self.parse_gaussian_parameter(p, f'{name}[{i}]')
            for i, p in enumerate(params)
        ]
        if self.used_labels is not None:
            message = (
                f'If both "{name}" and "used_labels" are defined, '
                'they must have the same length'
            )
            assert len(parsed_params) == len(self.used_labels), message
        return parsed_params

    @staticmethod
    def parse_gaussian_parameter(
        nums_range: TypeRangeFloat,
        name: str,
    ) -> tuple[float, float]:
        if isinstance(nums_range, (int, float)):
            return nums_range, nums_range

        if len(nums_range) != 2:
            raise ValueError(
                f'If {name} is a sequence, it must be of len 2, not {nums_range}',
            )
        min_value, max_value = nums_range
        if min_value > max_value:
            raise ValueError(
                f'If {name} is a sequence, the second value must be'
                f' equal or greater than the first, not {nums_range}',
            )
        return min_value, max_value

    def _guess_label_key(self, subject: Subject) -> None:
        if self.label_key is None:
            iterable = subject.get_images_dict(intensity_only=False).items()
            for name, image in iterable:
                if isinstance(image, LabelMap):
                    self.label_key = name
                    break
            else:
                message = f'No label maps found in subject: {subject}'
                raise RuntimeError(message)

    def apply_transform(self, subject: Subject) -> Subject:
        self._guess_label_key(subject)
        assert self.label_key is not None

        means: list[float] = []
        stds: list[float] = []
        label_map = subject.get_label_map(self.label_key).data

        # Find out if we face a partial-volume image or a label map.
        # One-hot-encoded label map is considered as a partial-volume image
        all_discrete = label_map.eq(label_map.float().round()).all()
        same_num_dims = label_map.squeeze().dim() < label_map.dim()
        is_discretized = all_discrete and same_num_dims

        if not is_discretized and self.discretize:
            # Take label with highest value in voxel
            max_label, label_map = label_map.max(dim=0, keepdim=True)
            # Remove values where all labels are 0 (i.e. missing labels)
            label_map[max_label == 0] = -1
            is_discretized = True

        if is_discretized:
            labels = label_map.unique().long().tolist()
            if -1 in labels:
                labels.remove(-1)
        else:
            labels = range(label_map.shape[0])

        # Raise error if mean and std are not defined for every label
        _check_mean_and_std_length(labels, self.mean_ranges, self.std_ranges)

        for label in labels:
            mean, std = self.get_params(label)
            means.append(mean)
            stds.append(std)

        transform = LabelsToImage(
            label_key=self.label_key,
            mean=means,
            std=stds,
            image_key=self.image_key,
            used_labels=self.used_labels,
            ignore_background=self.ignore_background,
            discretize=self.discretize,
            **self.get_base_args(),
        )
        transformed = transform(subject)
        assert isinstance(transformed, Subject)
        return transformed

    def get_params(self, label: int) -> tuple[float, float]:
        if self.mean_ranges is None:
            mean_range = self.default_mean
        else:
            mean_range = self.mean_ranges[label]
        if self.std_ranges is None:
            std_range = self.default_std
        else:
            std_range = self.std_ranges[label]
        mean = self.sample_uniform(*mean_range)
        std = self.sample_uniform(*std_range)
        return mean, std

    def _get_named_arguments(self) -> dict[str, object]:
        return {
            'label_key': self.label_key,
            'used_labels': self.used_labels,
            'image_key': self.image_key,
            'mean': self.mean_ranges,
            'std': self.std_ranges,
            'default_mean': self.default_mean,
            'default_std': self.default_std,
            'discretize': self.discretize,
            'ignore_background': self.ignore_background,
        }


class LabelsToImage(IntensityTransform):
    r"""Generate an image from a segmentation.

    Args:
        label_key: String designating the label map in the subject
            that will be used to generate the new image.
        used_labels: Sequence of integers designating the labels used
            to generate the new image. If categorical encoding is used,
            `label_channels` refers to the values of the
            categorical encoding. If one hot encoding or partial-volume
            label maps are used, `label_channels` refers to the
            channels of the label maps.
            Default uses all labels. Missing voxels will be filled with zero
            or with voxels from an already existing volume,
            see `image_key`.
        image_key: String designating the key to which the new volume will be
            saved. If this key corresponds to an already existing volume,
            missing voxels will be filled with the corresponding values
            in the original volume.
        mean: Sequence of means for each label.
            If not `None` and `label_channels` is not `None`,
            `mean` and `label_channels` must have the
            same length.
        std: Sequence of standard deviations for each label.
            If not `None` and `label_channels` is not `None`,
            `std` and `label_channels` must have the
            same length.
        discretize: If `True`, partial-volume label maps will be discretized.
            Does not have any effects if not using partial-volume label maps.
            Discretization is done taking the class of the highest value per
            voxel in the different partial-volume label maps using
            `torch.argmax()` on the channel dimension (i.e. 0).
        ignore_background: If `True`, input voxels labeled as `0` will not
            be modified.
        **kwargs: See [`Transform`][torchio.transforms.Transform] for additional
            keyword arguments.

    Note:
        It is recommended to blur the new images to make the result more
        realistic. See
        [`RandomBlur`][torchio.transforms.augmentation.RandomBlur].
    """

    def __init__(
        self,
        label_key: str,
        mean: Sequence[float] | None,
        std: Sequence[float] | None,
        image_key: str = 'image_from_labels',
        used_labels: Sequence[int] | None = None,
        ignore_background: bool = False,
        discretize: bool = False,
        **kwargs,
    ):
        super().__init__(**kwargs)
        parsed_label_key = _parse_label_key(label_key)
        assert parsed_label_key is not None
        self.label_key: str = parsed_label_key
        self.used_labels = _parse_used_labels(used_labels)
        self.means: Sequence[float] | None = mean
        self.stds: Sequence[float] | None = std
        self.image_key = image_key
        self.ignore_background = ignore_background
        self.discretize = discretize
        self.args_names = [
            'label_key',
            'mean',
            'std',
            'image_key',
            'used_labels',
            'ignore_background',
            'discretize',
        ]

    def apply_transform(self, subject: Subject) -> Subject:
        original_image = (
            subject.get_scalar_image(self.image_key)
            if self.image_key in subject
            else None
        )

        label_map_image = subject.get_label_map(self.label_key)
        label_map = label_map_image.data
        affine = label_map_image.affine

        # Find out if we face a partial-volume image or a label map.
        # One-hot-encoded label map is considered as a partial-volume image
        all_discrete = label_map.eq(label_map.float().round()).all()
        same_num_dims = label_map.squeeze().dim() < label_map.dim()
        is_discretized = all_discrete and same_num_dims

        if not is_discretized and self.discretize:
            # Take label with highest value in voxel
            max_label, label_map = label_map.max(dim=0, keepdim=True)
            # Remove values where all labels are 0 (i.e. missing labels)
            label_map[max_label == 0] = -1
            is_discretized = True

        tissues = torch.zeros(1, *label_map_image.spatial_shape).float()
        if is_discretized:
            labels_in_image = label_map.unique().long().tolist()
            if -1 in labels_in_image:
                labels_in_image.remove(-1)
        else:
            labels_in_image = range(label_map.shape[0])

        # Raise error if mean and std are not defined for every label
        _check_mean_and_std_length(
            labels_in_image,
            self.means,
            self.stds,
        )

        for i, label in enumerate(labels_in_image):
            if label == 0 and self.ignore_background:
                continue
            if self.used_labels is None or label in self.used_labels:
                assert self.means is not None
                assert self.stds is not None
                mean = self.means[i]
                std = self.stds[i]
                if is_discretized:
                    mask = label_map == label
                else:
                    mask = label_map[label]
                tissues += self.generate_tissue(mask, mean, std)

            else:
                # Modify label map to easily compute background mask
                if is_discretized:
                    label_map[label_map == label] = -1
                else:
                    label_map[label] = 0

        final_image = ScalarImage(affine=affine, tensor=tissues)

        if original_image is not None:
            if is_discretized:
                bg_mask = label_map == -1
            else:
                bg_mask = label_map.sum(dim=0, keepdim=True) < 0.5
            final_image.data[bg_mask] = original_image.data[bg_mask].float()

        subject.add_image(final_image, self.image_key)
        return subject

    def _get_named_arguments(self) -> dict[str, object]:
        return {
            'label_key': self.label_key,
            'mean': self.means,
            'std': self.stds,
            'image_key': self.image_key,
            'used_labels': self.used_labels,
            'ignore_background': self.ignore_background,
            'discretize': self.discretize,
        }

    @staticmethod
    def generate_tissue(
        data: torch.Tensor,
        mean: float,
        std: float,
    ) -> torch.Tensor:
        # Create the simulated tissue using a gaussian random variable
        gaussian = torch.randn(data.shape) * std + mean
        return gaussian * data


def _parse_label_key(label_key: str | None) -> str | None:
    if label_key is not None and not isinstance(label_key, str):
        message = f'"label_key" must be a string or None, not {type(label_key)}'
        raise TypeError(message)
    return label_key


def _parse_used_labels(
    used_labels: Sequence[int] | None,
) -> Sequence[int] | None:
    if used_labels is None:
        return None
    check_sequence(used_labels, 'used_labels')
    for e in used_labels:
        if not isinstance(e, int):
            message = (
                'Items in "used_labels" must be integers,'
                f' but some are not: {used_labels}'
            )
            raise ValueError(message)
    return used_labels


def _check_mean_and_std_length(
    labels: Sequence[int],
    means: Sequence[GaussianParameterT] | None,
    stds: Sequence[GaussianParameterT] | None,
) -> None:
    num_labels = len(labels)
    if means is not None:
        num_means = len(means)
        message = (
            '"mean" must define a value for each label but length of "mean"'
            f' is {num_means} while {num_labels} labels were found'
        )
        if num_means != num_labels:
            raise RuntimeError(message)
    if stds is not None:
        num_stds = len(stds)
        message = (
            '"std" must define a value for each label but length of "std"'
            f' is {num_stds} while {num_labels} labels were found'
        )
        if num_stds != num_labels:
            raise RuntimeError(message)
