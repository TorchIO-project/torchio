from __future__ import annotations

import warnings
from collections.abc import Sequence
from typing import TypeAlias
from typing import cast

import numpy as np
import torch

from ...data.subject import Subject
from ..transform import Transform
from . import RandomTransform

TypeTransformsDict: TypeAlias = dict[Transform, float] | Sequence[Transform]
HydraConfig: TypeAlias = dict[str, object]
HydraConfigDict: TypeAlias = dict[str, HydraConfig]


class Compose(Transform):
    """Compose several transforms together.

    Args:
        transforms: Sequence or dictionary of instances of
            [`Transform`][torchio.transforms.Transform]. If a dictionary
            is passed, the keys are used as names and the transforms can
            be accessed by name using ``compose["name"]``.
        **kwargs: See [`Transform`][torchio.transforms.Transform] for additional
            keyword arguments.
    """

    def __init__(
        self,
        transforms: Sequence[Transform] | dict[str, Transform],
        **kwargs,
    ):
        super().__init__(parse_input=False, **kwargs)
        if isinstance(transforms, dict):
            for key in transforms:
                if not isinstance(key, str):
                    message = (
                        'All keys in the transforms dictionary must be strings,'
                        f' but got key {key!r} of type {type(key).__name__!r}'
                    )
                    raise TypeError(message)
            transforms_dict = cast(dict[str, Transform], transforms)
            self._names: dict[str, int] = {
                name: i for i, name in enumerate(transforms_dict)
            }
            transforms_list = list(transforms_dict.values())
        else:
            self._names = {}
            transforms_list = list(transforms)
        for transform in transforms_list:
            if not callable(transform):
                message = (
                    'One or more of the objects passed to the Compose'
                    f' transform are not callable: "{transform}"'
                )
                raise TypeError(message)
        self.transforms = transforms_list

    def __len__(self):
        return len(self.transforms)

    def __getitem__(self, index: int | str | slice) -> Transform:
        if isinstance(index, str):
            if not self._names:
                message = (
                    f'String indexing is not supported for {type(self).__name__}'
                    ' instances created from a sequence'
                )
                raise TypeError(message)
            if index not in self._names:
                msg = f'Transform name not found: "{index}"'
                raise KeyError(msg)
            return self.transforms[self._names[index]]
        return self.transforms[index]

    def __repr__(self) -> str:
        return f'{self.name}({self.transforms})'

    def _get_base_args(self) -> dict[str, object]:
        init_args = super()._get_base_args()
        if 'parse_input' in init_args:
            init_args.pop('parse_input')
        return init_args

    def apply_transform(self, subject: Subject) -> Subject:
        for transform in self.transforms:
            subject = transform(subject)
        return subject

    def is_invertible(self) -> bool:
        return all(t.is_invertible() for t in self.transforms)

    def inverse(self, warn: bool = True) -> Compose:
        """Return a composed transform with inverted order and transforms.

        Args:
            warn: Issue a warning if some transforms are not invertible.
        """
        transforms = []
        for transform in self.transforms:
            if transform.is_invertible():
                transforms.append(transform.inverse())
            elif warn:
                message = f'Skipping {transform.name} as it is not invertible'
                warnings.warn(message, RuntimeWarning, stacklevel=2)
        transforms.reverse()
        result = Compose(transforms, **self._get_base_args())
        if not transforms and warn:
            warnings.warn(
                'No invertible transforms found',
                RuntimeWarning,
                stacklevel=2,
            )
        return result

    def to_hydra_config(self) -> HydraConfig:
        """Return a dictionary representation of the transform for Hydra instantiation."""
        transform_dict: HydraConfig = {'_target_': self._get_name_with_module()}
        transform_dict.update(self._get_reproducing_arguments())
        transforms_config: list[HydraConfig] = []
        for transform in self.transforms:
            transforms_config.append(transform.to_hydra_config())
        transform_dict['transforms'] = transforms_config
        return self._tuples_to_lists(transform_dict)


class OneOf(RandomTransform):
    """Apply only one of the given transforms.

    Args:
        transforms: Dictionary with instances of
            [`Transform`][torchio.transforms.Transform] as keys and
            probabilities as values. Probabilities are normalized so they sum
            to one. If a sequence is given, the same probability will be
            assigned to each transform.
        **kwargs: See [`Transform`][torchio.transforms.Transform] for additional
            keyword arguments.

    Examples:
        >>> import torchio as tio
        >>> colin = tio.datasets.Colin27()
        >>> transforms_dict = {
        ...     tio.RandomAffine(): 0.75,
        ...     tio.RandomElasticDeformation(): 0.25,
        ... }  # Using 3 and 1 as probabilities would have the same effect
        >>> transform = tio.OneOf(transforms_dict)
        >>> transformed = transform(colin)
    """

    def __init__(self, transforms: TypeTransformsDict, **kwargs):
        super().__init__(parse_input=False, **kwargs)
        self.transforms_dict = self._get_transforms_dict(transforms)

    def _get_base_args(self) -> dict:
        init_args = super()._get_base_args()
        if 'parse_input' in init_args:
            init_args.pop('parse_input')
        return init_args

    def apply_transform(self, subject: Subject) -> Subject:
        weights = torch.Tensor(list(self.transforms_dict.values()))
        index = torch.multinomial(weights, 1)
        transforms = list(self.transforms_dict.keys())
        transform = transforms[index]
        transformed = transform(subject)
        return transformed

    def _get_transforms_dict(
        self,
        transforms: TypeTransformsDict,
    ) -> dict[Transform, float]:
        if isinstance(transforms, dict):
            transforms_dict = dict(transforms)
            self._normalize_probabilities(transforms_dict)
        else:
            try:
                p = 1 / len(transforms)
            except TypeError as e:
                message = (
                    'Transforms argument must be a dictionary or a sequence,'
                    f' not {type(transforms)}'
                )
                raise ValueError(message) from e
            transforms_dict = {transform: p for transform in transforms}
        for transform in transforms_dict:
            if not isinstance(transform, Transform):
                message = (
                    'All keys in transform_dict must be instances of'
                    f'torchio.Transform, not "{type(transform)}"'
                )
                raise ValueError(message)
        return transforms_dict

    @staticmethod
    def _normalize_probabilities(
        transforms_dict: dict[Transform, float],
    ) -> None:
        probabilities = np.array(list(transforms_dict.values()), dtype=float)
        if np.any(probabilities < 0):
            message = (
                f'Probabilities must be greater or equal to zero, not "{probabilities}"'
            )
            raise ValueError(message)
        if np.all(probabilities == 0):
            message = (
                'At least one probability must be greater than zero,'
                f' but they are "{probabilities}"'
            )
            raise ValueError(message)
        for transform, probability in transforms_dict.items():
            transforms_dict[transform] = probability / probabilities.sum()
