"""Transform composition: Compose, OneOf, SomeOf."""

from __future__ import annotations

import copy
from collections.abc import Sequence
from typing import Any

import attrs
import torch
from torch import Tensor

from ..data.image import Image
from ..data.subject import Subject
from .transform import Transform


@attrs.define(slots=False, eq=False, kw_only=True, repr=False)
class Compose(Transform):
    """Apply a sequence of transforms.

    By default, the input is deep-copied before the pipeline runs,
    so the original data is never modified. Set ``copy=False`` to
    transform in-place (useful inside an outer ``Compose`` that
    already copied).

    Args:
        transforms: Sequence of transforms to apply.
        copy: Deep-copy the input before applying transforms.
    """

    transforms: list[Transform] = attrs.field(factory=list)
    copy: bool = True

    def __init__(
        self,
        transforms: Sequence[Transform] | None = None,
        *,
        copy: bool = True,
        **kwargs: Any,
    ) -> None:
        self.__attrs_init__(
            transforms=list(transforms) if transforms else [],
            copy=copy,
            **kwargs,
        )

    def forward(
        self,
        data: Subject | Image | Tensor,
    ) -> Subject | Image | Tensor:
        subject, unwrap = self._wrap(data)
        if self.copy:
            subject = copy.deepcopy(subject)
        for transform in self.transforms:
            # Skip child copying — Compose already copied
            old_copy = transform.copy
            transform.copy = False
            subject = transform(subject)
            transform.copy = old_copy
        return unwrap(subject)


@attrs.define(slots=False, eq=False, kw_only=True, repr=False)
class OneOf(Transform):
    """Randomly pick one transform from a collection.

    Args:
        transforms: A sequence of transforms, or a dict mapping
            transforms to their relative weights.
    """

    transforms: list[Transform] = attrs.field(factory=list)
    weights: list[float] = attrs.field(factory=list)

    def __init__(
        self,
        transforms: Sequence[Transform] | dict[Transform, float],
        **kwargs: Any,
    ) -> None:
        if isinstance(transforms, dict):
            t_list = list(transforms.keys())
            w_list = list(transforms.values())
            total = sum(w_list)
            w_list = [w / total for w in w_list]
        else:
            t_list = list(transforms)
            n = len(t_list)
            w_list = [1.0 / n] * n
        self.__attrs_init__(transforms=t_list, weights=w_list, **kwargs)

    def forward(
        self,
        data: Subject | Image | Tensor,
    ) -> Subject | Image | Tensor:
        subject, unwrap = self._wrap(data)
        if torch.rand(1).item() > self.p:
            return unwrap(subject)
        idx = torch.multinomial(
            torch.tensor(self.weights),
            num_samples=1,
        ).item()
        subject = self.transforms[idx](subject)
        return unwrap(subject)


@attrs.define(slots=False, eq=False, kw_only=True, repr=False)
class SomeOf(Transform):
    """Randomly pick N transforms from a collection.

    Args:
        transforms: Sequence of transforms to sample from.
        num_transforms: Number to apply. An ``int`` for a fixed count,
            or a ``(min, max)`` tuple to sample uniformly.
        replace: Sample with replacement.
    """

    transforms: list[Transform] = attrs.field(factory=list)
    num_transforms: int | tuple[int, int] = 1
    replace: bool = False

    def __init__(
        self,
        transforms: Sequence[Transform] | None = None,
        *,
        num_transforms: int | tuple[int, int] = 1,
        replace: bool = False,
        **kwargs: Any,
    ) -> None:
        self.__attrs_init__(
            transforms=list(transforms) if transforms else [],
            num_transforms=num_transforms,
            replace=replace,
            **kwargs,
        )

    @property
    def _min_n(self) -> int:
        if isinstance(self.num_transforms, int):
            return self.num_transforms
        return self.num_transforms[0]

    @property
    def _max_n(self) -> int:
        if isinstance(self.num_transforms, int):
            return self.num_transforms
        return self.num_transforms[1]

    def forward(
        self,
        data: Subject | Image | Tensor,
    ) -> Subject | Image | Tensor:
        subject, unwrap = self._wrap(data)
        if torch.rand(1).item() > self.p:
            return unwrap(subject)
        n = torch.randint(self._min_n, self._max_n + 1, (1,)).item()
        n_transforms = len(self.transforms)
        if self.replace:
            indices = torch.randint(0, n_transforms, (n,))
        else:
            n = min(n, n_transforms)
            indices = torch.randperm(n_transforms)[:n]
        for idx in indices:
            subject = self.transforms[idx](subject)
        return unwrap(subject)
