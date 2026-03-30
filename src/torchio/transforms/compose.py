"""Transform composition: Compose, OneOf, SomeOf."""

from __future__ import annotations

import copy
from collections.abc import Sequence
from typing import Any
from typing import cast

import torch

from .transform import T
from .transform import Transform


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

    def __init__(
        self,
        transforms: Sequence[Transform] | None = None,
        *,
        copy: bool = True,
        **kwargs: Any,
    ) -> None:
        super().__init__(copy=copy, **kwargs)
        self.transforms = list(transforms) if transforms else []

    def forward(self, data: T) -> T:
        subject, unwrap = self._wrap(data)
        if self.copy:
            subject = copy.deepcopy(subject)
        for transform in self.transforms:
            old_copy = transform.copy
            transform.copy = False
            subject = transform(subject)
            transform.copy = old_copy
        return unwrap(subject)

    def to_hydra(self) -> dict[str, Any]:
        cfg = super().to_hydra()
        cfg["transforms"] = [t.to_hydra() for t in self.transforms]
        return cfg


class OneOf(Transform):
    """Randomly pick one transform from a collection.

    Args:
        transforms: A sequence of transforms, or a dict mapping
            transforms to their relative weights.
    """

    def __init__(
        self,
        transforms: Sequence[Transform] | dict[Transform, float],
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        if isinstance(transforms, dict):
            weight_dict = cast(dict[Transform, float], transforms)
            self.transforms = list(weight_dict.keys())
            w_list: list[float] = list(weight_dict.values())
            total: float = sum(w_list)
            self.weights = [w / total for w in w_list]
        else:
            self.transforms = list(transforms)
            n = len(self.transforms)
            self.weights = [1.0 / n] * n

    def forward(self, data: T) -> T:
        subject, unwrap = self._wrap(data)
        if torch.rand(1).item() > self.p:
            return unwrap(subject)
        idx = int(
            torch.multinomial(
                torch.tensor(self.weights),
                num_samples=1,
            ).item()
        )
        subject = self.transforms[idx](subject)
        return unwrap(subject)

    def to_hydra(self) -> dict[str, Any]:
        cfg = super().to_hydra()
        cfg["transforms"] = [t.to_hydra() for t in self.transforms]
        return cfg


class SomeOf(Transform):
    """Randomly pick N transforms from a collection.

    Args:
        transforms: Sequence of transforms to sample from.
        num_transforms: Number to apply. An ``int`` for a fixed count,
            or a ``(min, max)`` tuple to sample uniformly.
        replace: Sample with replacement.
    """

    def __init__(
        self,
        transforms: Sequence[Transform] | None = None,
        *,
        num_transforms: int | tuple[int, int] = 1,
        replace: bool = False,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.transforms = list(transforms) if transforms else []
        self.num_transforms = num_transforms
        self.replace = replace

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

    def forward(self, data: T) -> T:
        subject, unwrap = self._wrap(data)
        if torch.rand(1).item() > self.p:
            return unwrap(subject)
        n = int(torch.randint(self._min_n, self._max_n + 1, size=(1,)).item())
        n_transforms = len(self.transforms)
        if self.replace:
            indices = torch.randint(0, n_transforms, (n,))
        else:
            n = min(n, n_transforms)
            indices = torch.randperm(n_transforms)[:n]
        for idx in indices:
            subject = self.transforms[idx](subject)
        return unwrap(subject)

    def to_hydra(self) -> dict[str, Any]:
        cfg = super().to_hydra()
        cfg["transforms"] = [t.to_hydra() for t in self.transforms]
        return cfg
