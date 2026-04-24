"""Transform composition: Compose, OneOf, SomeOf."""

from __future__ import annotations

import copy
from collections.abc import Mapping
from collections.abc import Sequence
from typing import Any
from typing import cast

import torch

from .transform import Transform


class Compose(Transform):
    """Compose several transforms together.

    The input is deep-copied once before the pipeline runs (by
    default), then each transform operates in-place on the copy.
    This avoids redundant copies when chaining many transforms.

    Args:
        transforms: Sequence of transforms to apply sequentially, or a
            mapping whose values are the transforms (keys are used as
            human-readable names and ignored at runtime).
        copy: If ``True`` (default), deep-copy the input before
            applying the pipeline. Set to ``False`` when this
            ``Compose`` is nested inside another ``Compose``.
        **kwargs: See [`Transform`][torchio.Transform] for additional
            keyword arguments.

    Examples:
        >>> import torchio as tio
        >>> preprocessing = tio.Compose([
        ...     tio.Flip(axes=(0,), p=0.5),
        ...     tio.Noise(std=(0.01, 0.1)),
        ... ])
        >>> augmented = preprocessing(subject)
        >>> named = tio.Compose({
        ...     "flip": tio.Flip(axes=(0,), p=0.5),
        ...     "noise": tio.Noise(std=(0.01, 0.1)),
        ... })
    """

    def __init__(
        self,
        transforms: Sequence[Transform] | Mapping[Any, Transform] | None = None,
        *,
        copy: bool = True,
        **kwargs: Any,
    ) -> None:
        super().__init__(copy=copy, **kwargs)
        if transforms is None:
            self.transforms: list[Transform] = []
        elif isinstance(transforms, Mapping):
            self.transforms = list(transforms.values())
        else:
            self.transforms = list(transforms)

    def forward(self, data):
        if self.copy:
            data = copy.deepcopy(data)
        subject, unwrap = self._wrap(data)
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
    """Apply one of the given transforms, chosen at random.

    Args:
        transforms: Sequence of transforms, or a ``dict`` mapping
            transforms to their relative weights. If a sequence is
            given, all transforms have equal probability.
        **kwargs: See [`Transform`][torchio.Transform] for additional
            keyword arguments.

    Examples:
        >>> import torchio as tio
        >>> augmentation = tio.OneOf({
        ...     tio.Noise(std=0.1): 0.7,
        ...     tio.Flip(axes=(0,)): 0.3,
        ... })
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

    def forward(self, data):
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
    """Apply a random subset of the given transforms.

    Args:
        transforms: Sequence of candidate transforms.
        num_transforms: How many transforms to apply. An ``int`` for a
            fixed count, or a ``(min, max)`` tuple to sample the count
            uniformly from that range.
        replace: If ``True``, sample with replacement (the same
            transform may be applied more than once).
        **kwargs: See [`Transform`][torchio.Transform] for additional
            keyword arguments.

    Examples:
        >>> import torchio as tio
        >>> augmentation = tio.SomeOf(
        ...     [tio.Noise(), tio.Flip(), tio.Noise(std=0.5)],
        ...     num_transforms=2,
        ... )
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

    def forward(self, data):
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
