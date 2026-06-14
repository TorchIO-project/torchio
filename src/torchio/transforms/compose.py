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
        copy: If `True` (default), deep-copy the input before
            applying the pipeline. Set to `False` when this
            `Compose` is nested inside another `Compose`.
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
        transforms: Sequence[Transform] | Mapping[str, Transform] | None = None,
        *,
        copy: bool = True,
        **kwargs: Any,
    ) -> None:
        super().__init__(copy=copy, **kwargs)
        if transforms is None:
            self.transforms: list[Transform] = []
        elif isinstance(transforms, Mapping):
            mapping = cast(Mapping[str, Transform], transforms)
            self.transforms = list(mapping.values())
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

    When applied to a batch with `per_instance=True` (the default),
    each batch element independently chooses which transform to apply.
    This requires shape- and schema-preserving transforms so the
    elements can be re-stacked. Pass `per_instance=False` to choose a
    single transform for the whole batch.

    Args:
        transforms: Sequence of transforms, or a `dict` mapping
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
        if self.copy:
            data = copy.deepcopy(data)
        batch, unwrap = self._wrap(data)
        if self.per_instance and batch.batch_size > 1:
            return unwrap(self._forward_per_element(batch))
        if torch.rand(1).item() >= self.p:
            return unwrap(batch)
        idx = int(
            torch.multinomial(
                torch.tensor(self.weights),
                num_samples=1,
            ).item()
        )
        batch = self.transforms[idx](batch)
        return unwrap(batch)

    def _forward_per_element(self, batch):
        """Apply an independently chosen transform to each batch element."""
        if self.p == 0:
            return batch
        weights = torch.tensor(self.weights)
        out_subjects = []
        any_applied = False
        for subject in batch.unbatch():
            if torch.rand(1).item() < self.p:
                any_applied = True
                idx = int(torch.multinomial(weights, num_samples=1).item())
                subject = _apply_to_element(subject, self.transforms[idx])
            out_subjects.append(subject)
        if not any_applied:
            return batch
        return _rebatch_with_history(out_subjects, "OneOf")

    def to_hydra(self) -> dict[str, Any]:
        cfg = super().to_hydra()
        cfg["transforms"] = [t.to_hydra() for t in self.transforms]
        return cfg


class SomeOf(Transform):
    """Apply a random subset of the given transforms.

    When applied to a batch with `per_instance=True` (the default),
    each batch element independently samples its own subset. This
    requires shape- and schema-preserving transforms so the elements
    can be re-stacked. Pass `per_instance=False` to sample a single
    subset for the whole batch.

    Args:
        transforms: Sequence of candidate transforms.
        num_transforms: How many transforms to apply. An `int` for a
            fixed count, or a `(min, max)` tuple to sample the count
            uniformly from that range.
        replace: If `True`, sample with replacement (the same
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
        if self.copy:
            data = copy.deepcopy(data)
        batch, unwrap = self._wrap(data)
        if self.per_instance and batch.batch_size > 1:
            return unwrap(self._forward_per_element(batch))
        if torch.rand(1).item() >= self.p:
            return unwrap(batch)
        batch = self._apply_subset(batch)
        return unwrap(batch)

    def _apply_subset(self, batch):
        """Apply a randomly chosen subset of transforms to *batch*."""
        n = int(torch.randint(self._min_n, self._max_n + 1, size=(1,)).item())
        n_transforms = len(self.transforms)
        if self.replace:
            indices = torch.randint(0, n_transforms, (n,))
        else:
            n = min(n, n_transforms)
            indices = torch.randperm(n_transforms)[:n]
        for idx in indices:
            batch = self.transforms[idx](batch)
        return batch

    def _forward_per_element(self, batch):
        """Apply an independently chosen subset to each batch element."""
        if self.p == 0:
            return batch
        out_subjects = []
        any_applied = False
        for subject in batch.unbatch():
            if torch.rand(1).item() < self.p:
                any_applied = True
                subject = _apply_to_element(subject, self._apply_subset)
            out_subjects.append(subject)
        if not any_applied:
            return batch
        return _rebatch_with_history(out_subjects, "SomeOf")

    def to_hydra(self) -> dict[str, Any]:
        cfg = super().to_hydra()
        cfg["transforms"] = [t.to_hydra() for t in self.transforms]
        return cfg


def _apply_to_element(subject: Any, apply_fn: Any) -> Any:
    """Apply a transform (or callable) to one element, preserving history.

    Wrapping a `Subject` directly into a batch discards its existing
    history, so the element is wrapped into a one-element batch seeded
    with its prior history; the transform then appends to that history.

    Args:
        subject: The single subject to transform (carrying its history).
        apply_fn: A transform or callable taking and returning a
            `SubjectsBatch`.

    Returns:
        The transformed subject, with its full history.
    """
    from ..data.batch import SubjectsBatch

    element_batch = SubjectsBatch.from_subjects([subject])
    element_batch.applied_transforms = list(subject.applied_transforms)
    element_batch = apply_fn(element_batch)
    return element_batch.unbatch()[0]


def _rebatch_with_history(subjects: list[Any], transform_name: str) -> Any:
    """Re-stack per-element subjects and freeze their distinct histories.

    Args:
        subjects: The transformed subjects, one per batch element.
        transform_name: Name of the branching transform, used for a
            clearer error message when shapes or schemas diverge.

    Returns:
        A `SubjectsBatch` whose `unbatch()` restores each element's own
        transform history.
    """
    from ..data.batch import SubjectsBatch

    _check_consistent_schema(subjects, transform_name)
    try:
        batch = SubjectsBatch.from_subjects(subjects)
    except (RuntimeError, KeyError) as error:
        msg = (
            f"Per-instance {transform_name} produced batch elements with"
            " different shapes or schemas, which cannot be re-stacked. Use"
            " only shape- and schema-preserving transforms with per-instance"
            f" {transform_name}, or pass per_instance=False."
        )
        raise RuntimeError(msg) from error
    batch.set_per_element_history([s.applied_transforms for s in subjects])
    return batch


def _check_consistent_schema(subjects: list[Any], transform_name: str) -> None:
    """Ensure all subjects share the same image names and classes.

    Per-element branching may apply different transforms to different
    elements; if those change the set of images (or their type), the
    elements can no longer be re-stacked into one batch. This raises a
    clear error instead of silently dropping data.

    Args:
        subjects: The subjects about to be re-stacked.
        transform_name: Name of the branching transform for the message.

    Raises:
        RuntimeError: If image names or classes differ across subjects.
    """
    if not subjects:
        return
    reference = {name: type(image) for name, image in subjects[0].images.items()}
    for subject in subjects[1:]:
        current = {name: type(image) for name, image in subject.images.items()}
        if current != reference:
            msg = (
                f"Per-instance {transform_name} produced batch elements with"
                " different image names or types, which cannot be re-stacked."
                " Use only schema-preserving transforms with per-instance"
                f" {transform_name}, or pass per_instance=False."
            )
            raise RuntimeError(msg)
