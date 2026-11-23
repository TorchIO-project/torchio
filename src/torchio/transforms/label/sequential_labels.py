"""SequentialLabels: renumber labels to consecutive integers."""

from __future__ import annotations

from typing import Any

import torch

from ...data.batch import SubjectsBatch
from ...data.image import LabelMap
from ..transform import Transform


class SequentialLabels(Transform):
    r"""Renumber labels in label maps to consecutive integers starting from 0.

    For example, if a label map has values `{0, 5, 10}`, this
    transform remaps them to `{0, 1, 2}`.

    Only [`LabelMap`][torchio.LabelMap] images are affected.

    Note:
        The background (label 0) is always mapped to 0. Even if there
        are no zeros in the input, zero will appear in the output.

    Note:
        Each sample in a batch is renumbered from its own set of label
        values, since different subjects can have different labels
        present.

    Args:
        **kwargs: See [`Transform`][torchio.Transform].

    Examples:
        >>> import torchio as tio
        >>> transform = tio.SequentialLabels()
    """

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)

    def make_params(self, batch: SubjectsBatch) -> dict[str, Any]:
        """Compute the remapping from each sample's own labels."""
        remappings: dict[str, list[dict[int, int]]] = {}
        for name, img_batch in batch.images.items():
            if not issubclass(img_batch._image_class, LabelMap):
                continue
            sample_remappings = []
            for sample_data in img_batch.data:
                unique = sorted(int(v) for v in sample_data.unique().tolist())
                sample_remappings.append({old: new for new, old in enumerate(unique)})
            remappings[name] = sample_remappings
        return {"remappings": remappings}

    def apply_transform(
        self,
        batch: SubjectsBatch,
        params: dict[str, Any],
    ) -> SubjectsBatch:
        """Apply sequential renumbering."""
        remappings = params["remappings"]
        for name, img_batch in batch.images.items():
            if name not in remappings:
                continue
            data = torch.zeros_like(img_batch.data)
            for sample_index, remapping in enumerate(remappings[name]):
                for old, new in remapping.items():
                    data[sample_index][img_batch.data[sample_index] == old] = new
            img_batch.data = data
        return batch

    @property
    def invertible(self) -> bool:
        """Whether this transform can be inverted."""
        return True

    def inverse(self, params: dict[str, Any]) -> _SequentialLabelsInverse:
        """Invert by restoring original label values."""
        return _SequentialLabelsInverse(
            remappings=params["remappings"],
            copy=False,
        )


class _SequentialLabelsInverse(Transform):
    """Inverse of SequentialLabels."""

    def __init__(
        self,
        *,
        remappings: dict[str, list[dict[int, int]]],
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self._remappings = remappings

    def make_params(self, batch: SubjectsBatch) -> dict[str, Any]:
        return {}

    def apply_transform(
        self,
        batch: SubjectsBatch,
        params: dict[str, Any],
    ) -> SubjectsBatch:
        for name, img_batch in batch.images.items():
            if name not in self._remappings:
                continue
            data = torch.zeros_like(img_batch.data)
            for sample_index, remapping in enumerate(self._remappings[name]):
                inverse = {v: k for k, v in remapping.items()}
                for old, new in inverse.items():
                    data[sample_index][img_batch.data[sample_index] == old] = new
            img_batch.data = data
        return batch
