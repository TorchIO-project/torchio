"""Mask: zero out voxels outside a mask region."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import torch
from torch import Tensor

from ...data.batch import SubjectsBatch
from ...data.image import LabelMap
from ..transform import IntensityTransform


class Mask(IntensityTransform):
    """Set voxels outside a mask to a constant value.

    Useful for brain extraction, region-of-interest cropping, or
    zeroing out background before training.

    Args:
        masking_method: Defines the mask. Can be:

            - A `str`: key to a [`LabelMap`][torchio.LabelMap] in
              the subject.
            - A callable: receives the image tensor and returns a
              boolean mask.
        outside_value: Value to assign to voxels outside the mask.
        labels: If using a label map, which label values to include
            in the mask. `None` means all nonzero values.
        **kwargs: See [`Transform`][torchio.Transform].

    Examples:
        >>> import torchio as tio
        >>> # Use a brain mask to zero out non-brain voxels
        >>> transform = tio.Mask(masking_method="brain")
        >>> # Use a callable mask
        >>> transform = tio.Mask(masking_method=lambda x: x > 0)
        >>> # Keep only specific labels
        >>> transform = tio.Mask(masking_method="seg", labels=[1, 2])
    """

    def __init__(
        self,
        *,
        masking_method: str | Callable = "brain",
        outside_value: float = 0.0,
        labels: list[int] | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.masking_method = masking_method
        self.outside_value = outside_value
        self.labels = labels

    def make_params(self, batch: SubjectsBatch) -> dict[str, Any]:
        """No random parameters."""
        return {}

    def apply_transform(
        self,
        batch: SubjectsBatch,
        params: dict[str, Any],
    ) -> SubjectsBatch:
        """Apply each element's mask to that element's selected images."""
        masks = self._resolve_masks(batch)
        for _name, img_batch in self._get_images(batch).items():
            expanded = torch.stack(
                [
                    mask.expand_as(data)
                    for mask, data in zip(masks, img_batch.data, strict=True)
                ]
            )
            img_batch.data = torch.where(expanded, img_batch.data, self.outside_value)
        return batch

    def _resolve_masks(self, batch: SubjectsBatch) -> list[Tensor]:
        """Build one boolean mask per batch element."""
        if callable(self.masking_method) and not isinstance(self.masking_method, str):
            first_img = next(iter(self._get_images(batch).values()))
            return [self.masking_method(data).bool() for data in first_img.data]

        if isinstance(self.masking_method, str):
            key = self.masking_method
            if key not in batch.images:
                msg = (
                    f'Masking method "{key}" not found in batch images.'
                    f" Available: {list(batch.images.keys())}"
                )
                raise KeyError(msg)
            mask_batch = batch.images[key]
            if not issubclass(mask_batch._image_class, LabelMap):
                msg = f'Masking method "{key}" must refer to a LabelMap.'
                raise TypeError(msg)
            mask_data = mask_batch.data
            if self.labels is not None:
                mask = torch.zeros_like(mask_data, dtype=torch.bool)
                for label in self.labels:
                    mask = mask | (mask_data == label)
                return list(mask)
            return list(mask_data.bool())

        msg = (
            f"masking_method must be a str or callable, got {type(self.masking_method)}"
        )
        raise TypeError(msg)
