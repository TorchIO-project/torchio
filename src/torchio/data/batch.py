"""Batch containers for stacked images and subjects."""

from __future__ import annotations

from typing import Any

import torch
from torch import Tensor
from typing_extensions import Self

from .affine import AffineMatrix
from .image import Image
from .image import ScalarImage
from .invertible import Invertible


class ImagesBatch(Invertible):
    """A batch of images with per-sample affines.

    Wraps a 5D tensor ``(B, C, I, J, K)`` and a list of ``AffineMatrix``
    matrices (one per sample). Created by stacking multiple ``Image``
    objects or directly from a 5D tensor.

    Args:
        data: 5D tensor with shape ``(B, C, I, J, K)``.
        affines: List of affine matrices, one per sample.
        image_class: The ``Image`` subclass to use when unbatching.
    """

    def __init__(
        self,
        data: Tensor,
        affines: list[AffineMatrix],
        *,
        image_class: type[Image] = ScalarImage,
    ) -> None:
        if data.ndim != 5:
            msg = f"Expected 5D tensor (B, C, I, J, K), got {data.ndim}D"
            raise ValueError(msg)
        if len(affines) != data.shape[0]:
            msg = f"Expected {data.shape[0]} affines, got {len(affines)}"
            raise ValueError(msg)
        self._data = data
        self._affines = affines
        self._image_class = image_class
        self.applied_transforms: list[Any] = []

    @classmethod
    def from_images(cls, images: list[Image]) -> Self:
        """Stack a list of images into a batch.

        All images must have the same shape.

        Args:
            images: List of ``Image`` instances to stack.
        """
        if not images:
            msg = "Cannot create batch from empty list"
            raise ValueError(msg)
        tensors = [img.data for img in images]
        stacked = torch.stack(tensors)
        affines = [img.affine.clone() for img in images]
        image_class = type(images[0])
        return cls(stacked, affines, image_class=image_class)

    @property
    def data(self) -> Tensor:
        """5D tensor with shape ``(B, C, I, J, K)``."""
        return self._data

    @data.setter
    def data(self, value: Tensor) -> None:
        if value.ndim != 5:
            msg = f"Expected 5D tensor, got {value.ndim}D"
            raise ValueError(msg)
        self._data = value

    @property
    def affines(self) -> list[AffineMatrix]:
        """List of affine matrices, one per sample."""
        return self._affines

    @property
    def batch_size(self) -> int:
        """Number of samples in the batch."""
        return self._data.shape[0]

    @property
    def device(self) -> torch.device:
        """Device the batch data resides on."""
        return self._data.device

    def to(self, *args: Any, **kwargs: Any) -> Self:
        """Move batch data to a device and/or cast dtype."""
        self._data = self._data.to(*args, **kwargs)
        for affine in self._affines:
            affine.to(*args, **kwargs)
        return self

    def __getitem__(self, index: int) -> Image:
        """Get a single image from the batch by index."""
        return self._image_class(
            self._data[index],
            affine=self._affines[index].clone(),
        )

    def __len__(self) -> int:
        return self.batch_size

    def unbatch(self) -> list[Image]:
        """Split the batch into individual images."""
        return [self[i] for i in range(self.batch_size)]

    def __repr__(self) -> str:
        b, c, i, j, k = self._data.shape
        cls = self._image_class.__name__
        return f"ImagesBatch({cls}, batch_size={b}, shape=({c}, {i}, {j}, {k}))"


class SubjectsBatch(Invertible):
    """A batch of subjects with stacked image data.

    Each named image entry becomes an ``ImagesBatch``. Metadata is
    stored as lists (one value per sample).

    Created by ``SubjectsLoader`` or ``SubjectsBatch.from_subjects()``.
    """

    def __init__(
        self,
        images: dict[str, ImagesBatch],
        *,
        metadata: dict[str, list[Any]] | None = None,
    ) -> None:
        self._images = images
        self._metadata: dict[str, list[Any]] = metadata or {}
        self.applied_transforms: list[Any] = []

    @classmethod
    def from_subjects(cls, subjects: list[Any]) -> Self:
        """Stack a list of subjects into a batch.

        Args:
            subjects: List of ``Subject`` instances.
        """
        from .subject import Subject

        if not subjects:
            msg = "Cannot create batch from empty list"
            raise ValueError(msg)

        # Collect image names and types from the first subject
        first: Subject = subjects[0]
        image_names = list(first.images.keys())

        # Stack images
        images: dict[str, ImagesBatch] = {}
        for name in image_names:
            img_list = [sub.images[name] for sub in subjects]
            images[name] = ImagesBatch.from_images(img_list)

        # Collect metadata (non-image, non-annotation entries)
        metadata: dict[str, list[Any]] = {}
        for key in first.metadata:
            metadata[key] = [sub.metadata[key] for sub in subjects]

        return cls(images, metadata=metadata)

    @property
    def batch_size(self) -> int:
        """Number of samples in the batch."""
        first = next(iter(self._images.values()))
        return first.batch_size

    @property
    def images(self) -> dict[str, ImagesBatch]:
        """Dict of named image batches."""
        return self._images

    @property
    def metadata(self) -> dict[str, list[Any]]:
        """Metadata lists (one value per sample)."""
        return self._metadata

    @property
    def device(self) -> torch.device:
        """Device of the batch data."""
        first = next(iter(self._images.values()))
        return first.device

    def to(self, *args: Any, **kwargs: Any) -> Self:
        """Move all data to a device and/or cast dtype."""
        for batch in self._images.values():
            batch.to(*args, **kwargs)
        return self

    def __getitem__(self, key: str) -> ImagesBatch:
        """Get a named image batch."""
        return self._images[key]

    def __getattr__(self, name: str) -> ImagesBatch:
        """Attribute-style access to image batches."""
        if name.startswith("_"):
            raise AttributeError(name)
        if name in self._images:
            return self._images[name]
        msg = f"SubjectsBatch has no attribute {name!r}"
        raise AttributeError(msg)

    def unbatch(self) -> list[Any]:
        """Split the batch back into individual Subjects."""
        from .subject import Subject

        n = self.batch_size
        subjects = []
        for i in range(n):
            kwargs: dict[str, Any] = {}
            for name, img_batch in self._images.items():
                kwargs[name] = img_batch[i]
            for key, values in self._metadata.items():
                kwargs[key] = values[i]
            sub = Subject(**kwargs)
            sub.applied_transforms = list(self.applied_transforms)
            subjects.append(sub)
        return subjects

    def __len__(self) -> int:
        return self.batch_size

    def __repr__(self) -> str:
        names = ", ".join(self._images.keys())
        return f"SubjectsBatch(batch_size={self.batch_size}, images=[{names}])"


# Alias for radiology users — see Subject/Study note in subject.py.
StudiesBatch = SubjectsBatch
