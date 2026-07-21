"""Schemas used to validate image and subject batches."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from .bboxes import BoundingBoxes
from .image import Image
from .points import Points
from .subject import Subject


@dataclass(frozen=True)
class _AnnotationSchema:
    """Describe one named annotation field."""

    value_type: type[Points] | type[BoundingBoxes]
    metadata_keys: tuple[str, ...]

    @classmethod
    def from_value(cls, value: Points | BoundingBoxes) -> _AnnotationSchema:
        """Build a schema from one annotation value."""
        return cls(type(value), tuple(value.metadata))

    def validate(
        self,
        value: Points | BoundingBoxes,
        *,
        index: int,
        context: str,
    ) -> None:
        """Validate one annotation against this schema."""
        if type(value) is not self.value_type:
            msg = (
                f"{context} at index {index} has type {type(value).__name__},"
                f" expected {self.value_type.__name__}"
            )
            raise ValueError(msg)
        _validate_keys(
            self.metadata_keys,
            value.metadata,
            index=index,
            context=f"{context} metadata",
        )


@dataclass(frozen=True)
class _ImageSchema:
    """Describe one named image field."""

    value_type: type[Image]
    shape: tuple[int, ...]
    dtype: str
    device: torch.device
    metadata_keys: tuple[str, ...]
    points: dict[str, _AnnotationSchema]
    bounding_boxes: dict[str, _AnnotationSchema]

    @classmethod
    def from_image(cls, image: Image) -> _ImageSchema:
        """Build a schema from one image."""
        return cls(
            value_type=type(image),
            shape=tuple(image.shape),
            dtype=_normalize_dtype(image.dtype),
            device=image.device,
            metadata_keys=tuple(image.metadata),
            points={
                name: _AnnotationSchema.from_value(value)
                for name, value in image.points.items()
            },
            bounding_boxes={
                name: _AnnotationSchema.from_value(value)
                for name, value in image.bounding_boxes.items()
            },
        )

    def validate(self, image: Image, *, index: int, name: str) -> None:
        """Validate one image against this schema."""
        context = f"Image {name!r}"
        if type(image) is not self.value_type:
            msg = (
                f"{context} at index {index} has type {type(image).__name__},"
                f" expected {self.value_type.__name__}"
            )
            raise ValueError(msg)
        for attribute in ("shape", "device"):
            expected = getattr(self, attribute)
            actual = getattr(image, attribute)
            if actual != expected:
                msg = (
                    f"{context} at index {index} has {attribute} {actual},"
                    f" expected {expected}"
                )
                raise ValueError(msg)
        actual_dtype = _normalize_dtype(image.dtype)
        if actual_dtype != self.dtype:
            msg = (
                f"{context} at index {index} has dtype {actual_dtype},"
                f" expected {self.dtype}"
            )
            raise ValueError(msg)
        _validate_keys(
            self.metadata_keys,
            image.metadata,
            index=index,
            context=f"{context} metadata",
        )
        _validate_annotations(
            self.points,
            image.points,
            index=index,
            context=f"{context} points",
        )
        _validate_annotations(
            self.bounding_boxes,
            image.bounding_boxes,
            index=index,
            context=f"{context} bounding boxes",
        )


@dataclass(frozen=True)
class _SubjectSchema:
    """Describe the fields shared by all subjects in a batch."""

    images: dict[str, _ImageSchema]
    metadata_keys: tuple[str, ...]
    points: dict[str, _AnnotationSchema]
    bounding_boxes: dict[str, _AnnotationSchema]

    @classmethod
    def from_subject(cls, subject: Subject) -> _SubjectSchema:
        """Build a schema from the first subject in a batch."""
        return cls(
            images={
                name: _ImageSchema.from_image(image)
                for name, image in subject.images.items()
            },
            metadata_keys=tuple(subject.metadata),
            points={
                name: _AnnotationSchema.from_value(value)
                for name, value in subject.points.items()
            },
            bounding_boxes={
                name: _AnnotationSchema.from_value(value)
                for name, value in subject.bounding_boxes.items()
            },
        )

    def validate(self, subject: Subject, *, index: int) -> None:
        """Validate one subject against this schema."""
        _validate_keys(
            self.images, subject.images, index=index, context="Subject images"
        )
        _validate_keys(
            self.metadata_keys,
            subject.metadata,
            index=index,
            context="Subject metadata",
        )
        _validate_annotations(
            self.points,
            subject.points,
            index=index,
            context="Subject points",
        )
        _validate_annotations(
            self.bounding_boxes,
            subject.bounding_boxes,
            index=index,
            context="Subject bounding boxes",
        )
        for name, schema in self.images.items():
            schema.validate(subject.images[name], index=index, name=name)


def _validate_annotations(
    reference: dict[str, _AnnotationSchema],
    current: dict[str, Points] | dict[str, BoundingBoxes],
    *,
    index: int,
    context: str,
) -> None:
    """Validate a named annotation store."""
    _validate_keys(reference, current, index=index, context=context)
    for name, schema in reference.items():
        schema.validate(current[name], index=index, context=f"{context} {name!r}")


def _validate_keys(
    reference: Any,
    current: Any,
    *,
    index: int,
    context: str,
) -> None:
    """Validate equivalent key sets while allowing reordered keys."""
    reference_set = set(reference)
    current_set = set(current)
    if reference_set == current_set:
        return
    missing = sorted(reference_set - current_set)
    unexpected = sorted(current_set - reference_set)
    msg = (
        f"{context} at index {index} has incompatible keys:"
        f" missing {missing}, unexpected {unexpected}"
    )
    raise ValueError(msg)


def _normalize_dtype(dtype: Any) -> str:
    """Return one comparable dtype name for Torch and NumPy dtypes."""
    return str(dtype).removeprefix("torch.")
