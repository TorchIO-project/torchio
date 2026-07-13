"""Batch containers for stacked images and subjects."""

from __future__ import annotations

import copy as _copy
import dataclasses
from collections.abc import Callable
from collections.abc import Sequence
from typing import TYPE_CHECKING
from typing import Any

import torch
from torch import Tensor
from typing_extensions import Self

from .affine import AffineMatrix
from .bboxes import BoundingBoxes
from .image import Image
from .image import ScalarImage
from .invertible import Invertible
from .points import Points

if TYPE_CHECKING:
    from .subject import Subject

#: Reserved param keys used for per-instance history bookkeeping.
_BATCH_META_KEYS = ("_batch_size", "_batched_keys", "_keep")


class ImagesBatch(Invertible):
    """A batch of images with per-sample affines.

    Wraps a 5D tensor `(B, C, I, J, K)` and a list of `AffineMatrix`
    matrices (one per sample). Created by stacking multiple `Image`
    objects or directly from a 5D tensor.

    Args:
        data: 5D tensor with shape `(B, C, I, J, K)`.
        affines: List of affine matrices, one per sample.
        image_class: The `Image` subclass to use when unbatching.
        image_templates: Optional per-element image prototypes used to
            preserve subclasses, metadata, annotations, and other image
            payload when unbatching. The sequence length must match the
            batch size. If `None`, images are reconstructed from
            `image_class`.
    """

    def __init__(
        self,
        data: Tensor,
        affines: list[AffineMatrix],
        *,
        image_class: type[Image] = ScalarImage,
        image_templates: Sequence[Image] | None = None,
    ) -> None:
        if data.ndim != 5:
            msg = f"Expected 5D tensor (B, C, I, J, K), got {data.ndim}D"
            raise ValueError(msg)
        if data.shape[0] == 0:
            msg = "Cannot create an empty image batch"
            raise ValueError(msg)
        if len(affines) != data.shape[0]:
            msg = f"Expected {data.shape[0]} affines, got {len(affines)}"
            raise ValueError(msg)
        if not isinstance(image_class, type) or not issubclass(image_class, Image):
            msg = f"Expected an Image subclass, got {image_class!r}"
            raise TypeError(msg)
        if image_templates is not None and len(image_templates) != data.shape[0]:
            msg = (
                f"Expected {data.shape[0]} image templates, got {len(image_templates)}"
            )
            raise ValueError(msg)
        self._data = data
        self._affines = affines
        self._image_class = image_class
        self._image_templates = (
            list(image_templates) if image_templates is not None else None
        )
        self.applied_transforms: list[Any] = []
        self._per_element_history: list[list[Any]] | None = None

    @classmethod
    def from_images(cls, images: Sequence[Image]) -> Self:
        """Stack a list of images into a batch.

        All images must have the same class, shape, dtype, device, and
        nested metadata/annotation schema.

        Args:
            images: Images to stack.
        """
        if not images:
            msg = "Cannot create batch from empty list"
            raise ValueError(msg)
        _validate_images(images)
        tensors = [img.data for img in images]
        stacked = torch.stack(tensors)
        affines = [img.affine.clone() for img in images]
        image_class = type(images[0])
        templates = [_make_image_template(image) for image in images]
        batch = cls(
            stacked,
            affines,
            image_class=image_class,
            image_templates=templates,
        )
        _assign_histories(
            batch,
            [list(image.applied_transforms) for image in images],
        )
        return batch

    @property
    def data(self) -> Tensor:
        """5D tensor with shape `(B, C, I, J, K)`."""
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
        """Move batch data to a device and/or cast dtype.

        Args:
            *args: Positional arguments forwarded to `torch.Tensor.to`.
            **kwargs: Keyword arguments forwarded to `torch.Tensor.to`.

        Returns:
            `self` (modified in-place).
        """
        self._data = self._data.to(*args, **kwargs)
        for affine in self._affines:
            affine.to(*args, **kwargs)
        if self._image_templates is not None:
            for template in self._image_templates:
                template.to(*args, **kwargs)
        return self

    def __getitem__(self, index: int) -> Image:
        """Get a single image from the batch by index.

        Args:
            index: Batch element index.

        Returns:
            The reconstructed image for the selected element.
        """
        if self._image_templates is None:
            image = self._image_class(
                self._data[index],
                affine=self._affines[index].clone(),
            )
        else:
            template = self._image_templates[index]
            image = template.new_like(
                data=self._data[index],
                affine=self._affines[index].clone(),
            )
            image._metadata = _copy.deepcopy(template.metadata)
        image.applied_transforms = _get_element_history(self, index)
        return image

    def __len__(self) -> int:
        return self.batch_size

    def unbatch(self) -> list[Image]:
        """Split the batch into individual images."""
        return [self[i] for i in range(self.batch_size)]

    @property
    def has_annotations(self) -> bool:
        """Whether any image has attached points or bounding boxes."""
        if self._image_templates is None:
            return False
        return any(
            template.points or template.bounding_boxes
            for template in self._image_templates
        )

    def set_per_element_history(self, histories: list[list[Any]]) -> None:
        """Freeze a distinct transform history for each batch element.

        Args:
            histories: One transform-history list per batch element.
        """
        if len(histories) != self.batch_size:
            msg = (
                f"Expected {self.batch_size} per-element histories,"
                f" got {len(histories)}"
            )
            raise ValueError(msg)
        self._per_element_history = [list(history) for history in histories]
        self.applied_transforms = []

    def clear_history(self) -> None:
        """Remove all applied transform records."""
        self.applied_transforms = []
        self._per_element_history = None

    def get_inverse_transform(self, **kwargs: Any) -> Any:
        """Build a transform that inverts the recorded history.

        Args:
            **kwargs: Forwarded to `Invertible.get_inverse_transform`.
        """
        if self._per_element_history is not None:
            msg = (
                "This image batch has per-element transform histories, so a"
                " single batch inverse is ambiguous. Call"
                " apply_inverse_transform() or unbatch() and invert each image."
            )
            raise RuntimeError(msg)
        return super().get_inverse_transform(**kwargs)

    def apply_inverse_transform(self, **kwargs: Any) -> ImagesBatch:
        """Apply the inverse of the recorded history.

        Args:
            **kwargs: Forwarded to `get_inverse_transform`.

        Returns:
            A batch with the transforms undone.
        """
        if self._per_element_history is not None:
            inverted = [image.apply_inverse_transform(**kwargs) for image in self]
            return type(self).from_images(inverted)
        return super().apply_inverse_transform(**kwargs)

    def __repr__(self) -> str:
        b, c, i, j, k = self._data.shape
        cls = self._image_class.__name__
        return f"ImagesBatch({cls}, batch_size={b}, shape=({c}, {i}, {j}, {k}))"


class SubjectsBatch(Invertible):
    """A batch of subjects with stacked image data.

    Each named image entry becomes an `ImagesBatch`. Metadata is
    stored as lists (one value per sample).

    Created by `SubjectsLoader` or `SubjectsBatch.from_subjects()`.

    Args:
        images: Named image batches. May be empty for metadata-only or
            annotation-only batches.
        points: Named subject-level point sets, stored as one `Points`
            instance per batch element.
        bounding_boxes: Named subject-level bounding boxes, stored as one
            `BoundingBoxes` instance per batch element.
        metadata: Named metadata values, stored as one value per batch
            element.

    All supplied batched fields must contain the same non-zero number of
    elements.
    """

    def __init__(
        self,
        images: dict[str, ImagesBatch] | None = None,
        *,
        points: dict[str, list[Points]] | None = None,
        bounding_boxes: dict[str, list[BoundingBoxes]] | None = None,
        metadata: dict[str, list[Any]] | None = None,
    ) -> None:
        self._images = dict(images or {})
        self._points = dict(points or {})
        self._bounding_boxes = dict(bounding_boxes or {})
        self._metadata = dict(metadata or {})
        self._batch_size = _resolve_batch_size(
            self._images,
            self._points,
            self._bounding_boxes,
            self._metadata,
        )
        self.applied_transforms: list[Any] = []
        # When per-element branching occurs (e.g. per-instance OneOf),
        # this stores the frozen per-element history prefix. Transforms
        # applied afterwards still append to `applied_transforms`, and
        # `unbatch()` merges the prefix with the sliced suffix.
        self._per_element_history: list[list[Any]] | None = None

    def set_per_element_history(self, histories: list[list[Any]]) -> None:
        """Freeze a distinct transform history for each batch element.

        Used when different elements receive different transforms (for
        example per-instance [`OneOf`][torchio.OneOf]). Resets the shared
        `applied_transforms` so that subsequent transforms accumulate as
        a common suffix.

        Args:
            histories: One history list per batch element.
        """
        if len(histories) != self.batch_size:
            msg = (
                f"Expected {self.batch_size} per-element histories,"
                f" got {len(histories)}"
            )
            raise ValueError(msg)
        self._per_element_history = [list(history) for history in histories]
        self.applied_transforms = []

    @classmethod
    def from_subjects(cls, subjects: Sequence[Any]) -> Self:
        """Stack a list of subjects into a batch.

        Args:
            subjects: `Subject` instances to stack.
        """
        first = _validate_subject_inputs(subjects)
        _validate_subject_schemas(subjects)

        batch = cls(
            _stack_subject_images(subjects, first),
            points=_collect_subject_points(subjects, first),
            bounding_boxes=_collect_subject_boxes(subjects, first),
            metadata=_collect_subject_metadata(subjects, first),
        )
        _assign_histories(
            batch,
            [list(subject.applied_transforms) for subject in subjects],
        )
        return batch

    @property
    def batch_size(self) -> int:
        """Number of samples in the batch."""
        return self._batch_size

    @property
    def images(self) -> dict[str, ImagesBatch]:
        """Dict of named image batches."""
        return self._images

    @property
    def points(self) -> dict[str, list[Points]]:
        """Dict of named point sets, one list entry per sample."""
        return self._points

    @property
    def bounding_boxes(self) -> dict[str, list[BoundingBoxes]]:
        """Dict of named bounding boxes, one list entry per sample."""
        return self._bounding_boxes

    @property
    def metadata(self) -> dict[str, list[Any]]:
        """Metadata lists (one value per sample)."""
        return self._metadata

    @property
    def has_annotations(self) -> bool:
        """Whether the batch contains subject- or image-level annotations."""
        return bool(
            self._points
            or self._bounding_boxes
            or any(image_batch.has_annotations for image_batch in self._images.values())
        )

    @property
    def device(self) -> torch.device:
        """Device of the batch data."""
        devices = [image.device for image in self._images.values()]
        devices.extend(
            points.device for values in self._points.values() for points in values
        )
        devices.extend(
            boxes.device for values in self._bounding_boxes.values() for boxes in values
        )
        if not devices:
            return torch.device("cpu")
        reference = devices[0]
        if any(device != reference for device in devices[1:]):
            msg = f"Inconsistent devices in SubjectsBatch: {devices}"
            raise RuntimeError(msg)
        return reference

    def to(self, *args: Any, **kwargs: Any) -> Self:
        """Move all data to a device and/or cast dtype.

        Args:
            *args: Positional arguments forwarded to each field's `to`
                method.
            **kwargs: Keyword arguments forwarded to each field's `to`
                method.

        Returns:
            `self` (modified in-place).
        """
        for batch in self._images.values():
            batch.to(*args, **kwargs)
        for values in self._points.values():
            for points in values:
                points.to(*args, **kwargs)
        for values in self._bounding_boxes.values():
            for boxes in values:
                boxes.to(*args, **kwargs)
        return self

    def __getitem__(self, key: str) -> Any:
        """Get a named batched field.

        Args:
            key: Image, point, bounding-box, or metadata field name.

        Returns:
            The corresponding batched field.
        """
        for store in (
            self._images,
            self._points,
            self._bounding_boxes,
            self._metadata,
        ):
            if key in store:
                return store[key]
        raise KeyError(key)

    def __getattr__(self, name: str) -> Any:
        """Access a named batched field as an attribute.

        Args:
            name: Image, point, bounding-box, or metadata field name.

        Returns:
            The corresponding batched field.
        """
        if name.startswith("_"):
            raise AttributeError(name)
        for store in (
            self._images,
            self._points,
            self._bounding_boxes,
            self._metadata,
        ):
            if name in store:
                return store[name]
        msg = f"SubjectsBatch has no attribute {name!r}"
        raise AttributeError(msg)

    def unbatch(self) -> list[Any]:
        """Split the batch back into individual Subjects.

        Per-instance transform history is sliced so that each subject
        receives only its own sampled parameters; transforms that were
        gated out for an element (per-element probability) are omitted
        from that subject's history.
        """
        from .subject import Subject

        histories = _get_element_histories(self)
        subjects = []
        for i in range(self.batch_size):
            kwargs: dict[str, Any] = {}
            for name, img_batch in self._images.items():
                kwargs[name] = img_batch[i]
            for name, values in self._points.items():
                kwargs[name] = _copy.deepcopy(values[i])
            for name, values in self._bounding_boxes.items():
                kwargs[name] = _copy.deepcopy(values[i])
            for key, values in self._metadata.items():
                kwargs[key] = _copy.deepcopy(values[i])
            sub = Subject(**kwargs)
            sub.applied_transforms = histories[i]
            subjects.append(sub)
        return subjects

    def map_subjects(
        self,
        callback: Callable[[Subject], Subject],
        *,
        copy: bool = True,
    ) -> Self:
        """Apply a callback to every subject and rebuild the batch.

        Each callback receives an unbatched `Subject` carrying its complete
        transform history. All returned subjects must have a compatible
        schema and image shapes so they can be re-stacked. Callback-added
        histories are retained. By default, image tensors are cloned before
        the callback so the input batch is unchanged. With `copy=False`,
        callbacks may mutate the input batch's image tensors.

        Args:
            callback: Callable taking and returning one `Subject`.
            copy: Clone each image tensor before invoking the callback.
                Set to `False` when the caller has already handled copy
                semantics.

        Returns:
            A new batch containing the callback results.

        Raises:
            TypeError: If the callback does not return a `Subject`.
            ValueError: If callback results cannot be batched together.

        Examples:
            >>> def normalize_identifier(subject):
            ...     subject.metadata["identifier"] = subject.identifier.strip()
            ...     return subject
            >>> result = batch.map_subjects(normalize_identifier)
        """
        from .subject import Subject

        mapped = []
        for index, subject in enumerate(self.unbatch()):
            if copy:
                for image in subject.images.values():
                    image.set_data(image.data.clone())
            result = callback(subject)
            if not isinstance(result, Subject):
                msg = (
                    f"Expected callback result at index {index} to be a Subject,"
                    f" got {type(result).__name__}"
                )
                raise TypeError(msg)
            mapped.append(result)
        return type(self).from_subjects(mapped)

    def __len__(self) -> int:
        return self.batch_size

    def adopt_history(self, source: SubjectsBatch, subjects: list[Any]) -> None:
        """Carry transform history from *source* after rebuilding the batch.

        Used by code that unbatches, processes, and re-stacks subjects
        (for example the MONAI and Cornucopia adapters). Preserves a
        per-element history if *source* had one, otherwise copies the
        shared history.

        Args:
            source: The batch the subjects were unbatched from.
            subjects: The processed subjects, in batch order.
        """
        if len(subjects) != source.batch_size:
            msg = (
                f"Expected {source.batch_size} subjects when adopting history,"
                f" got {len(subjects)}"
            )
            raise ValueError(msg)
        _assign_histories(
            self,
            [list(subject.applied_transforms) for subject in subjects],
        )

    def clear_history(self) -> None:
        """Remove all applied transform records, including per-element ones."""
        self.applied_transforms = []
        self._per_element_history = None

    def get_inverse_transform(self, **kwargs: Any) -> Any:
        """Build a transform that inverts the recorded history.

        Args:
            **kwargs: Forwarded to `Invertible.get_inverse_transform`.

        Raises:
            RuntimeError: If the batch carries per-element histories (from
                a per-instance `OneOf`/`SomeOf`), since a single batch
                inverse is ambiguous. Call `apply_inverse_transform`
                (which inverts each element) or `unbatch()` and invert
                each subject.
        """
        if self._per_element_history is not None:
            msg = (
                "This batch has per-element transform histories from a"
                " per-instance OneOf/SomeOf, so a single batch inverse is"
                " ambiguous. Call apply_inverse_transform() (which inverts"
                " each element) or unbatch() and invert each subject."
            )
            raise RuntimeError(msg)
        return super().get_inverse_transform(**kwargs)

    def apply_inverse_transform(self, **kwargs: Any) -> SubjectsBatch:
        """Apply the inverse of the recorded history.

        When the batch carries per-element histories, each element is
        inverted independently and the results are re-stacked.

        Args:
            **kwargs: Forwarded to `get_inverse_transform`.

        Returns:
            A batch with the transforms undone.
        """
        if self._per_element_history is not None:
            inverted = [s.apply_inverse_transform(**kwargs) for s in self.unbatch()]
            return type(self).from_subjects(inverted)
        return super().apply_inverse_transform(**kwargs)

    def __repr__(self) -> str:
        fields = []
        for label, store in (
            ("images", self._images),
            ("points", self._points),
            ("bboxes", self._bounding_boxes),
            ("metadata", self._metadata),
        ):
            if store:
                fields.append(f"{label}=[{', '.join(store)}]")
        return f"SubjectsBatch(batch_size={self.batch_size}, {', '.join(fields)})"


# Alias for radiology users (see Subject/Study note in subject.py).
StudiesBatch = SubjectsBatch


def _validate_subject_inputs(subjects: Sequence[Any]) -> Subject:
    """Validate subject input types and return the first subject."""
    from .subject import Subject

    if not subjects:
        msg = "Cannot create batch from empty list"
        raise ValueError(msg)
    for index, subject in enumerate(subjects):
        if not isinstance(subject, Subject):
            msg = f"Expected Subject at index {index}, got {type(subject).__name__}"
            raise TypeError(msg)
    return subjects[0]


def _stack_subject_images(
    subjects: Sequence[Subject],
    first: Subject,
) -> dict[str, ImagesBatch]:
    """Stack each named image across subjects."""
    return {
        name: ImagesBatch.from_images([subject.images[name] for subject in subjects])
        for name in first.images
    }


def _collect_subject_points(
    subjects: Sequence[Subject],
    first: Subject,
) -> dict[str, list[Points]]:
    """Collect subject-level points without sharing mutable objects."""
    return {
        name: [_copy.deepcopy(subject.points[name]) for subject in subjects]
        for name in first.points
    }


def _collect_subject_boxes(
    subjects: Sequence[Subject],
    first: Subject,
) -> dict[str, list[BoundingBoxes]]:
    """Collect subject-level boxes without sharing mutable objects."""
    return {
        name: [_copy.deepcopy(subject.bounding_boxes[name]) for subject in subjects]
        for name in first.bounding_boxes
    }


def _collect_subject_metadata(
    subjects: Sequence[Subject],
    first: Subject,
) -> dict[str, list[Any]]:
    """Collect subject metadata without sharing mutable objects."""
    return {
        key: [_copy.deepcopy(subject.metadata[key]) for subject in subjects]
        for key in first.metadata
    }


def _make_image_template(image: Image) -> Image:
    """Create a lightweight image carrying the per-element payload."""
    channels = image.shape[0]
    data = torch.empty(
        channels,
        1,
        1,
        1,
        dtype=image.data.dtype,
        device=image.data.device,
    )
    template = image.new_like(data=data)
    template._metadata = _copy.deepcopy(image.metadata)
    template.applied_transforms = list(image.applied_transforms)
    return template


def _validate_images(
    images: Sequence[Image],
    *,
    image_name: str | None = None,
) -> None:
    """Validate image compatibility before stacking."""
    first = images[0]
    context = "Image" if image_name is None else f"Image {image_name!r}"
    reference_metadata = first.metadata.keys()
    reference_points = first.points
    reference_boxes = first.bounding_boxes
    for index, image in enumerate(images[1:], 1):
        if type(image) is not type(first):
            msg = (
                f"{context} at index {index} is incompatible:"
                f" expected {type(first).__name__}, got {type(image).__name__}"
            )
            raise ValueError(msg)
        for attribute in ("shape", "dtype", "device"):
            expected = getattr(first, attribute)
            actual = getattr(image, attribute)
            if actual != expected:
                msg = (
                    f"{context} at index {index} has {attribute} {actual},"
                    f" expected {expected}"
                )
                raise ValueError(msg)
        _validate_keys(
            reference_metadata,
            image.metadata.keys(),
            index=index,
            field="metadata",
            context=context.lower(),
        )
        _validate_annotation_store(
            reference_points,
            image.points,
            index=index,
            field="point",
            context=context.lower(),
        )
        _validate_annotation_store(
            reference_boxes,
            image.bounding_boxes,
            index=index,
            field="bounding box",
            context=context.lower(),
        )


def _validate_subject_schemas(subjects: Sequence[Any]) -> None:
    """Validate all subject and nested image schemas."""
    first = subjects[0]
    reference_stores = (
        ("image", first.images),
        ("metadata", first.metadata),
        ("point", first.points),
        ("bounding box", first.bounding_boxes),
    )
    for index, subject in enumerate(subjects[1:], 1):
        current_stores = (
            subject.images,
            subject.metadata,
            subject.points,
            subject.bounding_boxes,
        )
        for (field, reference), current in zip(
            reference_stores,
            current_stores,
            strict=True,
        ):
            _validate_keys(
                reference.keys(),
                current.keys(),
                index=index,
                field=field,
                context="Subject",
            )
        for name in first.points:
            _validate_annotation(
                first.points[name],
                subject.points[name],
                index=index,
                field="point",
                context="Subject",
                name=name,
            )
        for name in first.bounding_boxes:
            _validate_annotation(
                first.bounding_boxes[name],
                subject.bounding_boxes[name],
                index=index,
                field="bounding box",
                context="Subject",
                name=name,
            )
    for name in first.images:
        _validate_images(
            [subject.images[name] for subject in subjects],
            image_name=name,
        )


def _validate_annotation_store(
    reference: dict[str, Any],
    current: dict[str, Any],
    *,
    index: int,
    field: str,
    context: str,
) -> None:
    """Validate a named annotation store."""
    _validate_keys(
        reference.keys(),
        current.keys(),
        index=index,
        field=field,
        context=context,
    )
    for name in reference:
        _validate_annotation(
            reference[name],
            current[name],
            index=index,
            field=field,
            context=context,
            name=name,
        )


def _validate_annotation(
    reference: Points | BoundingBoxes,
    current: Points | BoundingBoxes,
    *,
    index: int,
    field: str,
    context: str,
    name: str,
) -> None:
    """Validate one named annotation's type and metadata schema."""
    if type(current) is not type(reference):
        msg = (
            f"{context} at index {index} has incompatible {field} {name!r}:"
            f" expected {type(reference).__name__}, got {type(current).__name__}"
        )
        raise ValueError(msg)
    _validate_keys(
        reference.metadata.keys(),
        current.metadata.keys(),
        index=index,
        field=f"{field} {name!r} metadata",
        context=context,
    )


def _validate_keys(
    reference: Any,
    current: Any,
    *,
    index: int,
    field: str,
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
        f"{context} at index {index} has incompatible {field} keys:"
        f" missing {missing}, unexpected {unexpected}"
    )
    raise ValueError(msg)


def _resolve_batch_size(
    images: dict[str, ImagesBatch],
    points: dict[str, list[Points]],
    bounding_boxes: dict[str, list[BoundingBoxes]],
    metadata: dict[str, list[Any]],
) -> int:
    """Resolve and validate the shared length of every batched field."""
    sizes: list[tuple[str, int]] = []
    sizes.extend(
        (f"image {name!r}", image_batch.batch_size)
        for name, image_batch in images.items()
    )
    sizes.extend((f"point {name!r}", len(values)) for name, values in points.items())
    sizes.extend(
        (f"bounding box {name!r}", len(values))
        for name, values in bounding_boxes.items()
    )
    sizes.extend(
        (f"metadata {name!r}", len(values)) for name, values in metadata.items()
    )
    if not sizes:
        msg = "A SubjectsBatch must contain at least one batched field"
        raise ValueError(msg)
    reference_name, reference_size = sizes[0]
    if reference_size == 0:
        msg = "Cannot create an empty SubjectsBatch"
        raise ValueError(msg)
    for name, size in sizes[1:]:
        if size != reference_size:
            msg = (
                f"Inconsistent batch size: {reference_name} has"
                f" {reference_size} elements, but {name} has {size}"
            )
            raise ValueError(msg)
    return reference_size


def _histories_equal(first: list[Any], second: list[Any]) -> bool:
    """Compare histories, conservatively treating ambiguous equality as false."""
    try:
        return bool(first == second)
    except (RuntimeError, TypeError, ValueError):
        return False


def _assign_histories(batch: Any, histories: Sequence[list[Any]]) -> None:
    """Store histories in shared or per-element form."""
    copied = [list(history) for history in histories]
    if not copied:
        batch.applied_transforms = []
        batch._per_element_history = None
        return
    first = copied[0]
    if all(_histories_equal(first, history) for history in copied[1:]):
        batch.applied_transforms = list(first)
        batch._per_element_history = None
    else:
        batch.set_per_element_history(copied)


def _get_element_histories(batch: Any) -> list[list[Any]]:
    """Return each element's complete history."""
    return [_get_element_history(batch, index) for index in range(batch.batch_size)]


def _get_element_history(batch: Any, index: int) -> list[Any]:
    """Return one element's complete history."""
    suffix = _slice_history(batch.applied_transforms, index)
    if batch._per_element_history is None:
        return suffix
    return list(batch._per_element_history[index]) + suffix


def _slice_params(
    params: dict[str, Any],
    index: int,
    batched_keys: list[str],
) -> dict[str, Any]:
    """Slice a per-instance params dict down to a single element.

    Args:
        params: The batch-level parameter dict.
        index: The batch element to extract.
        batched_keys: Names of the params that hold one value per
            element.

    Returns:
        A new params dict with per-element values resolved and the
        internal bookkeeping keys removed.
    """
    sliced: dict[str, Any] = {}
    for key, value in params.items():
        if key in _BATCH_META_KEYS:
            continue
        if key in batched_keys and isinstance(value, list):
            sliced[key] = value[index]
        else:
            sliced[key] = value
    return sliced


def _slice_history(history: list[Any], index: int) -> list[Any]:
    """Build the per-subject transform history for batch element *index*.

    Batch-shared traces are copied unchanged. Per-instance traces are
    sliced to the element's own parameters, and traces whose per-element
    keep mask excludes this element are dropped.

    Args:
        history: The batch-level list of `AppliedTransform` records.
        index: The batch element whose history to build.

    Returns:
        The list of `AppliedTransform` records for the element.
    """
    sliced: list[Any] = []
    for trace in history:
        params = getattr(trace, "params", None)
        if not isinstance(params, dict) or "_batched_keys" not in params:
            sliced.append(trace)
            continue
        expected_size = params.get("_batch_size")
        if expected_size is not None and not 0 <= index < expected_size:
            msg = (
                f"Cannot extract per-instance history for element {index}:"
                f" the transform was recorded for a batch of size"
                f" {expected_size}"
            )
            raise IndexError(msg)
        keep = params.get("_keep")
        if keep is not None and not keep[index]:
            continue
        batched_keys = params["_batched_keys"]
        new_params = _slice_params(params, index, batched_keys)
        sliced.append(dataclasses.replace(trace, params=new_params))
    return sliced
