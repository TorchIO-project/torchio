"""Batch containers for stacked images and subjects."""

from __future__ import annotations

import copy as _copy
from collections.abc import Callable
from collections.abc import Sequence
from typing import TYPE_CHECKING
from typing import Any

import torch
from torch import Tensor
from typing_extensions import Self

from .affine import AffineMatrix
from .batch_history import _BatchedHistoryMixin
from .batch_schema import _ImageSchema
from .batch_schema import _SubjectSchema
from .bboxes import BoundingBoxes
from .image import Image
from .image import LabelMap
from .image import ScalarImage
from .points import Points

if TYPE_CHECKING:
    from .subject import Subject

#: Reserved param keys used for per-instance history bookkeeping.
_BATCH_META_KEYS = ("_batch_size", "_batched_keys", "_keep")


class ImagesBatch(_BatchedHistoryMixin):
    """A batch of images with per-sample affines and private prototypes.

    Wraps a 5D tensor `(B, C, I, J, K)` and a list of `AffineMatrix`
    matrices (one per sample). Use `from_images()` for lossless image
    round-trips or `from_tensor()` for an existing 5D tensor.

    Args:
        data: 5D tensor with shape `(B, C, I, J, K)`.
        affines: Affine matrices, one per sample.
        image_class: The `Image` subclass to use when unbatching.
    """

    def __init__(
        self,
        data: Tensor,
        affines: Sequence[AffineMatrix],
        *,
        image_class: type[Image] = ScalarImage,
    ) -> None:
        prototypes = _make_prototypes_from_class(data, image_class)
        self._initialize(data, affines, prototypes)

    def _initialize(
        self,
        data: Tensor,
        affines: Sequence[AffineMatrix],
        prototypes: Sequence[Image],
        histories: Sequence[Sequence[Any]] | None = None,
    ) -> None:
        """Initialize a validated image batch."""
        if data.ndim != 5:
            msg = f"Expected 5D tensor (B, C, I, J, K), got {data.ndim}D"
            raise ValueError(msg)
        if data.shape[0] == 0:
            msg = "Cannot create an empty image batch"
            raise ValueError(msg)
        if len(affines) != data.shape[0]:
            msg = f"Expected {data.shape[0]} affines, got {len(affines)}"
            raise ValueError(msg)
        if len(prototypes) != data.shape[0]:
            msg = f"Expected {data.shape[0]} prototypes, got {len(prototypes)}"
            raise ValueError(msg)
        self._data = data
        self._affines = [affine.clone() for affine in affines]
        self._prototypes = list(prototypes)
        self._initialize_histories(data.shape[0], histories)

    @classmethod
    def _from_parts(
        cls,
        data: Tensor,
        affines: Sequence[AffineMatrix],
        prototypes: Sequence[Image],
        histories: Sequence[Sequence[Any]] | None = None,
    ) -> Self:
        """Build an image batch from validated internal parts."""
        batch = cls.__new__(cls)
        batch._initialize(data, affines, prototypes, histories)
        return batch

    @classmethod
    def from_tensor(
        cls,
        data: Tensor,
        affines: Sequence[AffineMatrix] | None = None,
        *,
        image_class: type[Image] = ScalarImage,
    ) -> Self:
        """Build an image batch from a 5D tensor.

        Args:
            data: 5D tensor with shape `(B, C, I, J, K)`.
            affines: Optional affine matrices, one per element. Identity
                matrices are used when omitted.
            image_class: Image class used to synthesize private prototypes.

        Returns:
            A new image batch.
        """
        if data.ndim != 5:
            msg = f"Expected 5D tensor (B, C, I, J, K), got {data.ndim}D"
            raise ValueError(msg)
        resolved_affines = (
            [AffineMatrix().to(data.device) for _ in range(data.shape[0])]
            if affines is None
            else affines
        )
        return cls(data, resolved_affines, image_class=image_class)

    @classmethod
    def from_images(cls, images: Sequence[Image]) -> Self:
        """Stack images into a lossless batch.

        All images must share the same schema, shape, dtype, and device.

        Args:
            images: Images to stack.

        Returns:
            A new image batch.
        """
        if not images:
            msg = "Cannot create batch from empty list"
            raise ValueError(msg)
        schema = _ImageSchema.from_image(images[0])
        for index, image in enumerate(images[1:], 1):
            schema.validate(image, index=index, name="image")
        tensors = [image.data for image in images]
        stacked = torch.stack(tensors)
        affines = [image.affine for image in images]
        prototypes = [_make_image_prototype(image) for image in images]
        histories = [image.applied_transforms for image in images]
        return cls._from_parts(stacked, affines, prototypes, histories)

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
    def image_class(self) -> type[Image]:
        """Image class shared by every batch element."""
        return type(self._prototypes[0])

    @property
    def is_label(self) -> bool:
        """Whether the batch contains label images."""
        return issubclass(self.image_class, LabelMap)

    @property
    def batch_size(self) -> int:
        """Number of samples in the batch."""
        return self._data.shape[0]

    @property
    def device(self) -> torch.device:
        """Device the batch data resides on."""
        return self._data.device

    def to(self, *args: Any, **kwargs: Any) -> Self:
        """Move batch data and payload to a device or dtype.

        Args:
            *args: Positional arguments forwarded to `torch.Tensor.to`.
            **kwargs: Keyword arguments forwarded to `torch.Tensor.to`.

        Returns:
            `self` (modified in-place).
        """
        self._data = self._data.to(*args, **kwargs)
        for affine in self._affines:
            affine.to(*args, **kwargs)
        for prototype in self._prototypes:
            prototype.to(*args, **kwargs)
        return self

    def __getitem__(self, index: int) -> Image:
        """Get one reconstructed image.

        Args:
            index: Batch element index.

        Returns:
            The reconstructed image.
        """
        prototype = self._prototypes[index]
        image = prototype.new_like(
            data=self._data[index],
            affine=self._affines[index].clone(),
        )
        image._metadata = _copy.deepcopy(prototype.metadata)
        image.applied_transforms = list(self.history(index))
        return image

    def __len__(self) -> int:
        return self.batch_size

    def unbatch(self) -> list[Image]:
        """Split the batch into individual images."""
        return [self[i] for i in range(self.batch_size)]

    def _batch_items(self, items: Sequence[Any]) -> Self:
        """Rebuild an image batch from images."""
        return type(self).from_images(items)

    @property
    def has_annotations(self) -> bool:
        """Whether any image prototype carries annotations."""
        return any(
            prototype.points or prototype.bounding_boxes
            for prototype in self._prototypes
        )

    def __repr__(self) -> str:
        b, c, i, j, k = self._data.shape
        cls = self.image_class.__name__
        return f"ImagesBatch({cls}, batch_size={b}, shape=({c}, {i}, {j}, {k}))"


class SubjectsBatch(_BatchedHistoryMixin):
    """A batch of image columns and per-element object stores.

    Each image field becomes an `ImagesBatch`. Metadata, points, and
    bounding boxes are stored as lists with one value per element.

    Created by `SubjectsLoader` or `SubjectsBatch.from_subjects()`.

    Args:
        images: Named image batches.
        points: Named subject-level point sets.
        bounding_boxes: Named subject-level bounding boxes.
        metadata: Named metadata values.
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
        self._schema: _SubjectSchema | None = None
        self._initialize_histories(self._batch_size)

    @classmethod
    def from_subjects(cls, subjects: Sequence[Any]) -> Self:
        """Stack subjects into a lossless batch.

        Args:
            subjects: Subjects to stack.

        Returns:
            A new subject batch.
        """
        schema = _validate_subjects(subjects)
        batch = cls(
            _stack_subject_images(subjects, schema),
            points=_collect_subject_points(subjects, schema),
            bounding_boxes=_collect_subject_boxes(subjects, schema),
            metadata=_collect_subject_metadata(subjects, schema),
        )
        batch._schema = schema
        batch._set_histories(
            [subject.applied_transforms for subject in subjects],
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
        """Subject-level point sets, one value per element."""
        return self._points

    @property
    def bounding_boxes(self) -> dict[str, list[BoundingBoxes]]:
        """Subject-level bounding boxes, one value per element."""
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
            or any(image.has_annotations for image in self._images.values())
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
        """Move all spatial data to a device or dtype.

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
            key: Field name.

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
        """Access a named batched field as an attribute."""
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
            sub.applied_transforms = list(self.history(i))
            subjects.append(sub)
        return subjects

    def map_subjects(
        self,
        callback: Callable[[Subject], Subject],
        *,
        copy: bool = True,
    ) -> Self:
        """Apply a callback to every subject and rebuild the batch.

        Each callback receives an unbatched `Subject` carrying its exact
        history. By default, image tensors are cloned before the callback
        so the input batch is unchanged. With `copy=False`, callbacks may
        mutate the input batch's image tensors.

        Args:
            callback: Callable taking and returning one `Subject`.
            copy: Clone each image tensor before invoking the callback.

        Returns:
            A new batch containing the callback results.

        Raises:
            TypeError: If the callback does not return a `Subject`.
            ValueError: If callback results cannot be batched together.
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

    def _batch_items(self, items: Sequence[Any]) -> Self:
        """Rebuild a subject batch from subjects."""
        return type(self).from_subjects(items)

    def __len__(self) -> int:
        return self.batch_size

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


def _validate_subjects(
    subjects: Sequence[Any],
) -> _SubjectSchema:
    """Validate subject inputs and return their shared schema."""
    from .subject import Subject

    if not subjects:
        msg = "Cannot create batch from empty list"
        raise ValueError(msg)
    for index, subject in enumerate(subjects):
        if not isinstance(subject, Subject):
            msg = f"Expected Subject at index {index}, got {type(subject).__name__}"
            raise TypeError(msg)
    first = subjects[0]
    schema = _SubjectSchema.from_subject(first)
    for index, subject in enumerate(subjects[1:], 1):
        schema.validate(subject, index=index)
    return schema


def _stack_subject_images(
    subjects: Sequence[Subject],
    schema: _SubjectSchema,
) -> dict[str, ImagesBatch]:
    """Stack each image field across subjects."""
    return {
        name: ImagesBatch.from_images([subject.images[name] for subject in subjects])
        for name in schema.images
    }


def _collect_subject_points(
    subjects: Sequence[Subject],
    schema: _SubjectSchema,
) -> dict[str, list[Points]]:
    """Collect independent subject-level point values."""
    return {
        name: [_copy.deepcopy(subject.points[name]) for subject in subjects]
        for name in schema.points
    }


def _collect_subject_boxes(
    subjects: Sequence[Subject],
    schema: _SubjectSchema,
) -> dict[str, list[BoundingBoxes]]:
    """Collect independent subject-level bounding boxes."""
    return {
        name: [_copy.deepcopy(subject.bounding_boxes[name]) for subject in subjects]
        for name in schema.bounding_boxes
    }


def _collect_subject_metadata(
    subjects: Sequence[Subject],
    schema: _SubjectSchema,
) -> dict[str, list[Any]]:
    """Collect independent subject metadata values."""
    return {
        key: [_copy.deepcopy(subject.metadata[key]) for subject in subjects]
        for key in schema.metadata_keys
    }


def _make_prototypes_from_class(
    data: Tensor,
    image_class: type[Image],
) -> list[Image]:
    """Create minimal private prototypes for an existing tensor batch."""
    if data.ndim != 5:
        msg = f"Expected 5D tensor (B, C, I, J, K), got {data.ndim}D"
        raise ValueError(msg)
    if not isinstance(image_class, type) or not issubclass(image_class, Image):
        msg = f"Expected an Image subclass, got {image_class!r}"
        raise TypeError(msg)
    channels = data.shape[1]
    return [
        image_class(
            torch.empty(
                channels,
                1,
                1,
                1,
                dtype=data.dtype,
                device=data.device,
            )
        )
        for _ in range(data.shape[0])
    ]


def _make_image_prototype(image: Image) -> Image:
    """Create a lightweight prototype preserving image payload."""
    data = torch.empty(
        image.shape[0],
        1,
        1,
        1,
        dtype=image.data.dtype,
        device=image.data.device,
    )
    prototype = image.new_like(data=data)
    prototype._metadata = _copy.deepcopy(image.metadata)
    prototype.applied_transforms = list(image.applied_transforms)
    return prototype


def _resolve_batch_size(
    images: dict[str, ImagesBatch],
    points: dict[str, list[Points]],
    bounding_boxes: dict[str, list[BoundingBoxes]],
    metadata: dict[str, list[Any]],
) -> int:
    """Resolve and validate the shared length of all batch fields."""
    sizes: list[tuple[str, int]] = []
    sizes.extend(
        (f"image {name!r}", image.batch_size) for name, image in images.items()
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
