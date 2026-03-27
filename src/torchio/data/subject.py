"""Subject class."""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

import numpy as np

from ..types import TypeSpacing
from ..types import TypeSpatialShape
from ..types import TypeTensorShape
from .bboxes import BoundingBoxes
from .image import Image
from .points import Points

# Union of all spatial data types stored by Subject
_SpatialData = Image | Points | BoundingBoxes


class Subject:
    """Container for images, points, bounding boxes, and metadata.

    A `Subject` holds one or more named data entries and optional
    metadata. Data entries are classified automatically:

    - [`Image`][torchio.Image] (including `ScalarImage`, `LabelMap`)
    - [`Points`][torchio.Points]
    - [`BoundingBoxes`][torchio.BoundingBoxes]
    - Everything else is stored as metadata.

    At least one `Image` must be provided.

    Args:
        **kwargs: Named data entries and/or metadata values.

    Examples:
        >>> import torch, torchio as tio
        >>> subject = tio.Subject(
        ...     t1=tio.ScalarImage("t1.nii.gz"),
        ...     seg=tio.LabelMap("seg.nii.gz"),
        ...     landmarks=tio.Points(torch.randn(10, 3)),
        ...     tumors=tio.BoundingBoxes(
        ...         torch.tensor([[10, 20, 30, 50, 60, 70]]),
        ...         format=tio.BoundingBoxFormat.IJKIJK,
        ...     ),
        ...     age=45,
        ... )
        >>> subject.t1          # Image access
        >>> subject.landmarks   # Points access
        >>> subject.tumors      # BoundingBoxes access
        >>> subject.age         # metadata access (returns 45)
    """

    def __init__(self, **kwargs: Any) -> None:
        images: dict[str, Image] = {}
        points: dict[str, Points] = {}
        bounding_boxes: dict[str, BoundingBoxes] = {}
        metadata: dict[str, Any] = {}

        for k, v in kwargs.items():
            if isinstance(v, Image):
                images[k] = v
            elif isinstance(v, Points):
                points[k] = v
            elif isinstance(v, BoundingBoxes):
                bounding_boxes[k] = v
            else:
                metadata[k] = v

        if not images and not points and not bounding_boxes and not metadata:
            msg = "A Subject must contain at least one entry"
            raise ValueError(msg)

        self._images: dict[str, Image] = images
        self._points: dict[str, Points] = points
        self._bounding_boxes: dict[str, BoundingBoxes] = bounding_boxes
        self._metadata: dict[str, Any] = metadata
        self.applied_transforms: list[dict[str, Any]] = []

    # --- Access ---

    def __getattr__(self, name: str) -> Any:
        if name.startswith("_"):
            raise AttributeError(name)
        for store in (self._images, self._points, self._bounding_boxes):
            if name in store:
                return store[name]
        if name in self._metadata:
            return self._metadata[name]
        msg = f"{type(self).__name__} has no attribute {name!r}"
        raise AttributeError(msg)

    def __getitem__(
        self,
        item: str | int | slice | tuple[int | slice, ...],
    ) -> _SpatialData | Subject:
        """Look up a named entry, or spatially slice all images.

        When *item* is a ``str``, the corresponding data entry is
        returned (image, points, or bounding boxes).

        When *item* is an ``int``, ``slice``, or ``tuple`` of
        slices/ints, a **new** :class:`Subject` is returned with every
        image sliced identically.  All images must be spatially
        consistent (same ``spatial_shape``).  Only the **spatial**
        dimensions ``(I, J, K)`` are sliced — the channel dimension of
        each image is preserved.

        Args:
            item: A string key, or an int/slice/tuple for spatial
                indexing.

        Returns:
            A single data entry (when *item* is ``str``), or a new
            :class:`Subject` with sliced images.

        Examples:
            >>> subject["t1"]                # lookup by name
            >>> subject[10:20]               # slice dim I
            >>> subject[10:20, 30:60]        # slice I and J
            >>> subject[..., 50:100]         # slice dim K
            >>> subject[10:20, 10:20, 10:20] # all three spatial dims
        """
        if isinstance(item, str):
            for store in (self._images, self._points, self._bounding_boxes):
                if item in store:
                    return store[item]
            raise KeyError(item)

        return self._spatial_slice(item)

    def __contains__(self, name: object) -> bool:
        return any(
            name in store
            for store in (
                self._images,
                self._points,
                self._bounding_boxes,
            )
        )

    def __iter__(self) -> Iterator[str]:
        yield from self._images
        yield from self._points
        yield from self._bounding_boxes

    def __len__(self) -> int:
        return len(self._images) + len(self._points) + len(self._bounding_boxes)

    # --- Properties ---

    @property
    def metadata(self) -> dict[str, Any]:
        """Non-spatial metadata."""
        return self._metadata

    @property
    def spatial_shape(self) -> TypeSpatialShape:
        """Spatial shape, checked for consistency across all images."""
        self._check_consistent_attribute("spatial_shape")
        return self._first_image().spatial_shape

    @property
    def shape(self) -> TypeTensorShape:
        """Shape of the first image, checked for consistency."""
        self._check_consistent_attribute("shape")
        return self._first_image().shape

    @property
    def spacing(self) -> TypeSpacing:
        """Spacing from the first image, checked for consistency."""
        self._check_consistent_attribute("spacing")
        return self._first_image().spacing

    # --- Methods ---

    @property
    def images(self) -> dict[str, Image]:
        """Dict of all `Image` entries."""
        return dict(self._images)

    @property
    def points(self) -> dict[str, Points]:
        """Dict of all `Points` entries."""
        return dict(self._points)

    @property
    def bounding_boxes(self) -> dict[str, BoundingBoxes]:
        """Dict of all `BoundingBoxes` entries."""
        return dict(self._bounding_boxes)

    def all_points(self) -> dict[str | tuple[str, str], Points]:
        """Collect points from both subject-level and image-level.

        Subject-level points are keyed by their name (``str``).
        Image-level points are keyed by a ``(image_name, points_name)``
        tuple.

        Returns:
            Merged dict of all points across both levels.
        """
        result: dict[str | tuple[str, str], Points] = dict(self._points)
        for image_name, image in self._images.items():
            for points_name, pts in image.points.items():
                result[(image_name, points_name)] = pts
        return result

    def all_bounding_boxes(
        self,
    ) -> dict[str | tuple[str, str], BoundingBoxes]:
        """Collect bounding boxes from both subject-level and image-level.

        Subject-level boxes are keyed by their name (``str``).
        Image-level boxes are keyed by a ``(image_name, boxes_name)``
        tuple.

        Returns:
            Merged dict of all bounding boxes across both levels.
        """
        result: dict[str | tuple[str, str], BoundingBoxes] = dict(
            self._bounding_boxes,
        )
        for image_name, image in self._images.items():
            for box_name, boxes in image.bounding_boxes.items():
                result[(image_name, box_name)] = boxes
        return result

    def load(self) -> None:
        """Load all images from disk."""
        for image in self._images.values():
            image.load()

    def clear_history(self) -> None:
        """Remove all applied transform records."""
        self.applied_transforms = []

    # --- Internal ---

    def _first_image(self) -> Image:
        return next(iter(self._images.values()))

    def _check_consistent_attribute(
        self,
        attribute: str,
        rtol: float = 1e-6,
        atol: float = 1e-6,
    ) -> None:
        """Check that an attribute is consistent across all images."""
        values = []
        names = []
        for name, image in self._images.items():
            values.append(getattr(image, attribute))
            names.append(name)

        if len(values) < 2:
            return

        ref = values[0]
        for i, val in enumerate(values[1:], 1):
            if not np.allclose(ref, val, rtol=rtol, atol=atol):
                msg = f"Inconsistent {attribute}: {names[0]}={ref}, {names[i]}={val}"
                raise RuntimeError(msg)

    def _spatial_slice(
        self,
        item: int | slice | tuple[int | slice, ...],
    ) -> Subject:
        """Slice all images along spatial dimensions (I, J, K).

        The channel dimension of each image is preserved. All images
        must have the same ``spatial_shape``.
        """
        if not self._images:
            msg = "Cannot spatially slice a Subject with no images"
            raise RuntimeError(msg)

        self._check_consistent_attribute("spatial_shape")

        # Normalise to tuple
        if isinstance(item, (int, slice)) or item is Ellipsis:
            items: tuple[int | slice | type(Ellipsis), ...] = (item,)
        elif isinstance(item, tuple):
            items = item
        else:
            msg = f"Index type {type(item).__name__} not understood"
            raise TypeError(msg)

        # Slice each image, prepending slice(None) for channels
        sliced_images: dict[str, Image] = {}
        for name, image in self._images.items():
            sliced_images[name] = image[(slice(None), *items)]

        kwargs: dict[str, Any] = dict(sliced_images)
        kwargs.update(self._points)
        kwargs.update(self._bounding_boxes)
        kwargs.update(self._metadata)
        new = type(self)(**kwargs)
        new.applied_transforms = list(self.applied_transforms)
        return new

    def __repr__(self) -> str:
        parts = []
        if self._images:
            parts.append(f"images: {tuple(self._images.keys())}")
        if self._points:
            parts.append(f"points: {tuple(self._points.keys())}")
        if self._bounding_boxes:
            parts.append(f"bboxes: {tuple(self._bounding_boxes.keys())}")
        return f"{type(self).__name__}({'; '.join(parts)})"

    def _repr_html_(self) -> str:
        """Rich HTML representation for Jupyter notebooks."""
        from ..repr_html import subject_to_html

        return subject_to_html(self)
