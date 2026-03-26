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

        if not images:
            msg = "A Subject must contain at least one Image"
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

    def __getitem__(self, name: str) -> _SpatialData:
        for store in (self._images, self._points, self._bounding_boxes):
            if name in store:
                return store[name]
        raise KeyError(name)

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

    def images(self) -> dict[str, Image]:
        """Return dict of all `Image` entries."""
        return dict(self._images)

    def points(self) -> dict[str, Points]:
        """Return dict of all `Points` entries."""
        return dict(self._points)

    def bounding_boxes(self) -> dict[str, BoundingBoxes]:
        """Return dict of all `BoundingBoxes` entries."""
        return dict(self._bounding_boxes)

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

    def __repr__(self) -> str:
        parts = []
        if self._images:
            parts.append(f"images: {tuple(self._images.keys())}")
        if self._points:
            parts.append(f"points: {tuple(self._points.keys())}")
        if self._bounding_boxes:
            parts.append(f"bboxes: {tuple(self._bounding_boxes.keys())}")
        return f"{type(self).__name__}({'; '.join(parts)})"
