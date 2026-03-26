"""Subject class."""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

import numpy as np

from ..types import TypeSpacing
from ..types import TypeSpatialShape
from ..types import TypeTensorShape
from .image import Image


class Subject:
    """Container for images and metadata associated with a single subject.

    A `Subject` holds one or more named images (`Image`, `ScalarImage`,
    `LabelMap`) and optional metadata. Everything that is an `Image` instance
    is stored as an image; everything else is stored as metadata. All entries
    are accessible via attribute or dict-style access.

    Args:
        **kwargs: Named `Image` instances and/or metadata values.
            At least one `Image` must be provided.

    Examples:
        >>> import torchio as tio
        >>> subject = tio.Subject(
        ...     t1=tio.ScalarImage("t1.nii.gz"),
        ...     seg=tio.LabelMap("seg.nii.gz"),
        ...     age=45,
        ...     diagnosis="healthy",
        ... )
        >>> subject.t1          # Image access
        >>> subject["seg"]      # dict-style Image access
        >>> subject.age         # metadata access (returns 45)
        >>> subject.metadata    # {"age": 45, "diagnosis": "healthy"}

        To create from an existing dict, unpack it:

        >>> data = {"t1": tio.ScalarImage("t1.nii.gz"), "age": 45}
        >>> subject = tio.Subject(**data)
    """

    def __init__(self, **kwargs: Any) -> None:
        images: dict[str, Image] = {}
        metadata: dict[str, Any] = {}
        for k, v in kwargs.items():
            if isinstance(v, Image):
                images[k] = v
            else:
                metadata[k] = v

        if not images:
            msg = "A Subject must contain at least one Image"
            raise ValueError(msg)

        self._images: dict[str, Image] = images
        self._metadata: dict[str, Any] = metadata
        self.applied_transforms: list[dict[str, Any]] = []

    # --- Access ---

    def __getattr__(self, name: str) -> Any:
        if name.startswith("_"):
            raise AttributeError(name)
        if name in self._images:
            return self._images[name]
        if name in self._metadata:
            return self._metadata[name]
        msg = f"{type(self).__name__} has no attribute {name!r}"
        raise AttributeError(msg)

    def __getitem__(self, name: str) -> Image:
        return self._images[name]

    def __contains__(self, name: object) -> bool:
        return name in self._images

    def __iter__(self) -> Iterator[str]:
        return iter(self._images)

    def __len__(self) -> int:
        return len(self._images)

    # --- Properties ---

    @property
    def metadata(self) -> dict[str, Any]:
        """Non-image metadata."""
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
        keys = tuple(self._images.keys())
        n_images = len(self._images)
        return f"{type(self).__name__}(keys: {keys}; images: {n_images})"
