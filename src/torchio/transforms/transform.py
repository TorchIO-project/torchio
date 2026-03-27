"""Transform base classes."""

from __future__ import annotations

from dataclasses import dataclass
from dataclasses import field
from typing import Any

import attrs
import torch
from torch import Tensor
from torch import nn

from ..data.image import Image
from ..data.image import ScalarImage
from ..data.subject import Subject


def _validate_probability(
    instance: Any,
    attribute: attrs.Attribute,  # type: ignore[type-arg]
    value: float,
) -> None:
    if not 0 <= value <= 1:
        msg = f"Probability must be in [0, 1], got {value}"
        raise ValueError(msg)


@dataclass
class AppliedTransform:
    """Record of a transform application, stored in Subject history.

    Attributes:
        name: Class name of the transform.
        params: Sampled parameters (JSON-serializable).
    """

    name: str
    params: dict[str, Any] = field(default_factory=dict)


@attrs.define(slots=False, eq=False, kw_only=True)
class Transform(nn.Module):
    """Base class for all TorchIO transforms.

    Transforms accept a ``Subject``, ``Image``, or ``Tensor`` and return
    the same type. Internally, non-Subject inputs are wrapped in a
    temporary Subject and unwrapped on output.

    Args:
        p: Probability of applying the transform.
        include: Image names to include (``None`` = all).
        exclude: Image names to exclude (``None`` = none).
    """

    p: float = attrs.field(default=1.0, validator=_validate_probability)
    include: list[str] | None = None
    exclude: list[str] | None = None

    def __attrs_post_init__(self) -> None:
        nn.Module.__init__(self)

    def forward(
        self,
        data: Subject | Image | Tensor,
    ) -> Subject | Image | Tensor:
        """Apply the transform.

        Args:
            data: A Subject, Image, or 4D Tensor.

        Returns:
            Transformed data of the same type as input.
        """
        subject, unwrap = self._wrap(data)
        if torch.rand(1).item() > self.p:
            return unwrap(subject)
        params = self.make_params(subject)
        subject = self.apply(subject, params)
        subject.applied_transforms.append(
            AppliedTransform(
                name=type(self).__name__,
                params=params,
            ),
        )
        return unwrap(subject)

    def make_params(self, subject: Subject) -> dict[str, Any]:
        """Sample random parameters for this transform.

        Override in subclasses that have random behavior.

        Args:
            subject: The input subject (for shape-dependent sampling).

        Returns:
            Dict of sampled parameters.
        """
        return {}

    def apply(self, subject: Subject, params: dict[str, Any]) -> Subject:
        """Apply the transform with the given parameters.

        Must be overridden by subclasses.

        Args:
            subject: Subject to transform.
            params: Parameters from ``make_params``.

        Returns:
            Transformed subject.
        """
        raise NotImplementedError

    def _get_images(self, subject: Subject) -> dict[str, Image]:
        """Get images filtered by include/exclude."""
        images = subject.images
        if self.include is not None:
            images = {k: v for k, v in images.items() if k in self.include}
        if self.exclude is not None:
            images = {k: v for k, v in images.items() if k not in self.exclude}
        return images

    @staticmethod
    def _wrap(
        data: Subject | Image | Tensor,
    ) -> tuple[Subject, Any]:
        """Wrap non-Subject input into a Subject; return (subject, unwrap_fn)."""
        if isinstance(data, Subject):
            return data, _unwrap_subject
        if isinstance(data, Image):
            sub = Subject(tio_default_image=data)
            return sub, _unwrap_image
        if isinstance(data, Tensor):
            img = ScalarImage.from_tensor(data)
            sub = Subject(tio_default_image=img)
            return sub, _unwrap_tensor
        msg = f"Expected Subject, Image, or Tensor, got {type(data).__name__}"
        raise TypeError(msg)


def _unwrap_subject(subject: Subject) -> Subject:
    return subject


def _unwrap_image(subject: Subject) -> Image:
    return subject.tio_default_image


def _unwrap_tensor(subject: Subject) -> Tensor:
    return subject.tio_default_image.data


@attrs.define(slots=False, eq=False, kw_only=True)
class SpatialTransform(Transform):
    """Base for transforms that modify spatial geometry.

    Spatial transforms apply to all images (ScalarImage and LabelMap),
    and also transform any Points and BoundingBoxes attached to the
    Subject.
    """


@attrs.define(slots=False, eq=False, kw_only=True)
class IntensityTransform(Transform):
    """Base for transforms that modify voxel intensities.

    Intensity transforms apply only to ``ScalarImage`` instances,
    leaving ``LabelMap`` and annotations unchanged.
    """

    def _get_images(self, subject: Subject) -> dict[str, Image]:
        """Filter to ScalarImage only, then apply include/exclude."""
        images = {k: v for k, v in subject.images.items() if isinstance(v, ScalarImage)}
        if self.include is not None:
            images = {k: v for k, v in images.items() if k in self.include}
        if self.exclude is not None:
            images = {k: v for k, v in images.items() if k not in self.exclude}
        return images
