"""Mixin for data classes that carry transform history."""

from __future__ import annotations

from typing import Any
from typing import Self


class Invertible:
    """Mixin for objects that carry ``applied_transforms`` history.

    Provides ``apply_inverse_transform()`` and
    ``get_inverse_transform()`` to undo recorded transforms.

    Classes that inherit from this mixin must initialise
    ``self.applied_transforms = []`` in their constructor.
    """

    applied_transforms: list[Any]

    def get_inverse_transform(
        self,
        *,
        warn: bool = True,
        ignore_intensity: bool = False,
    ) -> Any:
        """Get a composed transform that inverts the applied history.

        Returns a [`Compose`][torchio.Compose] of the inverse of each
        applied transform, in reverse order. Non-invertible transforms
        are skipped (with a warning if ``warn=True``).

        Args:
            warn: Issue a warning for non-invertible transforms.
            ignore_intensity: Skip all intensity transforms.

        Returns:
            A ``Compose`` transform that undoes the history.
        """
        from ..transforms.inverse import get_inverse_transform

        return get_inverse_transform(
            self.applied_transforms,
            warn=warn,
            ignore_intensity=ignore_intensity,
        )

    def apply_inverse_transform(self, **kwargs: Any) -> Self:
        """Apply the inverse of all applied transforms, in reverse order.

        Non-invertible transforms are skipped. Intensity transforms
        can be ignored with ``ignore_intensity=True``.

        Args:
            **kwargs: Forwarded to
                ``get_inverse_transform()`` (``warn``,
                ``ignore_intensity``).

        Returns:
            Data with transforms undone.

        Examples:
            >>> transformed = transform(subject)
            >>> restored = transformed.apply_inverse_transform()
        """
        inverse_transform = self.get_inverse_transform(**kwargs)
        result = inverse_transform(self)
        if hasattr(result, "applied_transforms"):
            result.applied_transforms = []
        return result

    def clear_history(self) -> None:
        """Remove all applied transform records."""
        self.applied_transforms = []
