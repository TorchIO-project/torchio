"""Flip transform: reverse voxel order along spatial axes."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import torch

from ..data.batch import SubjectsBatch
from .transform import SpatialTransform

# Map anatomical labels to axis indices.
# Only the first letter (uppercased) is used.
_LABEL_TO_AXIS: dict[str, tuple[str, str]] = {
    "L": ("L", "R"),
    "R": ("L", "R"),
    "A": ("A", "P"),
    "P": ("A", "P"),
    "I": ("I", "S"),
    "S": ("I", "S"),
}


def _resolve_axes(
    axes: int | str | Sequence[int | str],
    orientation: tuple[str, str, str] | None = None,
) -> tuple[int, ...]:
    """Normalise axes to a tuple of ints (0, 1, 2).

    Accepts ints, anatomical strings (``'L'``, ``'Right'``, ``'AP'``,
    etc.), or a mix. String axes are resolved using the image
    orientation.
    """
    if isinstance(axes, (int, str)):
        axes = (axes,)
    result: list[int] = []
    for axis in axes:
        if isinstance(axis, int):
            if axis not in (0, 1, 2):
                msg = f"Axis must be 0, 1, or 2; got {axis}"
                raise ValueError(msg)
            result.append(axis)
        elif isinstance(axis, str):
            letter = axis[0].upper()
            if letter not in _LABEL_TO_AXIS:
                msg = (
                    f"Unknown anatomical label {axis!r}."
                    " Use L, R, A, P, I, S or full names"
                    " like 'Left', 'Right', etc."
                )
                raise ValueError(msg)
            if orientation is None:
                msg = (
                    "Cannot resolve anatomical axis label"
                    f" {axis!r} without image orientation"
                )
                raise ValueError(msg)
            pair = _LABEL_TO_AXIS[letter]
            for dim, code in enumerate(orientation):
                if code in pair:
                    result.append(dim)
                    break
        else:
            msg = f"Axis must be int or str, got {type(axis).__name__}"
            raise TypeError(msg)
    return tuple(sorted(set(result)))


class Flip(SpatialTransform):
    r"""Reverse the order of elements in an image along the given axes.

    Args:
        axes: Index or tuple of indices of the spatial dimensions along
            which the image might be flipped. Integers must be in
            ``{0, 1, 2}``. Anatomical labels may also be used, such as
            ``'Left'``, ``'Right'``, ``'Anterior'``, ``'Posterior'``,
            ``'Inferior'``, ``'Superior'``. Only the first letter of
            the string is used. Anatomical labels are resolved using
            the image orientation.
        flip_probability: Probability that each axis will be flipped
            (per-axis coin flip). This is independent of the ``p``
            parameter, which gates the entire transform.
        **kwargs: See [`Transform`][torchio.Transform] for additional
            keyword arguments.

    Tip:
        Specifying the axes as anatomical labels is useful when the
        image orientation is not known.

    Examples:
        >>> import torchio as tio
        >>> # Flip along the first spatial axis
        >>> transform = tio.Flip(axes=0)
        >>> # Flip along the lateral axis (anatomical label)
        >>> transform = tio.Flip(axes='LR')
        >>> # Random per-axis flip with 50% chance each
        >>> transform = tio.Flip(axes=(0, 1, 2), flip_probability=0.5)
    """

    def __init__(
        self,
        *,
        axes: int | str | Sequence[int | str] = 0,
        flip_probability: float = 1.0,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.axes = axes
        if not 0 <= flip_probability <= 1:
            msg = f"flip_probability must be in [0, 1], got {flip_probability}"
            raise ValueError(msg)
        self.flip_probability = flip_probability

    def make_params(
        self,
        batch: SubjectsBatch,
    ) -> dict[str, Any]:
        # Resolve string axes using the first image's orientation
        orientation = None
        first_img = next(iter(batch.images.values()))
        if first_img.batch_size > 0:
            img = first_img[0]
            orientation = img.orientation
        resolved = _resolve_axes(self.axes, orientation)

        # Per-axis coin flip
        flip_mask = torch.rand(3) < self.flip_probability
        axes_to_flip = tuple(a for a in resolved if flip_mask[a].item())
        return {"axes": axes_to_flip}

    def apply_transform(
        self,
        batch: SubjectsBatch,
        params: dict[str, Any],
    ) -> SubjectsBatch:
        axes = params["axes"]
        if not axes:
            return batch
        dims = [a - 3 for a in axes]
        for _name, img_batch in self._get_images(batch).items():
            img_batch.data = torch.flip(img_batch.data, dims)
        return batch
