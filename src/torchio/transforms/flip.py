"""Flip transform: reverse voxel order along spatial axes."""

from __future__ import annotations

from typing import Any

import attrs
import torch

from ..data.subject import Subject
from .transform import SpatialTransform


def _validate_axes(
    instance: Any,
    attribute: Any,
    value: tuple[int, ...],
) -> None:
    for axis in value:
        if axis not in (0, 1, 2):
            msg = f"Flip axes must be 0, 1, 2; got {axis}"
            raise ValueError(msg)


@attrs.define(slots=False, eq=False, kw_only=True, repr=False)
class Flip(SpatialTransform):
    """Flip (reverse) voxel data along one or more spatial axes.

    This is a pure ``torch.flip`` operation — GPU-compatible and
    differentiable. Flipping is its own inverse.

    Args:
        axes: Spatial axes to flip. ``0`` = first spatial axis (I),
            ``1`` = second (J), ``2`` = third (K).

    Examples:
        >>> flip_lr = tio.Flip(axes=(0,))           # flip left-right
        >>> flip_all = tio.Flip(axes=(0, 1, 2))     # flip all axes
        >>> random_flip = tio.Flip(axes=(0,), p=0.5) # 50% chance
    """

    axes: tuple[int, ...] = attrs.field(
        default=(0,),
        converter=tuple,
        validator=_validate_axes,
    )

    def make_params(self, subject: Subject) -> dict[str, Any]:
        return {"axes": self.axes}

    def apply_transform(self, subject: Subject, params: dict[str, Any]) -> Subject:
        axes = params["axes"]
        # torch.flip dims are 1-indexed for (C, I, J, K) layout
        dims = [a + 1 for a in axes]
        for _name, image in self._get_images(subject).items():
            image.set_data(torch.flip(image.data, dims))
        return subject
