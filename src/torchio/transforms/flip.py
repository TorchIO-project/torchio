"""Flip transform: reverse voxel order along spatial axes."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import torch

from ..data.subject import Subject
from .transform import SpatialTransform


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

    def __init__(
        self,
        *,
        axes: Sequence[int] = (0,),
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        axes = tuple(axes)
        for axis in axes:
            if axis not in (0, 1, 2):
                msg = f"Flip axes must be 0, 1, 2; got {axis}"
                raise ValueError(msg)
        self.axes = axes

    def make_params(self, subject: Subject) -> dict[str, Any]:
        return {"axes": self.axes}

    def apply_transform(self, subject: Subject, params: dict[str, Any]) -> Subject:
        axes = params["axes"]
        dims = [a + 1 for a in axes]
        for _name, image in self._get_images(subject).items():
            image.set_data(torch.flip(image.data, dims))
        return subject
