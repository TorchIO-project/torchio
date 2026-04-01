"""Flip transform: reverse voxel order along spatial axes."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import torch

from ..data.batch import SubjectsBatch
from .transform import SpatialTransform


class Flip(SpatialTransform):
    """Reverse the order of elements in an image along the given axes.

    Args:
        axes: Tuple of indices of the spatial dimensions along which
            the image will be flipped. Values must be in
            ``{0, 1, 2}``, corresponding to the ``I``, ``J``, ``K``
            axes of the ``(C, I, J, K)`` tensor layout.
        **kwargs: See [`Transform`][torchio.Transform] for additional
            keyword arguments.

    Examples:
        >>> import torchio as tio
        >>> # Flip along the first spatial axis
        >>> transform = tio.Flip(axes=(0,))
        >>> # Flip along all three axes
        >>> transform = tio.Flip(axes=(0, 1, 2))
        >>> # Random flip (50% probability)
        >>> transform = tio.Flip(axes=(0,), p=0.5)
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

    def make_params(self, batch: SubjectsBatch) -> dict[str, Any]:
        return {"axes": self.axes}

    def apply_transform(
        self,
        batch: SubjectsBatch,
        params: dict[str, Any],
    ) -> SubjectsBatch:
        dims = [a - 3 for a in params["axes"]]
        for _name, img_batch in self._get_images(batch).items():
            img_batch.data = torch.flip(img_batch.data, dims)
        return batch
