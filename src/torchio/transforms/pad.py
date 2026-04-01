"""Pad transform: add voxels to the borders."""

from __future__ import annotations

from typing import Any

import torch

from ..data.batch import SubjectsBatch
from ..types import TypeSixInts
from ..types import TypeThreeInts
from .transform import SpatialTransform

#: Accepted padding specifications.
#: ``int`` → same amount on each side of each axis.
#: 3-tuple → symmetric per axis ``(i, j, k)``.
#: 6-tuple → per-side ``(i_ini, i_fin, j_ini, j_fin, k_ini, k_fin)``.
PaddingParam = int | TypeThreeInts | TypeSixInts


def _parse_padding(padding: PaddingParam) -> TypeSixInts:
    """Normalise padding to a 6-tuple."""
    if isinstance(padding, int):
        return (padding, padding, padding, padding, padding, padding)
    values = list(padding)
    n = len(values)
    if n == 3:
        i, j, k = values
        return (i, i, j, j, k, k)
    if n == 6:
        return (values[0], values[1], values[2], values[3], values[4], values[5])
    msg = f"Padding must have 1, 3, or 6 values, got {n}"
    raise ValueError(msg)


class Pad(SpatialTransform):
    r"""Add a border of voxels to each side of the volume.

    Args:
        padding: Tuple
            $(i_\text{ini}, i_\text{fin}, j_\text{ini}, j_\text{fin},
            k_\text{ini}, k_\text{fin})$
            defining the number of voxels added to the edges of
            each axis. If the initial shape of the image is
            $I \times J \times K$, the final shape will be
            $(I + i_\text{ini} + i_\text{fin}) \times
            (J + j_\text{ini} + j_\text{fin}) \times
            (K + k_\text{ini} + k_\text{fin})$.
            If only three values $(i, j, k)$ are provided, then
            $i_\text{ini} = i_\text{fin} = i$, etc.
            If only one value $n$ is provided, all six values are $n$.
        padding_mode: One of ``'constant'``, ``'reflect'``,
            ``'replicate'``, or ``'circular'``. See
            [`torch.nn.functional.pad`](https://pytorch.org/docs/stable/generated/torch.nn.functional.pad.html).
        fill: Fill value when ``padding_mode='constant'``.
        **kwargs: See [`Transform`][torchio.Transform] for additional
            keyword arguments.

    Examples:
        >>> import torchio as tio
        >>> transform = tio.Pad(padding=10)
        >>> transform = tio.Pad(padding=(5, 10, 0))
        >>> transform = tio.Pad(padding=10, padding_mode='reflect')
    """

    def __init__(
        self,
        *,
        padding: PaddingParam,
        padding_mode: str = "constant",
        fill: float = 0,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.padding = _parse_padding(padding)
        self.padding_mode = padding_mode
        self.fill = fill

    def make_params(self, batch: SubjectsBatch) -> dict[str, Any]:
        return {
            "padding": self.padding,
            "padding_mode": self.padding_mode,
            "fill": self.fill,
        }

    def apply_transform(
        self,
        batch: SubjectsBatch,
        params: dict[str, Any],
    ) -> SubjectsBatch:
        i0, i1, j0, j1, k0, k1 = params["padding"]
        mode = params["padding_mode"]
        fill = params["fill"]
        # F.pad expects reversed order: (k0, k1, j0, j1, i0, i1)
        pad_arg = (k0, k1, j0, j1, i0, i1)
        for _name, img_batch in self._get_images(batch).items():
            img_batch.data = torch.nn.functional.pad(
                img_batch.data,
                pad_arg,
                mode=mode,
                value=fill,
            )
            # Update each affine's origin (shift back)
            for affine in img_batch.affines:
                origin_shift = affine.data[:3, :3] @ affine.data.new_tensor(
                    [-float(i0), -float(j0), -float(k0)],
                )
                affine._matrix[:3, 3] += origin_shift
        return batch

    @property
    def invertible(self) -> bool:
        return True

    def inverse(self, params: dict[str, Any]) -> Any:
        """Inverse of Pad is Crop."""
        from .crop import Crop

        return Crop(cropping=params["padding"], copy=False)
