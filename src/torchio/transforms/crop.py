"""Crop transform: remove voxels from the borders."""

from __future__ import annotations

from typing import Any

from ..data.batch import SubjectsBatch
from ..types import TypeSixInts
from ..types import TypeThreeInts
from .transform import SpatialTransform

#: Accepted cropping specifications.
#: ``int`` → same amount from each side of each axis.
#: 3-tuple → symmetric per axis ``(i, j, k)``.
#: 6-tuple → per-side ``(i_ini, i_fin, j_ini, j_fin, k_ini, k_fin)``.
CroppingParam = int | TypeThreeInts | TypeSixInts


def _parse_cropping(cropping: CroppingParam) -> TypeSixInts:
    """Normalise cropping to a 6-tuple (i_ini, i_fin, j_ini, j_fin, k_ini, k_fin)."""
    if isinstance(cropping, int):
        return (cropping, cropping, cropping, cropping, cropping, cropping)
    values = list(cropping)
    n = len(values)
    if n == 3:
        i, j, k = values
        return (i, i, j, j, k, k)
    if n == 6:
        return (values[0], values[1], values[2], values[3], values[4], values[5])
    msg = f"Cropping must have 1, 3, or 6 values, got {n}"
    raise ValueError(msg)


class Crop(SpatialTransform):
    r"""Remove a border of voxels from each side of the volume.

    Args:
        cropping: Number of voxels to crop. Accepted forms:

            - ``int``: crop the same amount from every side of every
              axis.
            - 3 values ``(i, j, k)``: crop symmetrically per axis.
            - 6 values ``(i₀, i₁, j₀, j₁, k₀, k₁)``: crop a
              specific amount from the start and end of each axis.

        **kwargs: See [`Transform`][torchio.Transform] for additional
            keyword arguments.

    Examples:
        >>> import torchio as tio
        >>> # Remove 10 voxels from each side
        >>> transform = tio.Crop(cropping=10)
        >>> # Remove 5 along I, 10 along J, 0 along K
        >>> transform = tio.Crop(cropping=(5, 10, 0))
        >>> # Fine-grained: (i_start, i_end, j_start, j_end, k_start, k_end)
        >>> transform = tio.Crop(cropping=(2, 3, 4, 5, 6, 7))
    """

    def __init__(
        self,
        *,
        cropping: CroppingParam = 0,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.cropping = _parse_cropping(cropping)

    def make_params(self, batch: SubjectsBatch) -> dict[str, Any]:
        return {"cropping": self.cropping}

    def apply_transform(
        self,
        batch: SubjectsBatch,
        params: dict[str, Any],
    ) -> SubjectsBatch:
        i0, i1, j0, j1, k0, k1 = params["cropping"]
        for _name, img_batch in self._get_images(batch).items():
            data = img_batch.data
            # data shape: (B, C, I, J, K)
            si = data.shape[-3]
            sj = data.shape[-2]
            sk = data.shape[-1]
            img_batch.data = data[
                ...,
                i0 : si - i1 or None,
                j0 : sj - j1 or None,
                k0 : sk - k1 or None,
            ]
            # Update each affine's origin
            for affine in img_batch.affines:
                origin_shift = affine.data[:3, :3] @ affine.data.new_tensor(
                    [float(i0), float(j0), float(k0)],
                )
                affine._matrix[:3, 3] += origin_shift
        return batch
