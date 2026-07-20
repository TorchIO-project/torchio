"""Pad transform: add voxels to the borders."""

from __future__ import annotations

import warnings
from typing import Any
from typing import Literal

import torch
from torch import Tensor

from ...data.batch import SubjectsBatch
from ...types import TypeSixInts
from ...types import TypeThreeInts
from ..intensity.normalize import _quantile
from ..transform import SpatialTransform

#: Accepted padding specifications.
#: `int` → same amount on each side of each axis.
#: 3-tuple → symmetric per axis `(i, j, k)`.
#: 6-tuple → per-side `(i_ini, i_fin, j_ini, j_fin, k_ini, k_fin)`.
PaddingParam = int | TypeThreeInts | TypeSixInts

#: Accepted padding modes.
PaddingMode = Literal[
    "constant",
    "reflect",
    "replicate",
    "circular",
    "mean",
    "median",
    "minimum",
]

_PADDING_MODES: tuple[PaddingMode, ...] = (
    "constant",
    "reflect",
    "replicate",
    "circular",
    "mean",
    "median",
    "minimum",
)

_STATISTIC_PADDING_MODES = ("mean", "median", "minimum")


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


def _parse_padding_mode(padding_mode: str) -> PaddingMode:
    """Validate and return a padding mode."""
    if padding_mode not in _PADDING_MODES:
        msg = f"padding_mode must be one of {_PADDING_MODES}, got {padding_mode!r}"
        raise ValueError(msg)
    return padding_mode


def _compute_padding_statistic(
    data: Tensor,
    padding_mode: PaddingMode,
) -> Tensor:
    """Compute one whole-volume padding statistic per batch element."""
    flat = data.flatten(start_dim=1)
    if padding_mode == "minimum":
        return flat.amin(dim=1)

    if not torch.is_floating_point(data):
        warnings.warn(
            f'The constant value computed for padding mode "{padding_mode}"'
            " might be truncated in the output, as the data type of the input"
            " image is not float. Consider converting the image to a floating"
            " point type before applying this transform.",
            RuntimeWarning,
            stacklevel=4,
        )

    float_flat = flat.float()
    if padding_mode == "mean":
        statistic = float_flat.mean(dim=1)
    else:
        statistic = torch.stack(
            [_quantile(values, 0.5) for values in float_flat],
        )
    return statistic.to(data.dtype)


def _pad_tensor(
    data: Tensor,
    padding: TypeSixInts,
    padding_mode: PaddingMode,
    fill: float,
) -> Tensor:
    """Pad a 4D image tensor or 5D image batch."""
    assert data.ndim in (4, 5)
    i0, i1, j0, j1, k0, k1 = padding
    pad_arg = k0, k1, j0, j1, i0, i1
    if padding_mode not in _STATISTIC_PADDING_MODES:
        return torch.nn.functional.pad(
            data,
            pad_arg,
            mode=padding_mode,
            value=fill,
        )

    is_unbatched = data.ndim == 4
    batch = data.unsqueeze(0) if is_unbatched else data
    statistic = _compute_padding_statistic(batch, padding_mode)
    padded = torch.nn.functional.pad(batch, pad_arg)
    interior = torch.ones(
        (1, 1, *batch.shape[-3:]),
        dtype=torch.bool,
        device=batch.device,
    )
    interior = torch.nn.functional.pad(interior, pad_arg)
    fill_values = statistic.reshape(-1, 1, 1, 1, 1)
    result = torch.where(interior, padded, fill_values)
    return result[0] if is_unbatched else result


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
        padding_mode: One of `'constant'`, `'reflect'`,
            `'replicate'`, `'circular'`, `'mean'`, `'median'`, or
            `'minimum'`. Statistical modes use one value computed from
            the whole image volume. For integer inputs, `'mean'` and
            `'median'` may be truncated to the input dtype and emit a
            warning.
        fill: Fill value when `padding_mode='constant'`.
        **kwargs: See [`Transform`][torchio.Transform] for additional
            keyword arguments.

    Examples:
        >>> import torchio as tio
        >>> transform = tio.Pad(padding=10)
        >>> transform = tio.Pad(padding=(5, 10, 0))
        >>> transform = tio.Pad(padding=10, padding_mode='reflect')
        >>> transform = tio.Pad(padding=10, padding_mode='minimum')
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
        self.padding_mode = _parse_padding_mode(padding_mode)
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
        for _name, img_batch in self._get_images(batch).items():
            img_batch.data = _pad_tensor(
                img_batch.data,
                (i0, i1, j0, j1, k0, k1),
                mode,
                fill,
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
