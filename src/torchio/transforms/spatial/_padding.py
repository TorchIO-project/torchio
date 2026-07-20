"""Shared spatial padding helpers."""

from __future__ import annotations

import warnings
from typing import Literal
from typing import get_args

import torch
from torch import Tensor

from ...types import TypeSixInts
from .._statistics import compute_quantile

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

_PADDING_MODES: tuple[PaddingMode, ...] = get_args(PaddingMode)
_STATISTIC_PADDING_MODES = "mean", "median", "minimum"


def parse_padding_mode(padding_mode: str) -> PaddingMode:
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

    float_flat = flat if data.dtype in (torch.float32, torch.float64) else flat.float()
    if padding_mode == "mean":
        statistic = float_flat.mean(dim=1)
    else:
        statistic = torch.stack(
            [compute_quantile(values, 0.5) for values in float_flat],
        )
    return statistic.to(data.dtype)


def pad_tensor(
    data: Tensor,
    padding: TypeSixInts,
    padding_mode: PaddingMode,
    fill: float,
) -> Tensor:
    """Pad a 4D image tensor or 5D image batch."""
    if data.ndim not in (4, 5):
        msg = f"Expected a 4D or 5D image tensor, got {data.ndim}D"
        raise ValueError(msg)
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
