"""Shared statistical helpers for transforms."""

from __future__ import annotations

import math

import torch
from torch import Tensor


def compute_quantile(values: Tensor, q: float) -> Tensor:
    """Compute a single quantile of a one-dimensional tensor.

    `torch.quantile` raises `RuntimeError: quantile() input tensor is too
    large` for inputs with more than `2**24` elements, which is easily
    reached by high-resolution volumes. `torch.kthvalue` has no such size
    limit and is much faster on large tensors, so it is used here with linear
    interpolation to reproduce the default `torch.quantile` behavior.

    This is adapted from the solution by Elie Goudout (`@ego-thales`):
    https://github.com/pytorch/pytorch/issues/157431#issuecomment-3026856373

    Args:
        values: One-dimensional tensor of values.
        q: Quantile to compute, in the `[0, 1]` range.

    Returns:
        Zero-dimensional tensor with the computed quantile.

    Raises:
        ValueError: If `q` is outside the `[0, 1]` range.
    """
    if not 0 <= q <= 1:
        msg = f"Only values 0 <= q <= 1 are supported, but got {q!r}"
        raise ValueError(msg)
    index = q * (values.numel() - 1)
    lower = math.floor(index)
    lower_value = torch.kthvalue(values, lower + 1).values
    if index == lower:
        return lower_value
    upper_value = torch.kthvalue(values, lower + 2).values
    weight = index - lower
    return lower_value.lerp(upper_value, weight)
