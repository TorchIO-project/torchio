"""Inverse transform utilities."""

from __future__ import annotations

import warnings
from typing import Any

from .compose import Compose
from .transform import _TRANSFORM_REGISTRY
from .transform import AppliedTransform
from .transform import IntensityTransform
from .transform import Transform


def get_inverse_transform(
    history: list[AppliedTransform],
    *,
    warn: bool = True,
    ignore_intensity: bool = False,
) -> Compose:
    """Build a Compose that inverts a list of applied transforms.

    Walks the history in reverse order. For each invertible transform,
    calls ``transform.inverse(params)`` to get the inverse transform
    instance. Non-invertible transforms are skipped.

    Args:
        history: List of ``AppliedTransform`` records.
        warn: Issue a warning for non-invertible transforms.
        ignore_intensity: Skip all intensity transforms.

    Returns:
        A ``Compose`` of inverse transforms.
    """
    steps: list[Transform] = []
    for trace in reversed(history):
        cls = _TRANSFORM_REGISTRY.get(trace.name)
        if cls is None:
            if warn:
                warnings.warn(
                    f"Unknown transform {trace.name!r} in history, skipping",
                    stacklevel=2,
                )
            continue
        if ignore_intensity and issubclass(cls, IntensityTransform):
            continue
        instance = object.__new__(cls)
        if not instance.invertible:
            if warn:
                warnings.warn(
                    f"{trace.name} is not invertible, skipping",
                    stacklevel=2,
                )
            continue
        steps.append(instance.inverse(trace.params))
    return Compose(steps, copy=False)


def apply_inverse_transform(
    data: Any,
    *,
    warn: bool = True,
    ignore_intensity: bool = False,
) -> Any:
    """Apply the inverse of all recorded transforms on the data.

    Works on any type that has ``applied_transforms`` (Subject,
    SubjectsBatch, Image, etc.). Non-invertible transforms are skipped.

    Args:
        data: Transformed data with an ``applied_transforms`` attribute.
        warn: Issue a warning for non-invertible transforms.
        ignore_intensity: Skip all intensity transforms.

    Returns:
        Data with transforms undone, same type as input.
    """
    if not hasattr(data, "applied_transforms"):
        return data
    inverse = get_inverse_transform(
        data.applied_transforms,
        warn=warn,
        ignore_intensity=ignore_intensity,
    )
    result = inverse(data)
    if hasattr(result, "applied_transforms"):
        result.applied_transforms = []
    return result
