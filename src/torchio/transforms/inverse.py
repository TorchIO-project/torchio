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
    data: Any | None = None,
    warn: bool = True,
    ignore_intensity: bool = False,
) -> Compose:
    """Build a Compose that inverts a list of applied transforms.

    Walks the history in reverse order. For each invertible transform,
    calls `transform.inverse(params)` to get the inverse transform
    instance. Non-invertible transforms are skipped.

    Args:
        history: List of `AppliedTransform` records.
        data: Data that will receive the inverse transforms.
        warn: Issue a warning for non-invertible transforms.
        ignore_intensity: Skip all intensity transforms.

    Returns:
        A `Compose` of inverse transforms.
    """
    steps: list[Transform] = []
    current_image_names = _get_image_names(data)
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
        inverse = instance.inverse(trace.params)
        if not _restrict_inverse_to_trace_images(
            inverse,
            trace,
            current_image_names,
        ):
            continue
        steps.append(inverse)
    # copy=True (the default) so applying the inverse does not mutate the
    # caller's data in place, matching every other transform.
    return Compose(steps)


def _restrict_inverse_to_trace_images(
    inverse: Transform,
    trace: AppliedTransform,
    current_image_names: set[str] | None,
) -> bool:
    if trace.image_names is None or current_image_names is None:
        return True
    matching_names = [name for name in trace.image_names if name in current_image_names]
    if not matching_names:
        return False
    inverse.include = matching_names
    inverse.exclude = None
    return True


def _get_image_names(data: Any | None) -> set[str] | None:
    if data is None:
        return None
    images = getattr(data, "images", None)
    if images is None:
        return None
    return set(images)


def apply_inverse_transform(
    data: Any,
    *,
    warn: bool = True,
    ignore_intensity: bool = False,
) -> Any:
    """Apply the inverse of all recorded transforms on the data.

    Works on any type that has `applied_transforms` (Subject,
    SubjectsBatch, Image, etc.). Non-invertible transforms are skipped.

    Args:
        data: Transformed data with an `applied_transforms` attribute.
        warn: Issue a warning for non-invertible transforms.
        ignore_intensity: Skip all intensity transforms.

    Returns:
        Data with transforms undone, same type as input.
    """
    if not hasattr(data, "applied_transforms"):
        return data
    # Batches with per-element histories (from per-instance OneOf/SomeOf)
    # know how to invert each element; delegate to their own method.
    if getattr(data, "_per_element_history", None) is not None:
        return data.apply_inverse_transform(
            warn=warn,
            ignore_intensity=ignore_intensity,
        )
    inverse = get_inverse_transform(
        data.applied_transforms,
        data=data,
        warn=warn,
        ignore_intensity=ignore_intensity,
    )
    result = inverse(data)
    if hasattr(result, "applied_transforms"):
        result.applied_transforms = []
    return result
