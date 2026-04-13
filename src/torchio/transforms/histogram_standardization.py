"""Histogram standardization following Nyúl and Udupa (1999).

This module provides:

- `compute_histogram_landmarks`: a standalone function that computes
  average percentile landmarks from a set of training images.
- [`HistogramStandardization`][torchio.HistogramStandardization]:
  a transform that applies piecewise-linear histogram mapping using
  precomputed landmarks.
"""

from __future__ import annotations

from collections.abc import Callable
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import Tensor

from ..data.batch import SubjectsBatch
from ..data.image import ScalarImage
from .transform import IntensityTransform

DEFAULT_CUTOFF: tuple[float, float] = (0.01, 0.99)
STANDARD_RANGE: tuple[float, float] = (0.0, 100.0)

# The v1-compatible percentile set: cutoff endpoints + deciles + quartiles.
# After dedup and sort this gives 12 values; the piecewise map uses 11 segments.
_DEFAULT_QUANTILES: tuple[float, ...] = (
    0.01,
    0.10,
    0.20,
    0.25,
    0.30,
    0.40,
    0.50,
    0.60,
    0.70,
    0.75,
    0.80,
    0.90,
    0.99,
)


def compute_histogram_landmarks(
    images: Sequence[ScalarImage | Path | str],
    *,
    quantiles: Sequence[float] | None = None,
    cutoff: tuple[float, float] = DEFAULT_CUTOFF,
    masking_method: Callable[[Tensor], Tensor] | None = None,
) -> Tensor:
    """Compute average histogram landmarks from training images.

    Implements the training phase of
    `Nyúl and Udupa (1999) <https://ieeexplore.ieee.org/document/836373>`_.
    The returned landmarks tensor can be passed directly to
    [`HistogramStandardization`][torchio.HistogramStandardization].

    Args:
        images: Training images.  Each element can be a
            [`ScalarImage`][torchio.ScalarImage], a file path, or a
            string path.
        quantiles: Quantile positions in ``[0, 1]`` used as control
            points.  Must be sorted and include the cutoff endpoints.
            ``None`` uses the default 13-point scheme (v1-compatible).
        cutoff: Lower and upper quantile bounds for the intensity
            range of interest.  Defaults to ``(0.01, 0.99)``.
        masking_method: Optional callable that takes a 4-D tensor
            ``(C, I, J, K)`` and returns a boolean mask of the same
            shape.  Only ``True`` voxels are used for percentile
            computation.  ``None`` uses all voxels.

    Returns:
        1-D tensor of landmark values, one per quantile.

    Examples:
        >>> import torchio as tio
        >>> from torchio.transforms.histogram_standardization import (
        ...     compute_histogram_landmarks,
        ... )
        >>> landmarks = compute_histogram_landmarks([  # doctest: +SKIP
        ...     tio.ScalarImage("subject_a_t1.nii"),
        ...     tio.ScalarImage("subject_b_t1.nii"),
        ... ])
    """
    if quantiles is None:
        quantiles = _build_quantiles(cutoff)
    else:
        quantiles = tuple(sorted(set(quantiles)))

    _validate_quantiles(quantiles, cutoff)
    percentiles = [100.0 * q for q in quantiles]

    all_percentile_values: list[np.ndarray] = []
    for img_source in images:
        tensor = _load_tensor(img_source)
        if masking_method is not None:
            mask = masking_method(tensor)
        else:
            mask = torch.ones_like(tensor, dtype=torch.bool)
        values = tensor[mask].numpy()
        pv = np.percentile(values, percentiles)
        all_percentile_values.append(pv)

    database = np.vstack(all_percentile_values)
    landmarks = _compute_average_mapping(database)
    return torch.as_tensor(landmarks, dtype=torch.float32)


def _build_quantiles(cutoff: tuple[float, float]) -> tuple[float, ...]:
    """Build the default quantile set from cutoff and standard positions."""
    raw = set(_DEFAULT_QUANTILES)
    raw.add(cutoff[0])
    raw.add(cutoff[1])
    return tuple(sorted(raw))


def _validate_quantiles(
    quantiles: tuple[float, ...],
    cutoff: tuple[float, float],
) -> None:
    """Validate quantile array."""
    if len(quantiles) < 2:
        msg = f"Need at least 2 quantiles, got {len(quantiles)}"
        raise ValueError(msg)
    if any(q < 0 or q > 1 for q in quantiles):
        msg = "All quantiles must be in [0, 1]"
        raise ValueError(msg)
    if cutoff[0] not in quantiles or cutoff[1] not in quantiles:
        msg = (
            f"Cutoff values {cutoff} must be included in quantiles. "
            f"Got quantiles: {quantiles}"
        )
        raise ValueError(msg)


def _load_tensor(source: ScalarImage | Path | str) -> Tensor:
    """Load a 4-D tensor from various source types."""
    if isinstance(source, ScalarImage):
        return source.data
    return ScalarImage(source).data


def _compute_average_mapping(database: np.ndarray) -> np.ndarray:
    """Map percentile landmarks to the standard range via linear regression.

    Args:
        database: ``(N, P)`` array of percentile values for *N* images
            and *P* quantile positions.

    Returns:
        ``(P,)`` array of averaged landmark values in the standard range.
    """
    pc_low = database[:, 0]
    pc_high = database[:, -1]
    s_low, s_high = STANDARD_RANGE
    slopes = (s_high - s_low) / (pc_high - pc_low)
    slopes = np.nan_to_num(slopes)
    intercept = float(np.mean(s_low - slopes * pc_low))
    n = len(database)
    mapping = slopes @ database / n + intercept
    return mapping


class HistogramStandardization(IntensityTransform):
    r"""Apply piecewise-linear histogram standardization.

    Implementation of
    [Nyúl and Udupa (1999)](https://ieeexplore.ieee.org/document/836373).

    Landmarks must be precomputed using
    [`compute_histogram_landmarks`][torchio.transforms.histogram_standardization.compute_histogram_landmarks]
    and are passed directly to this transform.  Each instance targets
    **one modality**; for multi-modal subjects, compose multiple
    instances with the ``include`` parameter:

    ```python
    tio.Compose([
        tio.HistogramStandardization(t1_landmarks, include=["t1"]),
        tio.HistogramStandardization(t2_landmarks, include=["t2"]),
    ])
    ```

    Args:
        landmarks: 1-D tensor (or path to a ``.npy`` / ``.pt`` file)
            of standard-space landmark values, as returned by
            [`compute_histogram_landmarks`][torchio.transforms.histogram_standardization.compute_histogram_landmarks].
        cutoff: Lower and upper quantile bounds.
        **kwargs: See [`Transform`][torchio.Transform].

    Examples:
        >>> import torchio as tio
        >>> landmarks = torch.linspace(0, 100, 13)
        >>> transform = tio.HistogramStandardization(landmarks)
    """

    def __init__(
        self,
        landmarks: Tensor | Path | str,
        *,
        cutoff: tuple[float, float] = DEFAULT_CUTOFF,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.landmarks = _load_landmarks(landmarks)
        self.cutoff = cutoff

    def make_params(self, batch: SubjectsBatch) -> dict[str, Any]:
        """No random parameters."""
        return {}

    def apply_transform(
        self,
        batch: SubjectsBatch,
        params: dict[str, Any],
    ) -> SubjectsBatch:
        """Apply histogram standardization to each selected image."""
        for _name, img_batch in self._get_images(batch).items():
            for i in range(img_batch.batch_size):
                img_batch.data[i] = _apply_histogram_standardization(
                    img_batch.data[i],
                    self.landmarks,
                    self.cutoff,
                )
        return batch


def _load_landmarks(source: Tensor | Path | str) -> Tensor:
    """Load landmarks from various sources."""
    if isinstance(source, Tensor):
        return source.float()
    path = Path(source)
    if path.suffix == ".npy":
        arr = np.load(path)
        return torch.as_tensor(arr, dtype=torch.float32)
    if path.suffix in (".pt", ".pth"):
        data = torch.load(path, weights_only=True)
        if isinstance(data, Tensor):
            return data.float()
        msg = f"Expected a Tensor in {path}, got {type(data).__name__}"
        raise TypeError(msg)
    msg = f"Unsupported landmarks file extension: {path.suffix}"
    raise ValueError(msg)


def _apply_histogram_standardization(
    tensor: Tensor,
    landmarks: Tensor,
    cutoff: tuple[float, float],
) -> Tensor:
    """Apply piecewise-linear histogram mapping to a 4-D tensor.

    Args:
        tensor: ``(C, I, J, K)`` image tensor.
        landmarks: 1-D standard-space landmarks.
        cutoff: ``(low, high)`` quantile cutoff.

    Returns:
        Standardized ``(C, I, J, K)`` tensor.
    """
    quantiles = _build_quantiles(cutoff)
    percentiles = [100.0 * q for q in quantiles]
    num_landmarks = len(landmarks)
    if num_landmarks != len(percentiles):
        msg = (
            f"Number of landmarks ({num_landmarks}) does not match "
            f"the number of quantile positions ({len(percentiles)}). "
            "Ensure the same quantile scheme was used for training."
        )
        raise ValueError(msg)

    data = tensor.float()
    flat = data.reshape(-1)

    # Compute input percentiles.
    pv = np.percentile(flat.cpu().numpy(), percentiles)
    input_landmarks = torch.as_tensor(pv, dtype=torch.float32, device=data.device)

    # Build piecewise-linear mapping.
    diff_landmarks = torch.diff(landmarks.to(data.device))
    diff_input = torch.diff(input_landmarks)

    # Handle flat segments (constant regions).
    eps = 1e-5
    diff_input = torch.where(
        diff_input.abs() < eps,
        torch.tensor(float("inf"), device=data.device),
        diff_input,
    )

    slopes = diff_landmarks / diff_input
    intercepts = landmarks[:-1].to(data.device) - slopes * input_landmarks[:-1]

    # Digitize: find which segment each voxel falls into.
    bin_edges = input_landmarks[1:-1]
    bin_ids = torch.bucketize(flat, bin_edges, right=False)

    result = slopes[bin_ids] * flat + intercepts[bin_ids]
    return result.reshape(data.shape)
