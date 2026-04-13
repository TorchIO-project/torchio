"""Ghosting: simulate MRI ghosting artifacts along the phase-encode axis."""

from __future__ import annotations

from typing import Any

import torch
from torch import Tensor

from ..data.batch import SubjectsBatch
from .parameter_range import to_nonneg_range
from .transform import IntensityTransform


class Ghosting(IntensityTransform):
    r"""Add random MRI ghosting artifacts.

    Discrete "ghost" replicas of the imaged anatomy appear along the
    phase-encode direction when signal intensity varies periodically
    during acquisition.  Common causes include pulsatile blood flow,
    cardiac motion, and respiratory motion.
    (See [mriquestions.com](http://mriquestions.com/why-discrete-ghosts.html).)

    The artifact is simulated by zeroing periodic planes in k-space
    along a randomly chosen axis, then restoring a fraction of the
    central k-space to avoid extreme artifacts.

    Args:
        num_ghosts: Number of ghost replicas.  A 2-tuple $(a, b)$
            samples $n \sim \mathcal{U}(a, b) \cap \mathbb{N}$.
        axes: Spatial axes along which ghosts may appear.  One is
            chosen at random per application.
        intensity: Artifact strength relative to the k-space maximum.
            A 2-tuple $(a, b)$ means
            $s \sim \mathcal{U}(a, b)$.
        restore: Fraction of central k-space to restore after
            zeroing.  ``None`` restores only the single central
            slice.
        **kwargs: See [`Transform`][torchio.Transform].

    Note:
        Execution time does not depend on the number of ghosts.

    Examples:
        >>> import torchio as tio
        >>> transform = tio.Ghosting()
        >>> transform = tio.Ghosting(num_ghosts=6, intensity=0.8)
    """

    def __init__(
        self,
        *,
        num_ghosts: int | tuple[int, int] = (4, 10),
        axes: tuple[int, ...] = (0, 1, 2),
        intensity: float | tuple[float, float] = (0.5, 1.0),
        restore: float | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.num_ghosts = to_nonneg_range(num_ghosts)
        self.axes = axes
        self.intensity = to_nonneg_range(intensity)
        self.restore = restore

    def make_params(self, batch: SubjectsBatch) -> dict[str, Any]:
        """Sample ghosting parameters."""
        n = max(1, round(self.num_ghosts.sample_1d()))
        axis = self.axes[int(torch.randint(len(self.axes), (1,)).item())]
        restore = self.restore
        if restore is None:
            restore = 0.0
        return {
            "num_ghosts": n,
            "axis": axis,
            "intensity": self.intensity.sample_1d(),
            "restore": restore,
        }

    def apply_transform(
        self,
        batch: SubjectsBatch,
        params: dict[str, Any],
    ) -> SubjectsBatch:
        """Add ghosting artifacts to each selected image."""
        for _name, img_batch in self._get_images(batch).items():
            img_batch.data = _add_ghosting(
                img_batch.data,
                num_ghosts=params["num_ghosts"],
                axis=params["axis"],
                intensity=params["intensity"],
                restore=params["restore"],
            )
        return batch


def _add_ghosting(
    data: Tensor,
    *,
    num_ghosts: int,
    axis: int,
    intensity: float,
    restore: float,
) -> Tensor:
    """Add ghosting artifacts to a 5D tensor via k-space manipulation.

    Args:
        data: ``(B, C, I, J, K)`` image tensor.
        num_ghosts: Number of ghost replicas.
        axis: Spatial axis (0, 1, or 2) for the phase-encode direction.
        intensity: Artifact strength (0 = none, 1 = strong).
        restore: Fraction of central k-space to restore.

    Returns:
        Corrupted ``(B, C, I, J, K)`` tensor.
    """
    if not num_ghosts or intensity == 0:
        return data

    result = data.float()

    for b in range(result.shape[0]):
        for c in range(result.shape[1]):
            channel = result[b, c]
            spectrum = torch.fft.fftshift(torch.fft.fftn(channel))

            size = spectrum.shape[axis]
            # Zero every num_ghosts-th plane along the axis.
            mask = torch.ones(size, device=data.device)
            step = max(size // num_ghosts, 1)
            mask[::step] = 1 - intensity

            # Reshape mask for broadcasting.
            shape = [1, 1, 1]
            shape[axis] = size
            mask = mask.reshape(*shape)
            spectrum = spectrum * mask

            # Restore the center of k-space.
            if restore > 0:
                mid = size // 2
                half_restore = max(int(size * restore / 2), 1)
                lo, hi = mid - half_restore, mid + half_restore
                orig_spectrum = torch.fft.fftshift(torch.fft.fftn(channel))
                slices = [slice(None)] * 3
                slices[axis] = slice(lo, hi)
                spectrum[tuple(slices)] = orig_spectrum[tuple(slices)]

            channel_back = torch.fft.ifftn(torch.fft.ifftshift(spectrum))
            result[b, c] = channel_back.real

    return result
