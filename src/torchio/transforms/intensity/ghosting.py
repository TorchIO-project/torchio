"""Ghosting: simulate MRI ghosting artifacts along the phase-encode axis."""

from __future__ import annotations

from typing import Any

import torch
from torch import Tensor

from ...data.batch import SubjectsBatch
from ..parameter_range import to_nonneg_range
from ..transform import IntensityTransform


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
        num_ghosts: Number of ghost replicas.  A scalar $n$ is
            deterministic; a 2-tuple $(a, b)$ samples
            $n \sim \mathcal{U}(a, b) \cap \mathbb{N}$.
        axes: Spatial axes along which ghosts may appear.  One is
            chosen at random per application.
        intensity: Artifact strength relative to the k-space maximum.
            A scalar is deterministic; a 2-tuple $(a, b)$ means
            $s \sim \mathcal{U}(a, b)$.
            The default `intensity=0` is a no-op (and warns).
        restore: Fraction of central k-space to restore after
            zeroing.  `None` restores only the single central
            slice.
        **kwargs: See [`Transform`][torchio.Transform].

    Note:
        Execution time does not depend on the number of ghosts.

    Examples:
        >>> import torchio as tio
        >>> transform = tio.Ghosting(intensity=0.8)
        >>> transform = tio.Ghosting(num_ghosts=6, intensity=0.8)
    """

    def __init__(
        self,
        *,
        num_ghosts: int | tuple[int, int] = 4,
        axes: tuple[int, ...] = (0, 1, 2),
        intensity: float | tuple[float, float] = 0.0,
        restore: float | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.num_ghosts = to_nonneg_range(num_ghosts)
        self.axes = axes
        self.intensity = to_nonneg_range(intensity)
        self.restore = restore
        self._warn_if_noop(
            is_noop=self.intensity.is_constant(0.0) or self.num_ghosts.is_constant(0.0),
            hint="intensity=(0.5, 1)",
        )

    def make_params(self, batch: SubjectsBatch) -> dict[str, Any]:
        """Sample ghosting parameters (per element when batched)."""
        restore = self.restore if self.restore is not None else 0.0
        n = self._resolve_n(batch)
        if n is None:
            num_ghosts = max(1, round(self.num_ghosts.sample_1d()))
            axis = self.axes[int(torch.randint(len(self.axes), (1,)).item())]
            return {
                "num_ghosts": num_ghosts,
                "axis": axis,
                "intensity": self.intensity.sample_1d(),
                "restore": restore,
            }
        keep = self._keep_mask(batch, n)
        num_ghosts_list: list[int] = []
        axis_list: list[int] = []
        intensity_list: list[float] = []
        for index in range(n):
            if keep is not None and not keep[index]:
                num_ghosts_list.append(0)
                axis_list.append(self.axes[0])
                intensity_list.append(0.0)
                continue
            num_ghosts_list.append(max(1, round(self.num_ghosts.sample_1d())))
            axis_list.append(self.axes[int(torch.randint(len(self.axes), (1,)).item())])
            intensity_list.append(self.intensity.sample_1d())
        params = {
            "num_ghosts": num_ghosts_list,
            "axis": axis_list,
            "intensity": intensity_list,
            "restore": restore,
        }
        self._tag_batched(
            params,
            batch,
            n,
            keep,
            ["num_ghosts", "axis", "intensity"],
        )
        return params

    @property
    def supports_per_instance_params(self) -> bool:
        return True

    @property
    def supports_per_instance_p(self) -> bool:
        return True

    def apply_transform(
        self,
        batch: SubjectsBatch,
        params: dict[str, Any],
    ) -> SubjectsBatch:
        """Add ghosting artifacts to each selected image."""
        per_instance = self._is_per_instance_params(params)
        restore = params["restore"]
        for _name, img_batch in self._get_images(batch).items():
            if per_instance:
                data = img_batch.data
                outputs = [
                    _add_ghosting(
                        data[index : index + 1],
                        num_ghosts=params["num_ghosts"][index],
                        axis=params["axis"][index],
                        intensity=params["intensity"][index],
                        restore=restore,
                    )
                    for index in range(data.shape[0])
                ]
                img_batch.data = torch.cat(outputs, dim=0)
            else:
                img_batch.data = _add_ghosting(
                    img_batch.data,
                    num_ghosts=params["num_ghosts"],
                    axis=params["axis"],
                    intensity=params["intensity"],
                    restore=restore,
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
        data: `(B, C, I, J, K)` image tensor.
        num_ghosts: Number of ghost replicas.
        axis: Spatial axis (0, 1, or 2) for the phase-encode direction.
        intensity: Artifact strength (0 = none, 1 = strong).
        restore: Fraction of central k-space to restore.

    Returns:
        Corrupted `(B, C, I, J, K)` tensor.
    """
    if not num_ghosts or intensity == 0:
        return data

    result = data.float()
    # FFT over spatial dims only: dims 2, 3, 4.
    spectrum = torch.fft.fftshift(
        torch.fft.fftn(result, dim=(-3, -2, -1)),
        dim=(-3, -2, -1),
    )

    # Build mask along the chosen axis.
    fft_dim = axis + 2  # map spatial axis to tensor dim
    size = result.shape[fft_dim]
    mask = torch.ones(size, device=data.device)
    step = max(size // num_ghosts, 1)
    mask[::step] = 1 - intensity

    # Reshape mask for broadcasting: (1, 1, 1, 1, 1) with size at fft_dim.
    shape = [1] * 5
    shape[fft_dim] = size
    mask = mask.reshape(*shape)
    corrupted = spectrum * mask

    # Restore the center of k-space.
    if restore > 0:
        mid = size // 2
        half_restore = max(int(size * restore / 2), 1)
        lo, hi = mid - half_restore, mid + half_restore
        slices: list[slice] = [slice(None)] * 5
        slices[fft_dim] = slice(lo, hi)
        corrupted[tuple(slices)] = spectrum[tuple(slices)]

    result = torch.fft.ifftn(
        torch.fft.ifftshift(corrupted, dim=(-3, -2, -1)),
        dim=(-3, -2, -1),
    ).real
    return result
