"""Spike: simulate k-space spike (herringbone) artifacts."""

from __future__ import annotations

from typing import Any

import torch
from einops import rearrange
from torch import Tensor

from ...data.batch import SubjectsBatch
from ..parameter_range import to_nonneg_range
from ..parameter_range import to_range
from ..transform import IntensityTransform


class Spike(IntensityTransform):
    r"""Add random MRI spike artifacts.

    Also known as
    [herringbone artifact](https://radiopaedia.org/articles/herringbone-artifact),
    crisscross artifact, or corduroy artifact.  Spikes in k-space
    create stripes in image space.

    The artifact is simulated by adding point impulses to the Fourier
    spectrum of the image.  All operations use `torch.fft` and run
    on GPU.

    Args:
        num_spikes: Number of spikes.  A scalar $n$ is deterministic;
            a 2-tuple $(a, b)$ samples
            $n \sim \mathcal{U}(a, b) \cap \mathbb{N}$.
        intensity: Ratio between the spike amplitude and the spectrum
            maximum.  A scalar is deterministic; a 2-tuple $(a, b)$
            means $r \sim \mathcal{U}(a, b)$.
            The default `intensity=0` is a no-op (and warns).
        **kwargs: See [`Transform`][torchio.Transform].

    Note:
        Execution time does not depend on the number of spikes.

    Examples:
        >>> import torchio as tio
        >>> transform = tio.Spike(intensity=2.0)
        >>> transform = tio.Spike(num_spikes=3, intensity=2.0)
    """

    def __init__(
        self,
        *,
        num_spikes: int | tuple[int, int] = 1,
        intensity: float | tuple[float, float] = 0.0,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.num_spikes = to_nonneg_range(num_spikes)
        self.intensity = to_range(intensity)
        self._warn_if_noop(
            is_noop=self.intensity.is_constant(0.0) or self.num_spikes.is_constant(0.0),
            hint="intensity=(1, 3)",
        )

    def make_params(self, batch: SubjectsBatch) -> dict[str, Any]:
        """Sample the number and positions of spikes (per element when batched)."""
        n = self._resolve_n(batch)
        if n is None:
            num_spikes = max(1, round(self.num_spikes.sample_1d()))
            positions = torch.rand(num_spikes, 3).tolist()
            intensity = self.intensity.sample_1d()
            return {
                "positions": positions,
                "intensity": intensity,
            }
        keep = self._keep_mask(batch, n)
        positions_list: list[list[list[float]]] = []
        intensity_list: list[float] = []
        keep_values = [True] * n if keep is None else keep.tolist()
        for should_keep in keep_values:
            if not should_keep:
                positions_list.append([])
                intensity_list.append(0.0)
                continue
            num_spikes = max(1, round(self.num_spikes.sample_1d()))
            positions_list.append(torch.rand(num_spikes, 3).tolist())
            intensity_list.append(self.intensity.sample_1d())
        params = {
            "positions": positions_list,
            "intensity": intensity_list,
        }
        self._tag_batched(params, batch, n, keep, ["positions", "intensity"])
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
        """Add spike artifacts to each selected image."""
        per_instance = self._is_per_instance_params(params)
        for _name, img_batch in self._get_images(batch).items():
            if per_instance:
                img_batch.data = _add_spikes_per_instance(
                    img_batch.data,
                    params["positions"],
                    params["intensity"],
                )
            else:
                img_batch.data = _add_spikes(
                    img_batch.data,
                    params["positions"],
                    params["intensity"],
                )
        return batch


def _add_spikes(
    data: Tensor,
    positions: list[list[float]],
    intensity: float,
) -> Tensor:
    """Add point spikes to the k-space of a 5D tensor.

    Args:
        data: `(B, C, I, J, K)` image tensor.
        positions: List of `[pi, pj, pk]` in `[0, 1)` range.
        intensity: Ratio between the spike amplitude and the spectrum
            maximum.

    Returns:
        Corrupted `(B, C, I, J, K)` tensor.
    """
    if intensity == 0 or not positions:
        return data

    result = data.float()
    shape = result.shape[2:]  # (I, J, K)

    # FFT over spatial dims.
    spectrum = torch.fft.fftshift(
        torch.fft.fftn(result, dim=(-3, -2, -1)),
        dim=(-3, -2, -1),
    )
    # Peak per (B, C): shape (B, C, 1, 1, 1) for broadcasting.
    peak = spectrum.abs().amax(dim=(-3, -2, -1), keepdim=True)

    for pos in positions:
        idx = [int(p * s) % s for p, s in zip(pos, shape, strict=True)]
        spectrum[:, :, idx[0], idx[1], idx[2]] += peak[..., 0, 0, 0] * intensity

    result = torch.fft.ifftn(
        torch.fft.ifftshift(spectrum, dim=(-3, -2, -1)),
        dim=(-3, -2, -1),
    ).real
    return result.to(data.dtype)


def _add_spikes_per_instance(
    data: Tensor,
    positions: list[list[list[float]]],
    intensities: list[float],
) -> Tensor:
    """Add independently sampled point spikes to a batched 5D tensor.

    Args:
        data: `(B, C, I, J, K)` image tensor.
        positions: Per-element lists of `[pi, pj, pk]` positions in `[0, 1)`.
        intensities: Per-element spike amplitude ratios.

    Returns:
        Corrupted `(B, C, I, J, K)` tensor, with inactive elements unchanged.
    """
    active = torch.as_tensor(
        [
            bool(batch_positions) and batch_intensity != 0
            for batch_positions, batch_intensity in zip(
                positions,
                intensities,
                strict=True,
            )
        ],
        device=data.device,
    )
    if not active.any().item():
        return data

    result = data.float()
    shape = result.shape[2:]  # (I, J, K)

    spectrum = torch.fft.fftshift(
        torch.fft.fftn(result, dim=(-3, -2, -1)),
        dim=(-3, -2, -1),
    )
    peak = spectrum.abs().amax(dim=(-3, -2, -1), keepdim=True)

    for batch_index, (batch_positions, batch_intensity) in enumerate(
        zip(
            positions,
            intensities,
            strict=True,
        )
    ):
        if not batch_positions or batch_intensity == 0:
            continue
        for pos in batch_positions:
            idx = [int(p * s) % s for p, s in zip(pos, shape, strict=True)]
            spectrum[batch_index, :, idx[0], idx[1], idx[2]] += (
                peak[batch_index, :, 0, 0, 0] * batch_intensity
            )

    transformed = torch.fft.ifftn(
        torch.fft.ifftshift(spectrum, dim=(-3, -2, -1)),
        dim=(-3, -2, -1),
    ).real.to(data.dtype)
    active = rearrange(active, "b -> b 1 1 1 1")
    return torch.where(active, transformed, data)
