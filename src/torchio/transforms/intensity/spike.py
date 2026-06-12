"""Spike: simulate k-space spike (herringbone) artifacts."""

from __future__ import annotations

from typing import Any

import torch
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
            seed = int(torch.randint(0, 2**31, (1,)).item())
            return {
                "positions": positions,
                "intensity": intensity,
                "seed": seed,
            }
        keep = self._keep_mask(batch, n)
        positions_list: list[list[list[float]]] = []
        intensity_list: list[float] = []
        for index in range(n):
            if keep is not None and not keep[index]:
                positions_list.append([])
                intensity_list.append(0.0)
                continue
            num_spikes = max(1, round(self.num_spikes.sample_1d()))
            positions_list.append(torch.rand(num_spikes, 3).tolist())
            intensity_list.append(self.intensity.sample_1d())
        seed = int(torch.randint(0, 2**31, (1,)).item())
        params = {
            "positions": positions_list,
            "intensity": intensity_list,
            "seed": seed,
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
                data = img_batch.data
                outputs = [
                    _add_spikes(
                        data[index : index + 1],
                        params["positions"][index],
                        params["intensity"][index],
                    )
                    for index in range(data.shape[0])
                ]
                img_batch.data = torch.cat(outputs, dim=0)
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
    return result
