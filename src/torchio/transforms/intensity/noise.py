"""Noise transform: add Gaussian noise to intensity images."""

from __future__ import annotations

from typing import Any

import torch
from einops import rearrange
from torch import Tensor
from torch.distributions import Distribution

from ...data.batch import SubjectsBatch
from ..parameter_range import to_nonneg_range
from ..parameter_range import to_range
from ..transform import IntensityTransform


class Noise(IntensityTransform):
    r"""Add Gaussian or Rician noise with random parameters.

    Add noise sampled from a normal distribution with random
    parameters. When `rician=True`, the magnitude of complex
    Gaussian noise is used instead, producing
    [Rician-distributed](https://en.wikipedia.org/wiki/Rice_distribution)
    noise typical of MRI acquisitions:

    $$I_{\text{noisy}} = \sqrt{(I + n_1)^2 + n_2^2}$$

    where $n_1, n_2 \sim \mathcal{N}(\mu, \sigma^2)$ independently.

    Args:
        mean: Mean $\mu$ of the Gaussian distribution from which the
            noise is sampled. If two values $(a, b)$ are provided,
            then $\mu \sim \mathcal{U}(a, b)$.
            If only one value $d$ is provided, $\mu = d$
            (deterministic). A `torch.distributions.Distribution`
            may also be passed for custom sampling.
        std: Standard deviation $\sigma$ of the Gaussian distribution
            from which the noise is sampled. If two values $(a, b)$
            are provided, then $\sigma \sim \mathcal{U}(a, b)$.
            If only one value $d$ is provided, $\sigma = d$
            (deterministic). Must be non-negative.
            A `torch.distributions.Distribution` may also be passed.
        rician: If `True`, add Rician noise instead of Gaussian.
        **kwargs: See [`Transform`][torchio.Transform] for additional
            keyword arguments.

    Examples:
        >>> import torchio as tio
        >>> # Gaussian noise (default)
        >>> transform = tio.Noise(std=0.1)
        >>> # Rician noise (typical for MRI)
        >>> transform = tio.Noise(std=0.1, rician=True)
        >>> # Random std from a uniform range
        >>> transform = tio.Noise(std=(0.05, 0.2))
        >>> # Custom distribution for std
        >>> from torch.distributions import LogNormal
        >>> transform = tio.Noise(std=LogNormal(loc=-2, scale=0.5))
    """

    def __init__(
        self,
        *,
        mean: float | tuple[float, float] | Distribution = 0.0,
        std: float | tuple[float, float] | Distribution = 0.25,
        rician: bool = False,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.mean = to_range(mean)
        self.std = to_nonneg_range(std)
        self.rician = rician

    def make_params(self, batch: SubjectsBatch) -> dict[str, Any]:
        seed = int(torch.randint(0, 2**31, (1,)).item())
        n = self._resolve_n(batch)
        keep = self._keep_mask(batch, n)
        # Identity for a gated-out element is zero mean and zero std.
        mean = self._mask_identity(self.mean.sample_1d(n), keep, identity=0.0)
        std = self._mask_identity(self.std.sample_1d(n), keep, identity=0.0)
        params = {
            "mean": self._serialize_param(mean),
            "std": self._serialize_param(std),
            "seed": seed,
            "rician": self.rician,
        }
        self._tag_batched(params, batch, n, keep, ["mean", "std"])
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
        mean = params["mean"]
        std = params["std"]
        seed = params["seed"]
        rician = params.get("rician", False)
        keep = params.get("_keep")
        generator = torch.Generator(device="cpu")
        generator.manual_seed(seed)
        for _name, img_batch in self._get_images(batch).items():
            data = img_batch.data
            mean_b = _broadcast(mean, data)
            std_b = _broadcast(std, data)
            noise = _sample_noise(data, mean_b, std_b, generator)
            if rician:
                noise_2 = _sample_noise(data, mean_b, std_b, generator)
                transformed = torch.sqrt((data + noise) ** 2 + noise_2**2)
            else:
                transformed = data + noise
            # The Rician map is non-linear, so zero mean/std is not a true
            # no-op (it returns |data|). Restore gated-out elements explicitly.
            img_batch.data = _restore_gated_out(transformed, data, keep)
        return batch


def _restore_gated_out(
    transformed: Tensor,
    original: Tensor,
    keep: list[bool] | None,
) -> Tensor:
    """Keep the original data for elements excluded by per-element gating.

    Args:
        transformed: The augmented `(B, C, I, J, K)` tensor.
        original: The input `(B, C, I, J, K)` tensor.
        keep: Per-element keep mask, or `None` when gating is inactive.

    Returns:
        A tensor equal to *transformed* for kept elements and to
        *original* for gated-out elements.
    """
    if keep is None:
        return transformed
    keep_mask = torch.tensor(keep, dtype=torch.bool, device=transformed.device)
    keep_mask = rearrange(keep_mask, "b -> b 1 1 1 1")
    return torch.where(keep_mask, transformed, original)


def _broadcast(value: float | list[float], data: Tensor) -> float | Tensor:
    """Broadcast a scalar or per-element value over a `(B, C, I, J, K)` tensor.

    Args:
        value: A scalar (batch-shared) or a per-element list.
        data: The tensor the value will be combined with.

    Returns:
        A Python float for the scalar case, or a `(B, 1, 1, 1, 1)`
        tensor for the per-element case.
    """
    if isinstance(value, list):
        tensor = torch.tensor(value, dtype=torch.float32, device=data.device)
        return rearrange(tensor, "b -> b 1 1 1 1")
    return value


def _sample_noise(
    data: Tensor,
    mean: float | Tensor,
    std: float | Tensor,
    generator: torch.Generator,
) -> Tensor:
    """Draw `mean + std * N(0, 1)` noise shaped like *data*.

    Sampling a standard normal and scaling keeps reproducibility while
    supporting per-element `mean`/`std` via broadcasting.
    """
    base = torch.randn(data.shape, generator=generator).to(data.device)
    return mean + std * base
