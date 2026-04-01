"""Noise transform: add Gaussian noise to intensity images."""

from __future__ import annotations

from typing import Any

import torch
from torch.distributions import Distribution

from ..data.batch import SubjectsBatch
from .parameter_range import to_nonneg_range
from .parameter_range import to_range
from .transform import IntensityTransform


class Noise(IntensityTransform):
    r"""Add Gaussian or Rician noise with random parameters.

    Add noise sampled from a normal distribution with random
    parameters. When ``rician=True``, the magnitude of complex
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
            (deterministic). A ``torch.distributions.Distribution``
            may also be passed for custom sampling.
        std: Standard deviation $\sigma$ of the Gaussian distribution
            from which the noise is sampled. If two values $(a, b)$
            are provided, then $\sigma \sim \mathcal{U}(a, b)$.
            If only one value $d$ is provided, $\sigma = d$
            (deterministic). Must be non-negative.
            A ``torch.distributions.Distribution`` may also be passed.
        rician: If ``True``, add Rician noise instead of Gaussian.
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
        return {
            "mean": self.mean.sample_1d(),
            "std": self.std.sample_1d(),
            "seed": seed,
            "rician": self.rician,
        }

    def apply_transform(
        self,
        batch: SubjectsBatch,
        params: dict[str, Any],
    ) -> SubjectsBatch:
        mean = params["mean"]
        std = params["std"]
        seed = params["seed"]
        rician = params.get("rician", False)
        generator = torch.Generator(device="cpu")
        generator.manual_seed(seed)
        for _name, img_batch in self._get_images(batch).items():
            noise = torch.normal(
                mean=mean,
                std=std,
                size=img_batch.data.shape,
                generator=generator,
            ).to(img_batch.data.device)
            if rician:
                noise_2 = torch.normal(
                    mean=mean,
                    std=std,
                    size=img_batch.data.shape,
                    generator=generator,
                ).to(img_batch.data.device)
                img_batch.data = torch.sqrt(
                    (img_batch.data + noise) ** 2 + noise_2**2,
                )
            else:
                img_batch.data = img_batch.data + noise
        return batch
