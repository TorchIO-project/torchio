"""Noise transform: add Gaussian noise to intensity images."""

from __future__ import annotations

from typing import Any

import attrs
import torch

from ..data.subject import Subject
from .transform import IntensityTransform


@attrs.define(slots=False, eq=False, kw_only=True)
class Noise(IntensityTransform):
    """Add Gaussian noise to intensity images.

    Pure ``torch`` operation — GPU-compatible and differentiable.

    Args:
        mean: Mean of the noise distribution.
        std: Standard deviation of the noise distribution.
            Must be non-negative.

    Examples:
        >>> noisy = tio.Noise(std=0.1)(subject)
        >>> noisy = tio.Noise(mean=0, std=0.05, p=0.5)(subject)
    """

    mean: float = 0.0
    std: float = attrs.field(default=0.25, validator=attrs.validators.ge(0))

    def make_params(self, subject: Subject) -> dict[str, Any]:
        seed = int(torch.randint(0, 2**31, (1,)).item())
        return {"mean": self.mean, "std": self.std, "seed": seed}

    def apply(self, subject: Subject, params: dict[str, Any]) -> Subject:
        mean = params["mean"]
        std = params["std"]
        seed = params["seed"]
        generator = torch.Generator(device="cpu")
        generator.manual_seed(seed)
        for _name, image in self._get_images(subject).items():
            noise_cpu = torch.normal(
                mean=mean,
                std=std,
                size=image.data.shape,
                generator=generator,
            )
            noise = noise_cpu.to(image.data)
            image.set_data(image.data + noise)
        return subject
