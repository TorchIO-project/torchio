"""Noise transform: add Gaussian noise to intensity images."""

from __future__ import annotations

from typing import Any

import torch

from ..data.subject import Subject
from .parameter_range import to_nonneg_range
from .parameter_range import to_range
from .transform import IntensityTransform


class Noise(IntensityTransform):
    """Add Gaussian noise to intensity images.

    Pure ``torch`` operation — GPU-compatible and differentiable.

    Parameters accept a scalar (deterministic) or a ``(lo, hi)`` tuple
    (sampled uniformly each call).

    Args:
        mean: Mean of the noise distribution.
        std: Standard deviation of the noise distribution.
            Must be non-negative.

    Examples:
        >>> tio.Noise(std=0.1)                 # fixed std
        >>> tio.Noise(std=(0.05, 0.2))         # random std each call
        >>> tio.Noise(mean=(-0.1, 0.1), std=(0.05, 0.2))  # both random
    """

    def __init__(
        self,
        *,
        mean: float | tuple[float, float] = 0.0,
        std: float | tuple[float, float] = 0.25,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.mean = to_range(mean)
        self.std = to_nonneg_range(std)

    def make_params(self, subject: Subject) -> dict[str, Any]:
        seed = int(torch.randint(0, 2**31, (1,)).item())
        return {
            "mean": self.mean.sample_1d(),
            "std": self.std.sample_1d(),
            "seed": seed,
            "_replay": False,
        }

    def apply_transform(self, subject: Subject, params: dict[str, Any]) -> Subject:
        mean = params["mean"]
        std = params["std"]
        seed = params["seed"]
        replay = params.get("_replay", True)
        for _name, image in self._get_images(subject).items():
            if replay:
                generator = torch.Generator(device="cpu")
                generator.manual_seed(seed)
                noise = torch.normal(
                    mean=mean,
                    std=std,
                    size=image.data.shape,
                    generator=generator,
                ).to(image.data.device)
            else:
                noise = torch.normal(
                    mean=mean,
                    std=std,
                    size=image.data.shape,
                    device=image.data.device,
                    dtype=image.data.dtype,
                )
            image.set_data(image.data + noise)
        return subject
