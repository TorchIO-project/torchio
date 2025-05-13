from __future__ import annotations

import torch

from ...types import TypeRangeFloat
from ...types import TypeSextetFloat
from ...types import TypeTripletFloat
from ..transform import Transform


class RandomTransform(Transform):
    """Base class for stochastic augmentation transforms.

    Args:
        **kwargs: See :class:`~torchio.transforms.Transform` for additional
            keyword arguments.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def parse_degrees(
        self,
        degrees: TypeRangeFloat,
    ) -> tuple[float, float]:
        return self._parse_range(degrees, 'degrees')

    def parse_translation(
        self,
        translation: TypeRangeFloat,
    ) -> tuple[float, float]:
        return self._parse_range(translation, 'translation')

    @staticmethod
    def sample_uniform(a: float, b: float) -> float:
        return torch.FloatTensor(1).uniform_(a, b).item()

    @staticmethod
    def _get_random_seed() -> int:
        """Generate a random seed.

        Returns:
            A random seed as an int.
        """
        return int(torch.randint(0, 2**31, (1,)).item())

    @staticmethod
    def sample_uniform_sextet(params: TypeSextetFloat) -> TypeTripletFloat:
        results = []
        for a, b in zip(params[::2], params[1::2]):
            results.append(RandomTransform.sample_uniform(a, b))
        sx, sy, sz = results
        return sx, sy, sz
