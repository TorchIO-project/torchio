"""Gamma: randomly change contrast via power-law transform."""

from __future__ import annotations

import math
from typing import Any

from ..data.batch import SubjectsBatch
from .parameter_range import to_range
from .transform import IntensityTransform


class Gamma(IntensityTransform):
    r"""Change image contrast by raising values to the power $\gamma$.

    The exponent is computed as $\gamma = e^{\beta}$, where $\beta$ is
    sampled from the specified range.  Positive $\beta$ increases
    contrast (gamma expansion), negative $\beta$ decreases it (gamma
    compression).  See the
    [Gamma correction](https://en.wikipedia.org/wiki/Gamma_correction)
    Wikipedia entry for more information.

    Note:
        Fractional exponentiation of negative values is not
        well-defined for non-complex numbers.  If negative values are
        found in the input image $I$, the applied transform is
        $\text{sign}(I) \cdot |I|^{\gamma}$ instead of the usual
        $I^{\gamma}$.  Use [`Normalize`][torchio.Normalize] to ensure
        all values are positive if needed.

    Args:
        log_gamma: Range for $\beta$ in $\gamma = e^{\beta}$.
            A scalar $d$ means $\beta \sim \mathcal{U}(-d, d)$.
            A 2-tuple $(a, b)$ means $\beta \sim \mathcal{U}(a, b)$.
            A ``Choice`` or ``Distribution`` may also be passed.
        **kwargs: See [`Transform`][torchio.Transform].

    Examples:
        >>> import torchio as tio
        >>> transform = tio.Gamma()
        >>> transform = tio.Gamma(log_gamma=0.5)
        >>> transform = tio.Gamma(log_gamma=(-0.3, 0.3))
    """

    def __init__(
        self,
        *,
        log_gamma: float | tuple[float, float] = (-0.3, 0.3),
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.log_gamma = to_range(log_gamma)

    def make_params(self, batch: SubjectsBatch) -> dict[str, Any]:
        """Sample the log-gamma value."""
        return {"log_gamma": self.log_gamma.sample_1d()}

    def apply_transform(
        self,
        batch: SubjectsBatch,
        params: dict[str, Any],
    ) -> SubjectsBatch:
        """Raise each intensity image to the power gamma."""
        gamma = math.exp(params["log_gamma"])
        for _name, img_batch in self._get_images(batch).items():
            data = img_batch.data
            img_batch.data = data.sign() * data.abs().pow(gamma)
        return batch

    @property
    def invertible(self) -> bool:
        """Whether this transform can be inverted."""
        return True

    def inverse(self, params: dict[str, Any]) -> _GammaInverse:
        """Invert by applying 1/gamma."""
        return _GammaInverse(log_gamma=params["log_gamma"], copy=False)


class _GammaInverse(IntensityTransform):
    """Inverse of Gamma for history replay."""

    def __init__(self, *, log_gamma: float, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._log_gamma = log_gamma

    def make_params(self, batch: SubjectsBatch) -> dict[str, Any]:
        return {}

    def apply_transform(
        self,
        batch: SubjectsBatch,
        params: dict[str, Any],
    ) -> SubjectsBatch:
        inv_gamma = math.exp(-self._log_gamma)
        for _name, img_batch in self._get_images(batch).items():
            data = img_batch.data
            img_batch.data = data.sign() * data.abs().pow(inv_gamma)
        return batch
