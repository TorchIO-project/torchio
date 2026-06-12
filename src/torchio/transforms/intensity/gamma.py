"""Gamma: randomly change contrast via power-law transform."""

from __future__ import annotations

import math
from typing import Any

import torch
from einops import rearrange
from torch import Tensor

from ...data.batch import SubjectsBatch
from ..parameter_range import to_range
from ..transform import IntensityTransform


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
            A scalar $x$ means $\beta = x$ (deterministic).
            A 2-tuple $(a, b)$ means $\beta \sim \mathcal{U}(a, b)$.
            A `Choice` or `Distribution` may also be passed.
            The default `log_gamma=0` is a no-op (and warns).
        **kwargs: See [`Transform`][torchio.Transform].

    Examples:
        >>> import torchio as tio
        >>> transform = tio.Gamma(log_gamma=0.5)
        >>> transform = tio.Gamma(log_gamma=(-0.3, 0.3))
    """

    def __init__(
        self,
        *,
        log_gamma: float | tuple[float, float] = 0.0,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.log_gamma = to_range(log_gamma)
        self._warn_if_noop(
            is_noop=self.log_gamma.is_constant(0.0),
            hint="log_gamma=(-0.3, 0.3)",
        )

    def make_params(self, batch: SubjectsBatch) -> dict[str, Any]:
        """Sample the log-gamma value (per element when per-instance)."""
        n = self._resolve_n(batch)
        keep = self._keep_mask(batch, n)
        log_gamma = self.log_gamma.sample_1d(n)
        log_gamma = self._mask_identity(log_gamma, keep, identity=0.0)
        params = {"log_gamma": self._serialize_param(log_gamma)}
        self._tag_batched(params, batch, n, keep, ["log_gamma"])
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
        """Raise each intensity image to the power gamma."""
        log_gamma = params["log_gamma"]
        for _name, img_batch in self._get_images(batch).items():
            gamma = _gamma_from_log(log_gamma, img_batch.data)
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


def _gamma_from_log(
    log_gamma: float | list[float],
    data: Tensor,
) -> float | Tensor:
    """Compute the gamma exponent, broadcasting over the batch if needed.

    Args:
        log_gamma: A scalar (batch-shared) or a per-element list.
        data: The `(B, C, I, J, K)` tensor the exponent applies to.

    Returns:
        A Python float for the scalar case, or a `(B, 1, 1, 1, 1)`
        tensor for the per-element case.
    """
    if isinstance(log_gamma, list):
        values = torch.tensor(log_gamma, dtype=torch.float32, device=data.device)
        return rearrange(torch.exp(values), "b -> b 1 1 1 1")
    return math.exp(log_gamma)


class _GammaInverse(IntensityTransform):
    """Inverse of Gamma for history replay."""

    def __init__(self, *, log_gamma: float | list[float], **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._log_gamma = log_gamma

    def make_params(self, batch: SubjectsBatch) -> dict[str, Any]:
        return {}

    def apply_transform(
        self,
        batch: SubjectsBatch,
        params: dict[str, Any],
    ) -> SubjectsBatch:
        for _name, img_batch in self._get_images(batch).items():
            gamma = _gamma_from_log(_negate_log(self._log_gamma), img_batch.data)
            data = img_batch.data
            img_batch.data = data.sign() * data.abs().pow(gamma)
        return batch


def _negate_log(log_gamma: float | list[float]) -> float | list[float]:
    """Negate a scalar or per-element log-gamma value."""
    if isinstance(log_gamma, list):
        return [-value for value in log_gamma]
    return -log_gamma
