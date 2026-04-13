"""Clamp: clip intensity values to a range."""

from __future__ import annotations

from typing import Any

from ..data.batch import SubjectsBatch
from .transform import IntensityTransform


class Clamp(IntensityTransform):
    r"""Clamp intensity values into the range $[a, b]$.

    Wraps :func:`torch.clamp`.

    Args:
        out_min: Minimum value $a$. ``None`` means no lower bound.
        out_max: Maximum value $b$. ``None`` means no upper bound.
        **kwargs: See [`Transform`][torchio.Transform].

    Examples:
        >>> import torchio as tio
        >>> # CT windowing: clip to [-1000, 1000] Hounsfield units
        >>> clamp = tio.Clamp(out_min=-1000, out_max=1000)
        >>> # Clip negative values only
        >>> clamp = tio.Clamp(out_min=0)
    """

    def __init__(
        self,
        *,
        out_min: float | None = None,
        out_max: float | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        if out_min is not None and out_max is not None and out_min > out_max:
            msg = f"out_min ({out_min}) must be <= out_max ({out_max})"
            raise ValueError(msg)
        self.out_min = out_min
        self.out_max = out_max

    def make_params(self, batch: SubjectsBatch) -> dict[str, Any]:
        """No random parameters."""
        return {"out_min": self.out_min, "out_max": self.out_max}

    def apply_transform(
        self,
        batch: SubjectsBatch,
        params: dict[str, Any],
    ) -> SubjectsBatch:
        """Clamp each selected image."""
        out_min = params["out_min"]
        out_max = params["out_max"]
        for _name, img_batch in self._get_images(batch).items():
            img_batch.data = img_batch.data.clamp(min=out_min, max=out_max)
        return batch
