"""ParameterRange: scalar-or-range pattern for transform parameters.

Allows transform parameters to be either deterministic (fixed scalar)
or random (sampled from a uniform distribution each call).

Examples:
    >>> ParameterRange(0.5)                    # deterministic: always (0.5, 0.5, 0.5)
    >>> ParameterRange((5.0, 15.0))            # U(5, 15) per axis
    >>> ParameterRange((1.0, 2.0, 3.0))        # deterministic per-axis
    >>> ParameterRange((0, 1, 10, 20, 100, 200))  # per-axis ranges
"""

from __future__ import annotations

import torch


class ParameterRange:
    """Encapsulates the range-or-scalar pattern for transform params.

    Args:
        value: Parameter specification. Accepted forms:

            - ``float``: deterministic value broadcast to 3 axes.
            - ``(lo, hi)``: uniform range ``U(lo, hi)`` for all axes.
            - ``(a, b, c)``: deterministic per-axis values.
            - ``(lo0, hi0, lo1, hi1, lo2, hi2)``: per-axis ranges.
    """

    def __init__(
        self,
        value: float | tuple[float, ...],
    ) -> None:
        self._original = value
        if isinstance(value, (int, float)):
            v = float(value)
            self._ranges = ((v, v), (v, v), (v, v))
        elif isinstance(value, tuple):
            self._ranges = _parse_tuple(value)
        else:
            msg = f"Expected float or tuple, got {type(value).__name__}"
            raise TypeError(msg)

    @property
    def is_deterministic(self) -> bool:
        """Whether this range always returns the same values."""
        return all(lo == hi for lo, hi in self._ranges)

    def sample(
        self,
        *,
        generator: torch.Generator | None = None,
    ) -> tuple[float, float, float]:
        """Sample a 3-tuple of values.

        Deterministic ranges return the fixed values. Random ranges
        sample from ``U(lo, hi)`` independently per axis.

        Args:
            generator: Optional torch Generator for reproducibility.

        Returns:
            Tuple of three floats.
        """
        if self.is_deterministic:
            return (self._ranges[0][0], self._ranges[1][0], self._ranges[2][0])
        values: list[float] = []
        for lo, hi in self._ranges:
            if lo == hi:
                values.append(lo)
            else:
                u = torch.empty(1).uniform_(lo, hi, generator=generator)
                values.append(u.item())
        return (values[0], values[1], values[2])

    def sample_1d(
        self,
        *,
        generator: torch.Generator | None = None,
    ) -> float:
        """Sample a single float from the first axis range.

        Useful for parameters that are not per-axis (e.g., noise std).

        Args:
            generator: Optional torch Generator for reproducibility.

        Returns:
            A single float.
        """
        lo, hi = self._ranges[0]
        if lo == hi:
            return lo
        return torch.empty(1).uniform_(lo, hi, generator=generator).item()

    def __repr__(self) -> str:
        v = self._original
        if isinstance(v, tuple):
            inner = ", ".join(str(x) for x in v)
            return f"({inner})"
        return str(v)


def _parse_tuple(
    value: tuple[float, ...],
) -> tuple[tuple[float, float], tuple[float, float], tuple[float, float]]:
    n = len(value)
    if n == 1:
        v = float(value[0])
        return ((v, v), (v, v), (v, v))
    if n == 2:
        lo, hi = float(value[0]), float(value[1])
        return ((lo, hi), (lo, hi), (lo, hi))
    if n == 3:
        a, b, c = float(value[0]), float(value[1]), float(value[2])
        return ((a, a), (b, b), (c, c))
    if n == 6:
        return (
            (float(value[0]), float(value[1])),
            (float(value[2]), float(value[3])),
            (float(value[4]), float(value[5])),
        )
    msg = f"Tuple must have 1, 2, 3, or 6 elements, got {n}"
    raise ValueError(msg)


def to_range(value: float | tuple[float, float]) -> ParameterRange:
    """Convert a scalar or tuple to a ParameterRange."""
    return ParameterRange(value)


def to_nonneg_range(value: float | tuple[float, float]) -> ParameterRange:
    """Like ``to_range``, but rejects negative values."""
    pr = ParameterRange(value)
    for lo, hi in pr._ranges:
        if lo < 0 or hi < 0:
            msg = f"Value must be non-negative, got {value}"
            raise ValueError(msg)
    return pr
