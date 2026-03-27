"""ParameterRange: scalar-or-range pattern for transform parameters.

Allows transform parameters to be either deterministic (fixed scalar)
or random (sampled from a uniform distribution each call).

Examples:
    >>> ParameterRange(0.5)                    # deterministic: always (0.5, 0.5, 0.5)
    >>> ParameterRange(0.2, around=1.0)        # U(0.8, 1.2) per axis
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
              If *around* is set, treated as half-width of a uniform
              range centered on *around*.
            - ``(lo, hi)``: uniform range ``U(lo, hi)`` for all axes.
            - ``(a, b, c)``: deterministic per-axis values.
            - ``(lo0, hi0, lo1, hi1, lo2, hi2)``: per-axis ranges.

        around: Center value when *value* is a scalar half-width.
            For example, ``ParameterRange(0.2, around=1.0)`` means
            ``U(0.8, 1.2)`` per axis.
    """

    def __init__(
        self,
        value: float | tuple[float, ...],
        *,
        around: float = 0.0,
    ) -> None:
        self._original = value
        self._around = around
        if isinstance(value, (int, float)):
            value = float(value)
            if value == 0.0:
                # Zero half-width → deterministic at `around`
                self._ranges = (
                    (around, around),
                    (around, around),
                    (around, around),
                )
            elif around != 0.0:
                # Scalar half-width around a center
                self._ranges = (
                    (around - value, around + value),
                    (around - value, around + value),
                    (around - value, around + value),
                )
            else:
                # Plain scalar → deterministic
                self._ranges = (
                    (value, value),
                    (value, value),
                    (value, value),
                )
        elif isinstance(value, tuple):
            self._ranges = _parse_tuple(value, around)
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
    around: float,
) -> tuple[tuple[float, float], tuple[float, float], tuple[float, float]]:
    n = len(value)
    if n == 1:
        # Single-element tuple → treat as scalar
        v = float(value[0])
        return (
            (around + v if around else v, around + v if around else v),
            (around + v if around else v, around + v if around else v),
            (around + v if around else v, around + v if around else v),
        )
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


# ── Converters for attrs fields ──────────────────────────────────────


def to_range(value: float | tuple[float, ...] | ParameterRange) -> ParameterRange:
    """Convert a scalar or tuple to a ParameterRange.

    Use as an ``attrs`` converter::

        field: ParameterRange = attrs.field(converter=to_range)
    """
    return value if isinstance(value, ParameterRange) else ParameterRange(value)


def to_nonneg_range(
    value: float | tuple[float, ...] | ParameterRange,
) -> ParameterRange:
    """Like ``to_range``, but rejects negative values.

    Use as an ``attrs`` converter for parameters like standard
    deviation that must be non-negative.
    """
    pr = to_range(value)
    for lo, hi in pr._ranges:
        if lo < 0 or hi < 0:
            msg = f"Value must be non-negative, got {value}"
            raise ValueError(msg)
    return pr
