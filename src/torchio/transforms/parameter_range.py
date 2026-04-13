"""ParameterRange: scalar, range, choice, or distribution for transform parameters.

Each axis can be specified independently when a 3-element tuple is
used, where each element can be a float, ``(lo, hi)`` range,
``Choice``, or ``Distribution``.

Examples:
    >>> ParameterRange(0.5)                              # deterministic
    >>> ParameterRange((5.0, 15.0))                      # U(5, 15) all axes
    >>> ParameterRange((1.0, 2.0, 3.0))                  # per-axis fixed
    >>> ParameterRange(Choice([-10, 0, 10]))             # discrete choice
    >>> ParameterRange((0, 0, Choice([-90, 0, 90])))     # per-axis mix
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import cast

import torch
from torch.distributions import Distribution

# ── Choice ───────────────────────────────────────────────────────────


class Choice:
    """A discrete set of values to sample from.

    Args:
        values: Sequence of numeric values to choose from.
        probabilities: Optional per-value probabilities.
            If ``None``, all values are equally likely.

    Examples:
        >>> Choice([-10, 0, 10])
        Choice([-10.0, 0.0, 10.0])
        >>> Choice([0.5, 1.0, 2.0], probabilities=[0.2, 0.6, 0.2])
        Choice([0.5, 1.0, 2.0], p=[0.20, 0.60, 0.20])
    """

    def __init__(
        self,
        values: Sequence[float | int],
        probabilities: Sequence[float] | None = None,
    ) -> None:
        if len(values) < 1:
            msg = "Choice requires at least one value"
            raise ValueError(msg)
        self._values = torch.tensor([float(v) for v in values])
        if probabilities is not None:
            if len(probabilities) != len(values):
                msg = f"Expected {len(values)} probabilities, got {len(probabilities)}"
                raise ValueError(msg)
            self._probs = torch.tensor([float(p) for p in probabilities])
        else:
            self._probs = torch.ones(len(values)) / len(values)

    def sample(self) -> float:
        """Pick one value at random."""
        idx = torch.multinomial(self._probs, 1).item()
        return float(self._values[int(idx)])

    def __repr__(self) -> str:
        vals = [f"{v:.1f}" if v == int(v) else f"{v}" for v in self._values.tolist()]
        parts = f"[{', '.join(vals)}]"
        if torch.allclose(self._probs, self._probs[0].expand_as(self._probs)):
            return f"Choice({parts})"
        probs = ", ".join(f"{p:.2f}" for p in self._probs.tolist())
        return f"Choice({parts}, p=[{probs}])"


# ── Per-axis sampler ─────────────────────────────────────────────────

#: What each axis stores after parsing.
AxisSpec = float | tuple[float, float] | Choice | Distribution


def _sample_axis(
    spec: AxisSpec,
    *,
    generator: torch.Generator | None = None,
) -> float:
    """Sample a single float from an axis specification."""
    if isinstance(spec, (int, float)):
        return float(spec)
    if isinstance(spec, Choice):
        return spec.sample()
    if isinstance(spec, Distribution):
        return spec.sample().item()
    lo, hi = spec
    if lo == hi:
        return float(lo)
    return torch.empty(1).uniform_(float(lo), float(hi), generator=generator).item()


# ── ParameterRange ───────────────────────────────────────────────────


class ParameterRange:
    """Encapsulates the range-or-scalar pattern for transform params.

    Args:
        value: Parameter specification. Accepted forms:

            - ``float``: deterministic value broadcast to 3 axes.
            - ``(lo, hi)``: uniform range for all axes.
            - ``(a, b, c)``: deterministic per-axis (when all plain
              numbers).
            - ``(lo0, hi0, lo1, hi1, lo2, hi2)``: per-axis ranges.
            - ``Choice``: discrete random, same for all axes.
            - ``Distribution``: sample from any distribution.
            - 3-tuple of mixed specs, e.g.,
              ``(0, Choice([-90, 0, 90]), (-10, 10))``.
    """

    def __init__(
        self,
        value: float | tuple | Distribution | Choice,
    ) -> None:
        self._original = value
        self._axes: tuple[AxisSpec, AxisSpec, AxisSpec]

        if isinstance(value, (int, float)):
            v = float(value)
            self._axes = (v, v, v)
        elif isinstance(value, (Choice, Distribution)):
            self._axes = (value, value, value)
        elif isinstance(value, tuple):
            self._axes = _parse_tuple(value)
        else:
            msg = (
                "Expected float, tuple, Distribution, or Choice,"
                f" got {type(value).__name__}"
            )
            raise TypeError(msg)

    @property
    def is_deterministic(self) -> bool:
        """Whether this range always returns the same values."""
        return all(isinstance(a, (int, float)) for a in self._axes)

    @property
    def _ranges(
        self,
    ) -> tuple[tuple[float, float], tuple[float, float], tuple[float, float]]:
        """Legacy: per-axis (lo, hi) ranges for validation code."""
        result: list[tuple[float, float]] = []
        for a in self._axes:
            if isinstance(a, (int, float)):
                result.append((float(a), float(a)))
            elif isinstance(a, tuple):
                result.append(cast(tuple[float, float], a))
            else:
                result.append((0.0, 0.0))
        return (result[0], result[1], result[2])

    @property
    def _distribution(self) -> Distribution | None:
        """Legacy: shared Distribution if the first axis uses one."""
        if isinstance(self._axes[0], Distribution):
            return self._axes[0]
        return None

    def sample(
        self,
        *,
        generator: torch.Generator | None = None,
    ) -> tuple[float, float, float]:
        """Sample a 3-tuple of values.

        Returns:
            Tuple of three floats.
        """
        return (
            _sample_axis(self._axes[0], generator=generator),
            _sample_axis(self._axes[1], generator=generator),
            _sample_axis(self._axes[2], generator=generator),
        )

    def sample_1d(
        self,
        *,
        generator: torch.Generator | None = None,
    ) -> float:
        """Sample a single float (from the first axis spec).

        Returns:
            A single float.
        """
        return _sample_axis(self._axes[0], generator=generator)

    def __repr__(self) -> str:
        v = self._original
        if isinstance(v, (Distribution, Choice)):
            return repr(v)
        if isinstance(v, tuple):
            inner = ", ".join(repr(x) for x in v)
            return f"({inner})"
        return str(v)


# ── Tuple parsing ────────────────────────────────────────────────────


def _is_plain_number(x: object) -> bool:
    return isinstance(x, (int, float))


def _parse_tuple(
    value: tuple,
) -> tuple[AxisSpec, AxisSpec, AxisSpec]:
    """Parse a tuple into three per-axis specs."""
    n = len(value)

    # 3-element: per-axis fixed OR per-axis mixed
    if n == 3:
        if all(_is_plain_number(v) for v in value):
            return (float(value[0]), float(value[1]), float(value[2]))
        return (
            _parse_single(value[0]),
            _parse_single(value[1]),
            _parse_single(value[2]),
        )

    # Remaining forms require all plain numbers
    if not all(_is_plain_number(v) for v in value):
        msg = f"Mixed per-axis specs require exactly 3 elements, got {n}"
        raise ValueError(msg)

    if n == 1:
        v = float(value[0])
        return (v, v, v)
    if n == 2:
        r: tuple[float, float] = (float(value[0]), float(value[1]))
        return (r, r, r)
    if n == 6:
        return (
            (float(value[0]), float(value[1])),
            (float(value[2]), float(value[3])),
            (float(value[4]), float(value[5])),
        )
    msg = f"Tuple must have 1, 2, 3, or 6 elements, got {n}"
    raise ValueError(msg)


def _parse_single(spec: object) -> AxisSpec:
    """Parse a single axis specification."""
    if isinstance(spec, (int, float)):
        return float(spec)
    if isinstance(spec, (Choice, Distribution)):
        return spec
    if isinstance(spec, tuple) and len(spec) == 2:
        lo, hi = spec
        if isinstance(lo, (int, float)) and isinstance(hi, (int, float)):
            return (float(lo), float(hi))
    msg = (
        "Per-axis spec must be a float, (lo, hi) tuple, Choice,"
        f" or Distribution, got {type(spec).__name__}"
    )
    raise TypeError(msg)


# ── Convenience constructors ─────────────────────────────────────────


def to_range(
    value: float | tuple | Distribution | Choice,
) -> ParameterRange:
    """Convert to a ParameterRange."""
    return ParameterRange(value)


def to_nonneg_range(
    value: float | tuple | Distribution | Choice,
) -> ParameterRange:
    """Like ``to_range``, but rejects negative values in tuple ranges."""
    pr = ParameterRange(value)
    if pr._distribution is None:
        for lo, hi in pr._ranges:
            if lo < 0 or hi < 0:
                msg = f"Value must be non-negative, got {value}"
                raise ValueError(msg)
    return pr
