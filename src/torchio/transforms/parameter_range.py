"""_ParameterRange: scalar, range, choice, or distribution for transform parameters.

Each axis can be specified independently when a 3-element tuple is
used, where each element can be a float, `(lo, hi)` range,
`Choice`, or `Distribution`.

Examples:
    >>> _ParameterRange(0.5)                              # deterministic
    >>> _ParameterRange((5.0, 15.0))                      # U(5, 15) all axes
    >>> _ParameterRange((1.0, 2.0, 3.0))                  # per-axis fixed
    >>> _ParameterRange(Choice([-10, 0, 10]))             # discrete choice
    >>> _ParameterRange((0, 0, Choice([-90, 0, 90])))     # per-axis mix
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import cast
from typing import overload

import torch
from torch.distributions import Distribution

# ── Choice ───────────────────────────────────────────────────────────


class Choice:
    """A discrete set of values to sample from.

    Args:
        values: Sequence of numeric values to choose from.
        probabilities: Optional per-value probabilities.
            If `None`, all values are equally likely.

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

    def sample_batched(self, n: int) -> torch.Tensor:
        """Pick ``n`` values at random, with replacement.

        Args:
            n: Number of independent draws.

        Returns:
            A 1D tensor of shape ``(n,)``.
        """
        idx = torch.multinomial(self._probs, n, replacement=True)
        return self._values[idx]

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


def _sample_axis_batched(
    spec: AxisSpec,
    n: int,
    *,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Sample ``n`` independent floats from an axis specification.

    Args:
        spec: The axis specification (float, ``(lo, hi)`` range,
            `Choice`, or `Distribution`).
        n: Number of independent samples to draw.
        generator: Optional generator for the uniform-range path.
            Note that `Choice` and `Distribution` ignore it (see
            [`_ParameterRange.sample`][]).

    Returns:
        A 1D tensor of shape ``(n,)``.
    """
    if isinstance(spec, (int, float)):
        return torch.full((n,), float(spec))
    if isinstance(spec, Choice):
        return spec.sample_batched(n)
    if isinstance(spec, Distribution):
        return spec.sample((n,)).reshape(n).to(torch.float32)
    lo, hi = spec
    if lo == hi:
        return torch.full((n,), float(lo))
    return torch.empty(n).uniform_(float(lo), float(hi), generator=generator)


# ── _ParameterRange ───────────────────────────────────────────────────


class _ParameterRange:
    """Encapsulates the range-or-scalar pattern for transform params.

    Args:
        value: Parameter specification. Accepted forms:

            - `float`: deterministic value broadcast to 3 axes.
            - `(lo, hi)`: uniform range for all axes.
            - `(a, b, c)`: deterministic per-axis (when all plain
              numbers).
            - `(lo0, hi0, lo1, hi1, lo2, hi2)`: per-axis ranges.
            - `Choice`: discrete random, same for all axes.
            - `Distribution`: sample from any distribution.
            - 3-tuple of mixed specs, e.g.,
              `(0, Choice([-90, 0, 90]), (-10, 10))`.
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

    def is_constant(self, value: float) -> bool:
        """Whether every axis deterministically equals `value`.

        Args:
            value: The value to compare against on every axis.

        Returns:
            `True` if each axis is a fixed number, or a degenerate
            `(v, v)` range, equal to `value`. `Choice` and
            `Distribution` axes are never constant.
        """
        for axis in self._axes:
            if isinstance(axis, (int, float)):
                if float(axis) != float(value):
                    return False
            elif isinstance(axis, tuple):
                low, high = axis
                if not (low == high == value):
                    return False
            else:  # Choice or Distribution
                return False
        return True

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

    @overload
    def sample(
        self,
        n: None = ...,
        *,
        generator: torch.Generator | None = ...,
    ) -> tuple[float, float, float]: ...
    @overload
    def sample(
        self,
        n: int,
        *,
        generator: torch.Generator | None = ...,
    ) -> torch.Tensor: ...
    def sample(
        self,
        n: int | None = None,
        *,
        generator: torch.Generator | None = None,
    ) -> tuple[float, float, float] | torch.Tensor:
        """Sample a 3-tuple of values, or a batch of them.

        Args:
            n: If `None` (default), draw a single 3-tuple (legacy
                behavior). If an integer, draw ``n`` independent
                3-tuples and return a tensor of shape ``(n, 3)``.
            generator: Optional generator for the uniform-range path.

        Returns:
            A tuple of three floats when ``n is None``, otherwise a
            tensor of shape ``(n, 3)``.
        """
        if n is None:
            return (
                _sample_axis(self._axes[0], generator=generator),
                _sample_axis(self._axes[1], generator=generator),
                _sample_axis(self._axes[2], generator=generator),
            )
        columns = [
            _sample_axis_batched(axis, n, generator=generator) for axis in self._axes
        ]
        return torch.stack(columns, dim=-1)

    @overload
    def sample_1d(
        self,
        n: None = ...,
        *,
        generator: torch.Generator | None = ...,
    ) -> float: ...
    @overload
    def sample_1d(
        self,
        n: int,
        *,
        generator: torch.Generator | None = ...,
    ) -> torch.Tensor: ...
    def sample_1d(
        self,
        n: int | None = None,
        *,
        generator: torch.Generator | None = None,
    ) -> float | torch.Tensor:
        """Sample a single float (from the first axis spec), or a batch.

        Args:
            n: If `None` (default), draw a single float (legacy
                behavior). If an integer, draw ``n`` independent values
                and return a tensor of shape ``(n,)``.
            generator: Optional generator for the uniform-range path.

        Returns:
            A single float when ``n is None``, otherwise a tensor of
            shape ``(n,)``.
        """
        if n is None:
            return _sample_axis(self._axes[0], generator=generator)
        return _sample_axis_batched(self._axes[0], n, generator=generator)

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
) -> _ParameterRange:
    """Convert to a _ParameterRange."""
    return _ParameterRange(value)


def to_nonneg_range(
    value: float | tuple | Distribution | Choice,
) -> _ParameterRange:
    """Like `to_range`, but rejects negative values in tuple ranges."""
    pr = _ParameterRange(value)
    if pr._distribution is None:
        for lo, hi in pr._ranges:
            if lo < 0 or hi < 0:
                msg = f"Value must be non-negative, got {value}"
                raise ValueError(msg)
    return pr
