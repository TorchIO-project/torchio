"""Tests for ParameterRange."""

from __future__ import annotations

import pytest
import torch

from torchio.transforms.parameter_range import ParameterRange


class TestParameterRangeParsing:
    def test_scalar_is_deterministic(self) -> None:
        pr = ParameterRange(0.5)
        assert pr.is_deterministic
        assert pr.sample() == (0.5, 0.5, 0.5)

    def test_scalar_with_around(self) -> None:
        """scalar=0.2 around=1.0 → sample from U(0.8, 1.2) per axis."""
        pr = ParameterRange(0.2, around=1.0)
        assert not pr.is_deterministic
        for _ in range(50):
            values = pr.sample()
            assert len(values) == 3
            for v in values:
                assert 0.8 <= v <= 1.2

    def test_two_tuple_is_range(self) -> None:
        """(lo, hi) → sample from U(lo, hi) per axis."""
        pr = ParameterRange((5.0, 15.0))
        assert not pr.is_deterministic
        for _ in range(50):
            values = pr.sample()
            for v in values:
                assert 5.0 <= v <= 15.0

    def test_three_tuple_is_fixed(self) -> None:
        """(a, b, c) → deterministic per-axis values."""
        pr = ParameterRange((1.0, 2.0, 3.0))
        assert pr.is_deterministic
        assert pr.sample() == (1.0, 2.0, 3.0)

    def test_six_tuple_is_per_axis_ranges(self) -> None:
        """(lo0, hi0, lo1, hi1, lo2, hi2) → per-axis ranges."""
        pr = ParameterRange((0.0, 1.0, 10.0, 20.0, 100.0, 200.0))
        assert not pr.is_deterministic
        for _ in range(50):
            v0, v1, v2 = pr.sample()
            assert 0.0 <= v0 <= 1.0
            assert 10.0 <= v1 <= 20.0
            assert 100.0 <= v2 <= 200.0

    def test_zero_scalar_is_deterministic(self) -> None:
        pr = ParameterRange(0.0, around=1.0)
        assert pr.is_deterministic
        assert pr.sample() == (1.0, 1.0, 1.0)

    def test_invalid_tuple_length(self) -> None:
        with pytest.raises(ValueError, match="1, 2, 3, or 6"):
            ParameterRange((1.0, 2.0, 3.0, 4.0))


class TestParameterRangeSampling:
    def test_reproducible_with_generator(self) -> None:
        pr = ParameterRange((0.0, 100.0))
        g1 = torch.Generator().manual_seed(42)
        g2 = torch.Generator().manual_seed(42)
        assert pr.sample(generator=g1) == pr.sample(generator=g2)

    def test_different_seeds_differ(self) -> None:
        pr = ParameterRange((0.0, 100.0))
        g1 = torch.Generator().manual_seed(1)
        g2 = torch.Generator().manual_seed(2)
        # Very unlikely to be equal
        assert pr.sample(generator=g1) != pr.sample(generator=g2)

    def test_scalar_sample_ignores_generator(self) -> None:
        pr = ParameterRange(5.0)
        g = torch.Generator().manual_seed(42)
        assert pr.sample(generator=g) == (5.0, 5.0, 5.0)

    def test_sample_1d(self) -> None:
        """sample_1d returns a single float."""
        pr = ParameterRange((0.0, 10.0))
        for _ in range(50):
            v = pr.sample_1d()
            assert isinstance(v, float)
            assert 0.0 <= v <= 10.0

    def test_sample_1d_deterministic(self) -> None:
        pr = ParameterRange(3.14)
        assert pr.sample_1d() == 3.14


class TestParameterRangeRepr:
    def test_scalar_repr(self) -> None:
        pr = ParameterRange(0.5)
        assert repr(pr) == "0.5"

    def test_range_repr(self) -> None:
        pr = ParameterRange((1.0, 2.0))
        assert repr(pr) == "(1.0, 2.0)"

    def test_three_tuple_repr(self) -> None:
        pr = ParameterRange((1.0, 2.0, 3.0))
        assert repr(pr) == "(1.0, 2.0, 3.0)"
