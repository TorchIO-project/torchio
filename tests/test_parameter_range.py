"""Tests for ParameterRange."""

from __future__ import annotations

import pytest
import torch

import torchio as tio
from torchio.transforms.parameter_range import ParameterRange


class TestParameterRangeParsing:
    def test_scalar_is_deterministic(self) -> None:
        pr = ParameterRange(0.5)
        assert pr.is_deterministic
        assert pr.sample() == (0.5, 0.5, 0.5)

    def test_two_tuple_is_range(self) -> None:
        """(lo, hi) → sample from U(lo, hi) per axis."""
        pr = ParameterRange((0.8, 1.2))
        assert not pr.is_deterministic
        for _ in range(50):
            values = pr.sample()
            assert len(values) == 3
            for v in values:
                assert 0.8 <= v <= 1.2

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
        pr = ParameterRange(0.0)
        assert pr.is_deterministic
        assert pr.sample() == (0.0, 0.0, 0.0)

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


class TestParameterRangeDistribution:
    def test_distribution_not_deterministic(self) -> None:
        from torch.distributions import Normal

        pr = ParameterRange(Normal(0.0, 1.0))
        assert not pr.is_deterministic

    def test_distribution_sample_1d(self) -> None:
        from torch.distributions import Uniform

        pr = ParameterRange(Uniform(5.0, 10.0))
        for _ in range(50):
            v = pr.sample_1d()
            assert 5.0 <= v <= 10.0

    def test_distribution_sample_3d(self) -> None:
        from torch.distributions import Normal

        pr = ParameterRange(Normal(0.0, 1.0))
        v0, v1, v2 = pr.sample()
        assert isinstance(v0, float)
        assert isinstance(v1, float)
        assert isinstance(v2, float)

    def test_distribution_repr(self) -> None:
        from torch.distributions import Normal

        pr = ParameterRange(Normal(0.0, 1.0))
        assert "Normal" in repr(pr)


# ── Coverage gap tests ───────────────────────────────────────────────


class TestChoiceEdgeCases:
    def test_empty_choice_raises(self) -> None:
        with pytest.raises(ValueError, match="at least one"):
            tio.Choice([])

    def test_mismatched_probabilities_raises(self) -> None:
        with pytest.raises(ValueError, match="probabilities"):
            tio.Choice([1, 2, 3], probabilities=[0.5, 0.5])

    def test_repr_uniform(self) -> None:
        c = tio.Choice([1.0, 2.0, 3.0])
        r = repr(c)
        assert "Choice(" in r
        assert "p=" not in r

    def test_repr_custom_probs(self) -> None:
        c = tio.Choice([1.0, 2.0], probabilities=[0.3, 0.7])
        r = repr(c)
        assert "p=" in r


class TestParameterRangeEdgeCases:
    def test_invalid_type_raises(self) -> None:
        with pytest.raises(TypeError, match="Expected float"):
            tio.ParameterRange("bad")  # type: ignore[arg-type]

    def test_ranges_for_choice_axis(self) -> None:
        pr = tio.ParameterRange(tio.Choice([1.0, 2.0]))
        lo, hi = pr._ranges[0]
        assert lo == 0.0
        assert hi == 0.0

    def test_mixed_specs_wrong_count_raises(self) -> None:
        with pytest.raises(ValueError, match="Mixed per-axis"):
            tio.ParameterRange((tio.Choice([1.0]), tio.Choice([2.0])))

    def test_single_element_tuple(self) -> None:
        pr = tio.ParameterRange((5.0,))
        assert pr._ranges == ((5.0, 5.0), (5.0, 5.0), (5.0, 5.0))

    def test_invalid_axis_spec_raises(self) -> None:
        with pytest.raises(TypeError, match="Per-axis spec"):
            tio.ParameterRange(("a", "b", "c"))  # type: ignore[arg-type]

    def test_invalid_tuple_length_raises(self) -> None:
        with pytest.raises(ValueError, match="1, 2, 3, or 6"):
            tio.ParameterRange((1.0, 2.0, 3.0, 4.0))
