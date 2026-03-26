"""Tests for axis validation and conversion utilities."""

from __future__ import annotations

import itertools

import pytest

from torchio.data.axes import ANATOMICAL_PAIRS
from torchio.data.axes import AxesType
from torchio.data.axes import axes_type
from torchio.data.axes import get_axis_mapping
from torchio.data.axes import validate_axes


class TestValidateAxes:
    """Test axis string validation."""

    # --- Valid voxel axes ---

    @pytest.mark.parametrize(
        "axes",
        ["".join(p) for p in itertools.permutations("IJK")],
    )
    def test_all_voxel_permutations_valid(self, axes: str):
        assert validate_axes(axes) == axes

    # --- Valid anatomical axes ---

    def test_ras_valid(self):
        assert validate_axes("RAS") == "RAS"

    def test_lpi_valid(self):
        assert validate_axes("LPI") == "LPI"

    def test_air_valid(self):
        """One from each pair, unusual order."""
        assert validate_axes("AIR") == "AIR"

    def test_all_anatomical_combinations_valid(self):
        """There are 48 valid anatomical axis strings (8 sign combos x 6 orders)."""
        count = 0
        for choices in itertools.product(*ANATOMICAL_PAIRS):
            for perm in itertools.permutations(choices):
                validate_axes("".join(perm))
                count += 1
        assert count == 48

    # --- Invalid axes ---

    def test_xyz_invalid(self):
        with pytest.raises(ValueError, match="Invalid"):
            validate_axes("XYZ")

    def test_too_short(self):
        with pytest.raises(ValueError, match="3 characters"):
            validate_axes("IJ")

    def test_too_long(self):
        with pytest.raises(ValueError, match="3 characters"):
            validate_axes("IJKL")

    def test_duplicate_voxel(self):
        with pytest.raises(ValueError, match="Invalid"):
            validate_axes("IIJ")

    def test_duplicate_anatomical(self):
        with pytest.raises(ValueError, match="Invalid"):
            validate_axes("RRS")

    def test_same_pair_twice(self):
        """R and L are from the same pair — invalid."""
        with pytest.raises(ValueError, match="Invalid"):
            validate_axes("RLS")

    def test_mixed_voxel_anatomical(self):
        """I and J are voxel, R is anatomical — neither system matches."""
        with pytest.raises(ValueError, match="Invalid"):
            validate_axes("IJR")

    def test_lowercase_invalid(self):
        with pytest.raises(ValueError, match=r"3 characters|Invalid"):
            validate_axes("ijk")

    def test_empty_string(self):
        with pytest.raises(ValueError, match="3 characters"):
            validate_axes("")


class TestAxesType:
    """Test axis type detection."""

    def test_ijk_is_voxel(self):
        assert axes_type("IJK") == AxesType.VOXEL

    def test_kji_is_voxel(self):
        assert axes_type("KJI") == AxesType.VOXEL

    def test_ras_is_anatomical(self):
        assert axes_type("RAS") == AxesType.ANATOMICAL

    def test_lpi_is_anatomical(self):
        assert axes_type("LPI") == AxesType.ANATOMICAL


class TestAxisMapping:
    """Test computing permutation and flip between axis systems."""

    def test_ijk_to_ijk_identity(self):
        perm, flips = get_axis_mapping("IJK", "IJK")
        assert perm == (0, 1, 2)
        assert flips == (False, False, False)

    def test_ijk_to_kji(self):
        perm, flips = get_axis_mapping("IJK", "KJI")
        assert perm == (2, 1, 0)
        assert flips == (False, False, False)

    def test_ijk_to_jki(self):
        perm, flips = get_axis_mapping("IJK", "JKI")
        assert perm == (1, 2, 0)
        assert flips == (False, False, False)

    def test_ras_to_ras_identity(self):
        perm, flips = get_axis_mapping("RAS", "RAS")
        assert perm == (0, 1, 2)
        assert flips == (False, False, False)

    def test_ras_to_lpi(self):
        """R→L (flip), A→P (flip), S→I (flip), same order."""
        perm, flips = get_axis_mapping("RAS", "LPI")
        assert perm == (0, 1, 2)
        assert flips == (True, True, True)

    def test_ras_to_asr(self):
        """R→R (col 0→2), A→A (col 1→0), S→S (col 2→1). No flips."""
        perm, flips = get_axis_mapping("RAS", "ASR")
        assert perm == (1, 2, 0)
        assert flips == (False, False, False)

    def test_ras_to_lai(self):
        """R→L (flip, col 0→0), A→A (col 1→1), S→I (flip, col 2→2)."""
        perm, flips = get_axis_mapping("RAS", "LAI")
        assert perm == (0, 1, 2)
        assert flips == (True, False, True)

    def test_ras_to_psl(self):
        """A→P (flip, col 1→0), S→S (no flip, col 2→1), R→L (flip, col 0→2)."""
        perm, flips = get_axis_mapping("RAS", "PSL")
        assert perm == (1, 2, 0)
        assert flips == (True, False, True)

    def test_cross_type_raises(self):
        """Cannot map between voxel and anatomical directly."""
        with pytest.raises(ValueError, match="same type"):
            get_axis_mapping("IJK", "RAS")
