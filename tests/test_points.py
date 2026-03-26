"""Tests for Points."""

from __future__ import annotations

import copy

import numpy as np
import pytest
import torch

from torchio.data.points import Points


class TestPointsCreation:
    def test_from_tensor(self):
        coords = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        pts = Points(coords)
        assert pts.data.shape == (2, 3)

    def test_from_numpy(self):
        coords = np.array([[1.0, 2.0, 3.0]])
        pts = Points(coords)
        assert isinstance(pts.data, torch.Tensor)
        assert pts.data.shape == (1, 3)

    def test_with_affine(self):
        coords = torch.tensor([[1.0, 2.0, 3.0]])
        affine = np.diag([2.0, 2.0, 2.0, 1.0])
        pts = Points(coords, affine=affine)
        np.testing.assert_array_equal(pts.affine.numpy(), affine)

    def test_default_affine_is_identity(self):
        pts = Points(torch.tensor([[0.0, 0.0, 0.0]]))
        np.testing.assert_array_equal(pts.affine.numpy(), np.eye(4))

    def test_with_metadata(self):
        pts = Points(
            torch.tensor([[1.0, 2.0, 3.0]]),
            metadata={"structure": "hippocampus"},
        )
        assert pts.metadata["structure"] == "hippocampus"

    def test_empty_points(self):
        pts = Points(torch.zeros(0, 3))
        assert len(pts) == 0
        assert pts.data.shape == (0, 3)

    def test_wrong_shape_raises(self):
        with pytest.raises(ValueError, match="N, 3"):
            Points(torch.tensor([1.0, 2.0, 3.0]))

    def test_wrong_columns_raises(self):
        with pytest.raises(ValueError, match="N, 3"):
            Points(torch.tensor([[1.0, 2.0]]))

    def test_default_axes_ijk(self):
        pts = Points(torch.tensor([[1.0, 2.0, 3.0]]))
        assert pts.axes == "IJK"

    def test_custom_axes(self):
        pts = Points(torch.tensor([[1.0, 2.0, 3.0]]), axes="RAS")
        assert pts.axes == "RAS"

    def test_invalid_axes_raises(self):
        with pytest.raises(ValueError, match="Invalid"):
            Points(torch.tensor([[1.0, 2.0, 3.0]]), axes="XYZ")


class TestPointsProperties:
    def test_len(self):
        pts = Points(torch.randn(5, 3))
        assert len(pts) == 5

    def test_num_points(self):
        pts = Points(torch.randn(7, 3))
        assert pts.num_points == 7


class TestPointsNewLike:
    def test_new_like_preserves_affine(self):
        affine = np.diag([2.0, 2.0, 2.0, 1.0])
        pts = Points(torch.randn(3, 3), affine=affine)
        new = pts.new_like(data=torch.randn(5, 3))
        np.testing.assert_array_equal(new.affine.numpy(), affine)

    def test_new_like_preserves_metadata(self):
        pts = Points(torch.randn(3, 3), metadata={"label": "tumor"})
        new = pts.new_like(data=torch.randn(2, 3))
        assert new.metadata["label"] == "tumor"

    def test_new_like_with_new_affine(self):
        pts = Points(torch.randn(3, 3))
        new_affine = np.diag([3.0, 3.0, 3.0, 1.0])
        new = pts.new_like(data=torch.randn(2, 3), affine=new_affine)
        np.testing.assert_array_equal(new.affine.numpy(), new_affine)

    def test_new_like_preserves_axes(self):
        pts = Points(torch.randn(3, 3), axes="RAS")
        new = pts.new_like(data=torch.randn(2, 3))
        assert new.axes == "RAS"


class TestPointsToAxes:
    """Test conversion between axis conventions."""

    def test_ijk_to_kji(self):
        pts = Points(torch.tensor([[1.0, 2.0, 3.0]]))
        converted = pts.to_axes("KJI")
        expected = torch.tensor([[3.0, 2.0, 1.0]])
        torch.testing.assert_close(converted.data, expected)
        assert converted.axes == "KJI"

    def test_ijk_to_jki(self):
        pts = Points(torch.tensor([[10.0, 20.0, 30.0]]))
        converted = pts.to_axes("JKI")
        expected = torch.tensor([[20.0, 30.0, 10.0]])
        torch.testing.assert_close(converted.data, expected)

    def test_ras_to_lpi(self):
        pts = Points(torch.tensor([[10.0, 20.0, 30.0]]), axes="RAS")
        converted = pts.to_axes("LPI")
        expected = torch.tensor([[-10.0, -20.0, -30.0]])
        torch.testing.assert_close(converted.data, expected)

    def test_ras_to_asr(self):
        pts = Points(torch.tensor([[10.0, 20.0, 30.0]]), axes="RAS")
        converted = pts.to_axes("ASR")
        expected = torch.tensor([[20.0, 30.0, 10.0]])
        torch.testing.assert_close(converted.data, expected)

    def test_roundtrip_ijk_kji(self):
        data = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        pts = Points(data)
        roundtrip = pts.to_axes("KJI").to_axes("IJK")
        torch.testing.assert_close(roundtrip.data, data)

    def test_same_axes_noop(self):
        data = torch.tensor([[1.0, 2.0, 3.0]])
        pts = Points(data)
        converted = pts.to_axes("IJK")
        torch.testing.assert_close(converted.data, data)

    def test_voxel_to_anatomical(self):
        """IJK → RAS uses the affine."""
        affine = np.diag([2.0, 3.0, 4.0, 1.0])
        pts = Points(torch.tensor([[10.0, 10.0, 10.0]]), affine=affine)
        converted = pts.to_axes("RAS")
        expected = torch.tensor([[20.0, 30.0, 40.0]])
        torch.testing.assert_close(converted.data, expected)

    def test_anatomical_to_voxel(self):
        """RAS → IJK uses inverse affine."""
        affine = np.diag([2.0, 3.0, 4.0, 1.0])
        pts = Points(
            torch.tensor([[20.0, 30.0, 40.0]]),
            axes="RAS",
            affine=affine,
        )
        converted = pts.to_axes("IJK")
        expected = torch.tensor([[10.0, 10.0, 10.0]])
        torch.testing.assert_close(converted.data, expected, atol=1e-5, rtol=1e-5)

    def test_voxel_to_anatomical_roundtrip(self):
        affine = np.diag([2.0, 3.0, 4.0, 1.0])
        affine[:3, 3] = [10, 20, 30]
        data = torch.tensor([[5.0, 10.0, 15.0]])
        pts = Points(data, affine=affine)
        roundtrip = pts.to_axes("RAS").to_axes("IJK")
        torch.testing.assert_close(roundtrip.data, data, atol=1e-5, rtol=1e-5)

    def test_cross_type_raises_without_matching(self):
        """IJK → RAS should work using the affine."""
        pts = Points(torch.tensor([[1.0, 2.0, 3.0]]))
        converted = pts.to_axes("RAS")
        assert converted.axes == "RAS"


class TestPointsTransform:
    def test_to_world(self):
        # Points at voxel (1, 0, 0), affine scales by 2mm
        affine = np.diag([2.0, 3.0, 4.0, 1.0])
        pts = Points(torch.tensor([[1.0, 0.0, 0.0]]), affine=affine)
        world = pts.to_world()
        expected = torch.tensor([[2.0, 0.0, 0.0]])
        torch.testing.assert_close(world, expected)

    def test_to_world_with_origin(self):
        affine = np.eye(4)
        affine[:3, 3] = [10.0, 20.0, 30.0]
        pts = Points(torch.tensor([[0.0, 0.0, 0.0]]), affine=affine)
        world = pts.to_world()
        expected = torch.tensor([[10.0, 20.0, 30.0]])
        torch.testing.assert_close(world, expected)


class TestPointsRepr:
    def test_repr(self):
        pts = Points(torch.randn(5, 3))
        r = repr(pts)
        assert "Points" in r
        assert "5" in r

    def test_repr_with_axes(self):
        pts = Points(torch.randn(5, 3), axes="RAS")
        r = repr(pts)
        assert "RAS" in r


class TestPointsCopy:
    def test_copy(self):
        pts = Points(torch.randn(3, 3), metadata={"a": 1})
        copied = copy.deepcopy(pts)
        assert torch.equal(copied.data, pts.data)
        copied.metadata["a"] = 2
        assert pts.metadata["a"] == 1  # original unchanged

    def test_copy_preserves_axes(self):
        pts = Points(torch.randn(3, 3), axes="RAS")
        copied = copy.deepcopy(pts)
        assert copied.axes == "RAS"
