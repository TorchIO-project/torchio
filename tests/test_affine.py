"""Tests for AffineMatrix."""

from __future__ import annotations

import numpy as np
import pytest

from torchio import AffineMatrix


class TestAffineCreation:
    def test_identity(self):
        affine = AffineMatrix()
        np.testing.assert_array_equal(affine.numpy(), np.eye(4))

    def test_from_numpy(self):
        matrix = np.diag([2.0, 3.0, 4.0, 1.0])
        affine = AffineMatrix(matrix)
        np.testing.assert_array_equal(affine.numpy(), matrix)

    def test_from_list(self):
        matrix = np.eye(4).tolist()
        affine = AffineMatrix(matrix)
        np.testing.assert_array_equal(affine.numpy(), np.eye(4))

    def test_must_be_4x4(self):
        with pytest.raises(ValueError, match=r"4.*4"):
            AffineMatrix(np.eye(3))

    def test_must_be_2d(self):
        with pytest.raises(ValueError, match=r"4.*4"):
            AffineMatrix(np.ones((4, 4, 4)))

    def test_always_float64(self):
        matrix = np.eye(4, dtype=np.float32)
        affine = AffineMatrix(matrix)
        assert affine.numpy().dtype == np.float64

    def test_copies_input(self):
        matrix = np.eye(4)
        affine = AffineMatrix(matrix)
        matrix[0, 0] = 999
        assert affine.numpy()[0, 0] == 1.0


class TestAffineFromSpacingOrigin:
    def test_isotropic(self):
        affine = AffineMatrix.from_spacing(spacing=(2.0, 2.0, 2.0))
        np.testing.assert_allclose(affine.spacing, (2.0, 2.0, 2.0))
        np.testing.assert_allclose(affine.origin, (0.0, 0.0, 0.0))

    def test_anisotropic(self):
        affine = AffineMatrix.from_spacing(spacing=(0.5, 0.8, 1.2))
        np.testing.assert_allclose(affine.spacing, (0.5, 0.8, 1.2))

    def test_with_origin(self):
        affine = AffineMatrix.from_spacing(
            spacing=(1.0, 1.0, 1.0),
            origin=(100.0, 200.0, 300.0),
        )
        np.testing.assert_allclose(affine.origin, (100.0, 200.0, 300.0))

    def test_with_direction(self):
        # 90-degree rotation around z-axis
        direction = np.array(
            [
                [0, -1, 0],
                [1, 0, 0],
                [0, 0, 1],
            ],
            dtype=np.float64,
        )
        affine = AffineMatrix.from_spacing(
            spacing=(2.0, 2.0, 2.0),
            direction=direction,
        )
        np.testing.assert_allclose(affine.spacing, (2.0, 2.0, 2.0))
        np.testing.assert_allclose(affine.direction, direction, atol=1e-10)


class TestAffineProperties:
    def test_spacing_identity(self):
        affine = AffineMatrix()
        np.testing.assert_allclose(affine.spacing, (1.0, 1.0, 1.0))

    def test_spacing_scaled(self):
        affine = AffineMatrix(np.diag([0.5, 0.8, 1.2, 1.0]))
        np.testing.assert_allclose(affine.spacing, (0.5, 0.8, 1.2))

    def test_origin_identity(self):
        affine = AffineMatrix()
        np.testing.assert_allclose(affine.origin, (0.0, 0.0, 0.0))

    def test_origin_translated(self):
        matrix = np.eye(4)
        matrix[:3, 3] = [10, 20, 30]
        affine = AffineMatrix(matrix)
        np.testing.assert_allclose(affine.origin, (10.0, 20.0, 30.0))

    def test_direction_identity(self):
        affine = AffineMatrix()
        np.testing.assert_allclose(affine.direction, np.eye(3))

    def test_direction_with_rotation(self):
        # 90-degree rotation around z
        matrix = np.eye(4)
        matrix[:3, :3] = [[0, -1, 0], [1, 0, 0], [0, 0, 1]]
        affine = AffineMatrix(matrix)
        expected = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]], dtype=np.float64)
        np.testing.assert_allclose(affine.direction, expected, atol=1e-10)

    def test_orientation_ras(self):
        affine = AffineMatrix()
        assert affine.orientation == ("R", "A", "S")

    def test_orientation_las(self):
        matrix = np.diag([-1.0, 1.0, 1.0, 1.0])
        affine = AffineMatrix(matrix)
        assert affine.orientation == ("L", "A", "S")


class TestAffineInverse:
    def test_inverse_identity(self):
        affine = AffineMatrix()
        inv = affine.inverse()
        np.testing.assert_allclose(inv.numpy(), np.eye(4))

    def test_inverse_scaling(self):
        affine = AffineMatrix(np.diag([2.0, 3.0, 4.0, 1.0]))
        inv = affine.inverse()
        np.testing.assert_allclose(inv.spacing, (0.5, 1 / 3, 0.25))

    def test_inverse_roundtrip(self):
        matrix = np.eye(4)
        matrix[:3, :3] = [[0, -2, 0], [3, 0, 0], [0, 0, 4]]
        matrix[:3, 3] = [10, 20, 30]
        affine = AffineMatrix(matrix)
        roundtrip = affine.inverse().inverse()
        np.testing.assert_allclose(roundtrip.numpy(), affine.numpy(), atol=1e-10)


class TestAffineCompose:
    def test_compose_identity(self):
        affine = AffineMatrix(np.diag([2.0, 3.0, 4.0, 1.0]))
        result = affine.compose(AffineMatrix())
        np.testing.assert_allclose(result.numpy(), affine.numpy())

    def test_compose_translations(self):
        a = AffineMatrix.from_spacing(spacing=(1, 1, 1), origin=(10, 0, 0))
        b = AffineMatrix.from_spacing(spacing=(1, 1, 1), origin=(0, 20, 0))
        result = a.compose(b)
        np.testing.assert_allclose(result.origin, (10, 20, 0))

    def test_compose_matmul(self):
        m1 = np.eye(4)
        m1[:3, 3] = [1, 2, 3]
        m2 = np.diag([2.0, 2.0, 2.0, 1.0])
        result = AffineMatrix(m1).compose(AffineMatrix(m2))
        np.testing.assert_allclose(result.numpy(), m1 @ m2)


class TestAffineMatmul:
    def test_matmul_operator(self):
        m1 = np.eye(4)
        m1[:3, 3] = [1, 2, 3]
        m2 = np.diag([2.0, 2.0, 2.0, 1.0])
        result = AffineMatrix(m1) @ AffineMatrix(m2)
        np.testing.assert_allclose(result.numpy(), m1 @ m2)

    def test_matmul_identity(self):
        affine = AffineMatrix(np.diag([2.0, 3.0, 4.0, 1.0]))
        result = affine @ AffineMatrix()
        np.testing.assert_allclose(result.numpy(), affine.numpy())

    def test_matmul_inverse(self):
        affine = AffineMatrix(np.diag([2.0, 3.0, 4.0, 1.0]))
        result = affine @ affine.inverse()
        np.testing.assert_allclose(result.numpy(), np.eye(4), atol=1e-10)

    def test_matmul_returns_affine(self):
        result = AffineMatrix() @ AffineMatrix()
        assert isinstance(result, AffineMatrix)

    def test_matmul_not_implemented_for_other_types(self):
        with pytest.raises(TypeError):
            AffineMatrix() @ "not an affine"


class TestAffineApply:
    def test_apply_identity(self):
        affine = AffineMatrix()
        points = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        result = affine.apply(points)
        np.testing.assert_allclose(result, points)

    def test_apply_translation(self):
        matrix = np.eye(4)
        matrix[:3, 3] = [10, 20, 30]
        affine = AffineMatrix(matrix)
        points = np.array([[0.0, 0.0, 0.0]])
        result = affine.apply(points)
        np.testing.assert_allclose(result, [[10.0, 20.0, 30.0]])

    def test_apply_scaling(self):
        affine = AffineMatrix(np.diag([2.0, 3.0, 4.0, 1.0]))
        points = np.array([[1.0, 1.0, 1.0]])
        result = affine.apply(points)
        np.testing.assert_allclose(result, [[2.0, 3.0, 4.0]])

    def test_apply_single_point(self):
        affine = AffineMatrix(np.diag([2.0, 2.0, 2.0, 1.0]))
        point = np.array([[5.0, 5.0, 5.0]])
        result = affine.apply(point)
        np.testing.assert_allclose(result, [[10.0, 10.0, 10.0]])


class TestAffineNumpyInterop:
    def test_array_protocol(self):
        matrix = np.diag([2.0, 3.0, 4.0, 1.0])
        affine = AffineMatrix(matrix)
        result = np.asarray(affine)
        np.testing.assert_array_equal(result, matrix)

    def test_array_with_dtype(self):
        affine = AffineMatrix(np.diag([2.0, 3.0, 4.0, 1.0]))
        result = np.array(affine, dtype=np.float32)
        assert result.dtype == np.float32
        np.testing.assert_allclose(result, np.diag([2.0, 3.0, 4.0, 1.0]))

    def test_array_with_copy(self):
        affine = AffineMatrix()
        result = np.array(affine, copy=True)
        result[0, 0] = 999
        assert affine.numpy()[0, 0] == 1.0

    def test_matmul_with_numpy(self):
        affine = AffineMatrix(np.diag([2.0, 2.0, 2.0, 1.0]))
        vec = np.array([1.0, 1.0, 1.0, 1.0])
        result = np.asarray(affine) @ vec
        np.testing.assert_allclose(result, [2.0, 2.0, 2.0, 1.0])


class TestAffineRepr:
    def test_repr_identity(self):
        affine = AffineMatrix()
        r = repr(affine)
        assert "AffineMatrix" in r
        assert "1.00" in r

    def test_repr_scaled(self):
        affine = AffineMatrix(np.diag([0.5, 0.8, 1.2, 1.0]))
        r = repr(affine)
        assert "0.50" in r
        assert "0.80" in r
        assert "1.20" in r


class TestAffineCopy:
    def test_copy(self):
        import copy

        affine = AffineMatrix(np.diag([2.0, 3.0, 4.0, 1.0]))
        copied = copy.copy(affine)
        np.testing.assert_array_equal(copied.numpy(), affine.numpy())
        # Verify independence
        copied.numpy()[0, 0] = 999
        assert affine.numpy()[0, 0] == 2.0

    def test_deepcopy(self):
        import copy

        affine = AffineMatrix(np.diag([2.0, 3.0, 4.0, 1.0]))
        copied = copy.deepcopy(affine)
        np.testing.assert_array_equal(copied.numpy(), affine.numpy())


class TestAffineEquality:
    def test_equal(self):
        a = AffineMatrix(np.diag([2.0, 3.0, 4.0, 1.0]))
        b = AffineMatrix(np.diag([2.0, 3.0, 4.0, 1.0]))
        assert a == b

    def test_not_equal(self):
        a = AffineMatrix(np.diag([2.0, 3.0, 4.0, 1.0]))
        b = AffineMatrix()
        assert a != b

    def test_not_equal_to_other_type(self):
        a = AffineMatrix()
        assert a != "not an affine"
