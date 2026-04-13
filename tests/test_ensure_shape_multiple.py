"""Tests for the EnsureShapeMultiple transform."""

from __future__ import annotations

import pytest
import torch

import torchio as tio
from torchio.data.affine import AffineMatrix
from torchio.data.batch import SubjectsBatch

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_subject(
    shape: tuple[int, int, int] = (10, 10, 10),
    spacing: tuple[float, float, float] = (1.0, 1.0, 1.0),
    *,
    with_label: bool = False,
) -> tio.Subject:
    affine = AffineMatrix.from_spacing(spacing)
    image = tio.ScalarImage(torch.rand(1, *shape), affine=affine)
    kwargs: dict = {"t1": image}
    if with_label:
        kwargs["seg"] = tio.LabelMap(
            torch.randint(0, 3, (1, *shape)),
            affine=affine,
        )
    return tio.Subject(**kwargs)


# ---------------------------------------------------------------------------
# Padding (default method)
# ---------------------------------------------------------------------------


class TestPad:
    def test_pad_to_next_multiple(self) -> None:
        subject = _make_subject((10, 10, 10))
        result = tio.EnsureShapeMultiple(8)(subject)
        assert result.t1.spatial_shape == (16, 16, 16)

    def test_pad_asymmetric_shape(self) -> None:
        subject = _make_subject((10, 17, 25))
        result = tio.EnsureShapeMultiple(8)(subject)
        # 10→16, 17→24, 25→32
        assert result.t1.spatial_shape == (16, 24, 32)

    def test_pad_no_op_when_already_multiple(self) -> None:
        subject = _make_subject((16, 24, 8))
        result = tio.EnsureShapeMultiple(8)(subject)
        assert result.t1.spatial_shape == (16, 24, 8)

    def test_pad_per_axis_tuple(self) -> None:
        subject = _make_subject((10, 10, 10))
        result = tio.EnsureShapeMultiple((4, 8, 16))(subject)
        # 10→12, 10→16, 10→16
        assert result.t1.spatial_shape == (12, 16, 16)


# ---------------------------------------------------------------------------
# Cropping
# ---------------------------------------------------------------------------


class TestCrop:
    def test_crop_to_previous_multiple(self) -> None:
        subject = _make_subject((10, 10, 10))
        result = tio.EnsureShapeMultiple(8, method="crop")(subject)
        assert result.t1.spatial_shape == (8, 8, 8)

    def test_crop_asymmetric_shape(self) -> None:
        subject = _make_subject((10, 17, 25))
        result = tio.EnsureShapeMultiple(8, method="crop")(subject)
        # 10→8, 17→16, 25→24
        assert result.t1.spatial_shape == (8, 16, 24)

    def test_crop_no_op_when_already_multiple(self) -> None:
        subject = _make_subject((16, 24, 8))
        result = tio.EnsureShapeMultiple(8, method="crop")(subject)
        assert result.t1.spatial_shape == (16, 24, 8)

    def test_crop_per_axis_tuple(self) -> None:
        subject = _make_subject((10, 10, 10))
        result = tio.EnsureShapeMultiple((4, 6, 8), method="crop")(subject)
        # 10→8, 10→6, 10→8
        assert result.t1.spatial_shape == (8, 6, 8)

    def test_crop_small_shape_clamps_to_one(self) -> None:
        """When cropping would result in 0, clamp to at least 1."""
        subject = _make_subject((3, 3, 3))
        result = tio.EnsureShapeMultiple(8, method="crop")(subject)
        # floor(3/8) = 0 → 0*8 = 0 → max(0, 1) = 1
        assert all(s >= 1 for s in result.t1.spatial_shape)


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


class TestValidation:
    def test_invalid_method_raises(self) -> None:
        with pytest.raises(ValueError, match="method"):
            tio.EnsureShapeMultiple(8, method="resize")

    def test_method_must_be_crop_or_pad(self) -> None:
        # Valid methods should not raise
        tio.EnsureShapeMultiple(8, method="crop")
        tio.EnsureShapeMultiple(8, method="pad")


# ---------------------------------------------------------------------------
# Input types
# ---------------------------------------------------------------------------


class TestInputTypes:
    def test_accepts_subject(self) -> None:
        subject = _make_subject((10, 10, 10))
        result = tio.EnsureShapeMultiple(8)(subject)
        assert isinstance(result, tio.Subject)
        assert result.t1.spatial_shape == (16, 16, 16)

    def test_accepts_image(self) -> None:
        image = tio.ScalarImage(torch.rand(1, 10, 10, 10))
        result = tio.EnsureShapeMultiple(8)(image)
        assert isinstance(result, tio.Image)
        assert result.spatial_shape == (16, 16, 16)

    def test_accepts_tensor(self) -> None:
        tensor = torch.rand(1, 10, 10, 10)
        result = tio.EnsureShapeMultiple(8)(tensor)
        assert isinstance(result, torch.Tensor)
        assert result.shape == (1, 16, 16, 16)


# ---------------------------------------------------------------------------
# Batch
# ---------------------------------------------------------------------------


class TestBatch:
    def test_batch_pad(self) -> None:
        subjects = [
            tio.Subject(t1=tio.ScalarImage(torch.rand(1, 10, 10, 10))) for _ in range(3)
        ]
        batch = SubjectsBatch.from_subjects(subjects)
        result = tio.EnsureShapeMultiple(8)(batch)
        assert result.t1.data.shape == (3, 1, 16, 16, 16)

    def test_batch_crop(self) -> None:
        subjects = [
            tio.Subject(t1=tio.ScalarImage(torch.rand(1, 10, 10, 10))) for _ in range(3)
        ]
        batch = SubjectsBatch.from_subjects(subjects)
        result = tio.EnsureShapeMultiple(8, method="crop")(batch)
        assert result.t1.data.shape == (3, 1, 8, 8, 8)


# ---------------------------------------------------------------------------
# Multiple images
# ---------------------------------------------------------------------------


class TestMultipleImages:
    def test_all_images_transformed(self) -> None:
        subject = _make_subject((10, 10, 10), with_label=True)
        result = tio.EnsureShapeMultiple(8)(subject)
        assert result.t1.spatial_shape == (16, 16, 16)
        assert result.seg.spatial_shape == (16, 16, 16)


# ---------------------------------------------------------------------------
# Probability
# ---------------------------------------------------------------------------


class TestProbability:
    def test_p_zero_is_no_op(self) -> None:
        subject = _make_subject((10, 10, 10))
        result = tio.EnsureShapeMultiple(8, p=0)(subject)
        assert result.t1.spatial_shape == (10, 10, 10)


# ---------------------------------------------------------------------------
# Power of 2 use case (common for U-Nets)
# ---------------------------------------------------------------------------


class TestPowerOfTwo:
    def test_three_pooling_layers(self) -> None:
        """Common U-Net use case: 3 pooling layers → multiple of 8."""
        subject = _make_subject((181, 217, 181))
        result = tio.EnsureShapeMultiple(2**3)(subject)
        for s in result.t1.spatial_shape:
            assert s % 8 == 0

    def test_four_pooling_layers(self) -> None:
        """4 pooling layers → multiple of 16."""
        subject = _make_subject((181, 217, 181))
        result = tio.EnsureShapeMultiple(2**4)(subject)
        for s in result.t1.spatial_shape:
            assert s % 16 == 0


# ── Coverage gap tests ───────────────────────────────────────────────


class TestEnsureShapeMultipleValidation:
    def test_zero_multiple_raises(self) -> None:
        with pytest.raises(ValueError, match=">= 1"):
            tio.EnsureShapeMultiple(target_multiple=0)

    def test_wrong_tuple_length_raises(self) -> None:
        with pytest.raises(ValueError, match="1 or 3"):
            tio.EnsureShapeMultiple(target_multiple=(2, 4))

    def test_negative_in_tuple_raises(self) -> None:
        with pytest.raises(ValueError, match=">= 1"):
            tio.EnsureShapeMultiple(target_multiple=(2, -1, 4))
