"""Tests for the CropOrPad transform."""

from __future__ import annotations

import pytest
import torch

import torchio as tio
from torchio.data.affine import Affine
from torchio.data.batch import SubjectsBatch

# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------


def _make_subject(
    shape: tuple[int, int, int] = (20, 20, 20),
    spacing: tuple[float, float, float] = (1.0, 1.0, 1.0),
    *,
    with_label: bool = False,
) -> tio.Subject:
    affine = Affine.from_spacing(spacing)
    image = tio.ScalarImage(
        torch.rand(1, *shape),
        affine=affine,
    )
    kwargs: dict = {"t1": image}
    if with_label:
        kwargs["seg"] = tio.LabelMap(
            torch.randint(0, 3, (1, *shape)),
            affine=affine,
        )
    return tio.Subject(**kwargs)


# ---------------------------------------------------------------------------
# Basic crop / pad / no-op
# ---------------------------------------------------------------------------


class TestCropOrPadBasic:
    def test_no_op_when_already_target_shape(self) -> None:
        subject = _make_subject((10, 10, 10))
        result = tio.CropOrPad(target_shape=10)(subject)
        assert result.t1.shape == (1, 10, 10, 10)

    def test_pad_when_smaller(self) -> None:
        subject = _make_subject((8, 8, 8))
        result = tio.CropOrPad(target_shape=12)(subject)
        assert result.t1.shape == (1, 12, 12, 12)

    def test_crop_when_larger(self) -> None:
        subject = _make_subject((20, 20, 20))
        result = tio.CropOrPad(target_shape=10)(subject)
        assert result.t1.shape == (1, 10, 10, 10)

    def test_mixed_crop_and_pad(self) -> None:
        """Some axes need cropping, others padding."""
        subject = _make_subject((30, 10, 20))
        result = tio.CropOrPad(target_shape=(20, 20, 20))(subject)
        assert result.t1.shape == (1, 20, 20, 20)

    def test_odd_difference_centering(self) -> None:
        """When the difference is odd, ini gets ceil and fin gets floor."""
        subject = _make_subject((10, 10, 10))
        result = tio.CropOrPad(target_shape=13)(subject)
        assert result.t1.shape == (1, 13, 13, 13)

    def test_crop_odd_difference_centering(self) -> None:
        subject = _make_subject((13, 13, 13))
        result = tio.CropOrPad(target_shape=10)(subject)
        assert result.t1.shape == (1, 10, 10, 10)


# ---------------------------------------------------------------------------
# target_shape specifications
# ---------------------------------------------------------------------------


class TestTargetShapeParam:
    def test_single_int(self) -> None:
        subject = _make_subject((20, 20, 20))
        result = tio.CropOrPad(target_shape=10)(subject)
        assert result.t1.shape == (1, 10, 10, 10)

    def test_three_tuple(self) -> None:
        subject = _make_subject((20, 20, 20))
        result = tio.CropOrPad(target_shape=(10, 15, 20))(subject)
        assert result.t1.shape == (1, 10, 15, 20)

    def test_none_leaves_axis_unchanged(self) -> None:
        subject = _make_subject((30, 20, 10))
        result = tio.CropOrPad(target_shape=(10, None, 20))(subject)
        assert result.t1.shape == (1, 10, 20, 20)

    def test_all_none_is_no_op(self) -> None:
        subject = _make_subject((30, 20, 10))
        result = tio.CropOrPad(target_shape=(None, None, None))(subject)
        assert result.t1.shape == (1, 30, 20, 10)

    def test_none_with_units(self) -> None:
        # 20 voxels at 2 mm = 40 mm, target None → keep 20
        subject = _make_subject((20, 20, 20), spacing=(2.0, 2.0, 2.0))
        result = tio.CropOrPad(
            target_shape=(30.0, None, 30.0),
            units="mm",
        )(subject)
        assert result.t1.shape == (1, 15, 20, 15)

    def test_invalid_tuple_length(self) -> None:
        with pytest.raises(ValueError, match="1 or 3"):
            tio.CropOrPad(target_shape=(1, 2))  # type: ignore[arg-type]

    def test_invalid_tuple_length_four(self) -> None:
        with pytest.raises(ValueError, match="1 or 3"):
            tio.CropOrPad(target_shape=(1, 2, 3, 4))  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Units
# ---------------------------------------------------------------------------


class TestUnits:
    def test_voxels_default(self) -> None:
        subject = _make_subject((20, 20, 20), spacing=(2.0, 2.0, 2.0))
        result = tio.CropOrPad(target_shape=10)(subject)
        assert result.t1.shape == (1, 10, 10, 10)

    def test_mm(self) -> None:
        # 20 voxels at 2 mm spacing = 40 mm. Target 30 mm → 15 voxels.
        subject = _make_subject((20, 20, 20), spacing=(2.0, 2.0, 2.0))
        result = tio.CropOrPad(target_shape=30.0, units="mm")(subject)
        assert result.t1.shape == (1, 15, 15, 15)

    def test_cm(self) -> None:
        # 20 voxels at 2 mm spacing = 40 mm. Target 3 cm = 30 mm → 15 voxels.
        subject = _make_subject((20, 20, 20), spacing=(2.0, 2.0, 2.0))
        result = tio.CropOrPad(target_shape=3.0, units="cm")(subject)
        assert result.t1.shape == (1, 15, 15, 15)

    def test_mm_per_axis(self) -> None:
        # spacing (1, 2, 4) mm
        # target (10, 20, 40) mm → (10, 10, 10) voxels
        subject = _make_subject((20, 20, 20), spacing=(1.0, 2.0, 4.0))
        result = tio.CropOrPad(
            target_shape=(10.0, 20.0, 40.0),
            units="mm",
        )(subject)
        assert result.t1.shape == (1, 10, 10, 10)

    def test_mm_rounds_to_nearest(self) -> None:
        # spacing 3 mm, target 10 mm → 10/3 ≈ 3.33 → round → 3 voxels
        subject = _make_subject((20, 20, 20), spacing=(3.0, 3.0, 3.0))
        result = tio.CropOrPad(target_shape=10.0, units="mm")(subject)
        assert result.t1.shape == (1, 3, 3, 3)

    def test_mm_rounds_up_at_half(self) -> None:
        # spacing 2 mm, target 5 mm → 5/2 = 2.5 → round → 2 voxels
        # (Python round uses banker's rounding: 2.5 → 2)
        subject = _make_subject((20, 20, 20), spacing=(2.0, 2.0, 2.0))
        result = tio.CropOrPad(target_shape=5.0, units="mm")(subject)
        assert result.t1.shape == (1, 2, 2, 2)

    def test_invalid_units(self) -> None:
        with pytest.raises(ValueError, match="units"):
            tio.CropOrPad(target_shape=10, units="inches")  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# only_crop / only_pad
# ---------------------------------------------------------------------------


class TestOnlyCropOnlyPad:
    def test_only_crop_true_skips_padding(self) -> None:
        subject = _make_subject((20, 10, 20))
        result = tio.CropOrPad(
            target_shape=(15, 15, 15),
            only_crop=True,
        )(subject)
        # Axis 0: 20→15 (crop), axis 1: 10→15 (skip), axis 2: 20→15 (crop)
        assert result.t1.shape == (1, 15, 10, 15)

    def test_only_pad_true_skips_cropping(self) -> None:
        subject = _make_subject((20, 10, 20))
        result = tio.CropOrPad(
            target_shape=(15, 15, 15),
            only_pad=True,
        )(subject)
        # Axis 0: 20→15 (skip), axis 1: 10→15 (pad), axis 2: 20→15 (skip)
        assert result.t1.shape == (1, 20, 15, 20)

    def test_only_crop_no_op_when_all_smaller(self) -> None:
        subject = _make_subject((5, 5, 5))
        result = tio.CropOrPad(target_shape=10, only_crop=True)(subject)
        assert result.t1.shape == (1, 5, 5, 5)

    def test_only_pad_no_op_when_all_larger(self) -> None:
        subject = _make_subject((20, 20, 20))
        result = tio.CropOrPad(target_shape=10, only_pad=True)(subject)
        assert result.t1.shape == (1, 20, 20, 20)

    def test_both_raises(self) -> None:
        with pytest.raises(ValueError, match="cannot both be True"):
            tio.CropOrPad(target_shape=10, only_crop=True, only_pad=True)


# ---------------------------------------------------------------------------
# Padding mode / fill
# ---------------------------------------------------------------------------


class TestPaddingMode:
    def test_constant_fill(self) -> None:
        tensor = torch.ones(1, 4, 4, 4)
        image = tio.ScalarImage(tensor)
        subject = tio.Subject(t1=image)
        result = tio.CropOrPad(target_shape=8, fill=-1)(subject)
        # Padded corners should be -1
        assert result.t1.data[0, 0, 0, 0] == -1
        # Interior should be 1
        assert result.t1.data[0, 2, 2, 2] == 1

    def test_reflect_mode(self) -> None:
        subject = _make_subject((4, 4, 4))
        result = tio.CropOrPad(
            target_shape=8,
            padding_mode="reflect",
        )(subject)
        assert result.t1.shape == (1, 8, 8, 8)


# ---------------------------------------------------------------------------
# Affine correctness
# ---------------------------------------------------------------------------


class TestAffine:
    def test_crop_shifts_origin_forward(self) -> None:
        subject = _make_subject((20, 20, 20))
        orig = subject.t1.affine.origin
        result = tio.CropOrPad(target_shape=10)(subject)
        new = result.t1.affine.origin
        # With identity direction and 1mm spacing, cropping 5 from start
        # shifts origin by +5 on each axis
        for o, n in zip(orig, new, strict=True):
            assert n > o

    def test_pad_shifts_origin_backward(self) -> None:
        subject = _make_subject((10, 10, 10))
        orig = subject.t1.affine.origin
        result = tio.CropOrPad(target_shape=20)(subject)
        new = result.t1.affine.origin
        for o, n in zip(orig, new, strict=True):
            assert n < o

    def test_affine_with_anisotropic_spacing(self) -> None:
        spacing = (0.5, 1.0, 2.0)
        subject = _make_subject((20, 20, 20), spacing=spacing)
        result = tio.CropOrPad(target_shape=10)(subject)
        assert result.t1.affine.spacing == pytest.approx(spacing)


# ---------------------------------------------------------------------------
# All images
# ---------------------------------------------------------------------------


class TestAllImages:
    def test_crop_or_pad_all_images(self) -> None:
        subject = _make_subject((20, 20, 20), with_label=True)
        result = tio.CropOrPad(target_shape=10)(subject)
        assert result.t1.shape == (1, 10, 10, 10)
        assert result.seg.shape == (1, 10, 10, 10)


# ---------------------------------------------------------------------------
# Invertibility
# ---------------------------------------------------------------------------


class TestInvertibility:
    def test_crop_then_inverse(self) -> None:
        tensor = torch.rand(1, 20, 20, 20)
        subject = tio.Subject(t1=tio.ScalarImage(tensor.clone()))
        transformed = tio.CropOrPad(target_shape=10)(subject)
        assert transformed.t1.shape == (1, 10, 10, 10)
        restored = transformed.apply_inverse_transform()
        assert restored.t1.shape == (1, 20, 20, 20)

    def test_pad_then_inverse(self) -> None:
        tensor = torch.rand(1, 10, 10, 10)
        subject = tio.Subject(t1=tio.ScalarImage(tensor.clone()))
        transformed = tio.CropOrPad(target_shape=20)(subject)
        assert transformed.t1.shape == (1, 20, 20, 20)
        restored = transformed.apply_inverse_transform()
        assert restored.t1.shape == (1, 10, 10, 10)
        torch.testing.assert_close(restored.t1.data, tensor)

    def test_mixed_then_inverse(self) -> None:
        tensor = torch.rand(1, 30, 10, 20)
        subject = tio.Subject(t1=tio.ScalarImage(tensor.clone()))
        transformed = tio.CropOrPad(target_shape=20)(subject)
        assert transformed.t1.shape == (1, 20, 20, 20)
        restored = transformed.apply_inverse_transform()
        assert restored.t1.shape == (1, 30, 10, 20)


# ---------------------------------------------------------------------------
# Input types
# ---------------------------------------------------------------------------


class TestInputTypes:
    def test_accepts_image(self) -> None:
        image = tio.ScalarImage(torch.rand(1, 20, 20, 20))
        result = tio.CropOrPad(target_shape=10)(image)
        assert isinstance(result, tio.Image)
        assert result.shape == (1, 10, 10, 10)

    def test_accepts_tensor(self) -> None:
        tensor = torch.rand(1, 20, 20, 20)
        result = tio.CropOrPad(target_shape=10)(tensor)
        assert isinstance(result, torch.Tensor)
        assert result.shape == (1, 10, 10, 10)

    def test_accepts_subject(self) -> None:
        subject = _make_subject((20, 20, 20))
        result = tio.CropOrPad(target_shape=10)(subject)
        assert isinstance(result, tio.Subject)
        assert result.t1.shape == (1, 10, 10, 10)


# ---------------------------------------------------------------------------
# Batch mode
# ---------------------------------------------------------------------------


class TestBatch:
    def test_batch_crop(self) -> None:
        subjects = [
            tio.Subject(
                t1=tio.ScalarImage(torch.rand(1, 20, 20, 20)),
            )
            for _ in range(3)
        ]
        batch = SubjectsBatch.from_subjects(subjects)
        result = tio.CropOrPad(target_shape=10)(batch)
        assert result.t1.data.shape == (3, 1, 10, 10, 10)

    def test_batch_pad(self) -> None:
        subjects = [
            tio.Subject(
                t1=tio.ScalarImage(torch.rand(1, 10, 10, 10)),
            )
            for _ in range(3)
        ]
        batch = SubjectsBatch.from_subjects(subjects)
        result = tio.CropOrPad(target_shape=20)(batch)
        assert result.t1.data.shape == (3, 1, 20, 20, 20)


# ---------------------------------------------------------------------------
# Probability
# ---------------------------------------------------------------------------


class TestProbability:
    def test_p_zero_is_no_op(self) -> None:
        subject = _make_subject((20, 20, 20))
        result = tio.CropOrPad(target_shape=10, p=0)(subject)
        assert result.t1.shape == (1, 20, 20, 20)


# ---------------------------------------------------------------------------
# Random location
# ---------------------------------------------------------------------------


class TestRandomLocation:
    def test_random_crop_shape(self) -> None:
        subject = _make_subject((30, 30, 30))
        result = tio.CropOrPad(target_shape=10, location="random")(subject)
        assert result.t1.shape == (1, 10, 10, 10)

    def test_random_crop_varies(self) -> None:
        """Two random crops of the same subject should (usually) differ."""
        torch.manual_seed(0)
        data = torch.arange(20 * 20 * 20, dtype=torch.float32).reshape(1, 20, 20, 20)
        transform = tio.CropOrPad(target_shape=5, location="random")
        r1 = transform(tio.ScalarImage(data.clone()))
        r2 = transform(tio.ScalarImage(data.clone()))
        assert not torch.equal(r1.data, r2.data)

    def test_random_pad_is_still_centered(self) -> None:
        """Padding should be centered even with location='random'."""
        subject = _make_subject((10, 10, 10))
        result_center = tio.CropOrPad(target_shape=20, location="center")(subject)
        result_random = tio.CropOrPad(target_shape=20, location="random")(subject)
        # Pure padding — both should produce the same result
        torch.testing.assert_close(result_center.t1.data, result_random.t1.data)

    def test_random_mixed_crop_and_pad(self) -> None:
        subject = _make_subject((30, 5, 20))
        result = tio.CropOrPad(target_shape=10, location="random")(subject)
        assert result.t1.shape == (1, 10, 10, 10)

    def test_random_with_none_axis(self) -> None:
        subject = _make_subject((30, 20, 10))
        result = tio.CropOrPad(
            target_shape=(10, None, 10),
            location="random",
        )(subject)
        assert result.t1.shape == (1, 10, 20, 10)

    def test_random_batch(self) -> None:
        from torchio.data.batch import SubjectsBatch

        subjects = [
            tio.Subject(
                t1=tio.ScalarImage(torch.rand(1, 20, 20, 20)),
            )
            for _ in range(3)
        ]
        batch = SubjectsBatch.from_subjects(subjects)
        result = tio.CropOrPad(target_shape=10, location="random")(batch)
        assert result.t1.data.shape == (3, 1, 10, 10, 10)

    def test_invalid_location(self) -> None:
        with pytest.raises(ValueError, match="location"):
            tio.CropOrPad(target_shape=10, location="top-left")  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Laziness preservation
# ---------------------------------------------------------------------------


class TestLaziness:
    def test_preserves_laziness_of_original(self, tmp_path) -> None:
        """CropOrPad should not load the original image's data."""
        import nibabel as nib
        import numpy as np

        path = tmp_path / "test.nii.gz"
        nib.save(
            nib.Nifti1Image(np.zeros((20, 20, 20)), np.eye(4)),
            path,
        )
        image = tio.ScalarImage(path)
        subject = tio.Subject(t1=image)
        assert not subject.t1.is_loaded
        tio.CropOrPad(target_shape=10)(subject)
        assert not subject.t1.is_loaded
