"""Tests for the transform base classes and composition."""

from __future__ import annotations

from dataclasses import asdict
from typing import Any

import nibabel as nib
import numpy as np
import pytest
import SimpleITK as sitk
import torch

import torchio as tio
from torchio.data.bboxes import BoundingBoxes
from torchio.data.bboxes import BoundingBoxFormat
from torchio.data.points import Points

# ── Helpers ──────────────────────────────────────────────────────────


def _make_subject() -> tio.Subject:
    return tio.Subject(
        t1=tio.ScalarImage(torch.rand(1, 8, 8, 8)),
        seg=tio.LabelMap(torch.randint(0, 3, (1, 8, 8, 8))),
        landmarks=Points(torch.rand(3, 3)),
        tumors=BoundingBoxes(
            torch.rand(2, 6),
            format=BoundingBoxFormat.IJKIJK,
        ),
        age=42,
    )


class _IdentityTransform(tio.Transform):
    """Transform that does nothing — for testing the base flow."""

    def apply_transform(self, batch: Any, params: dict) -> Any:
        return batch


class _DoubleIntensity(tio.IntensityTransform):
    """Doubles intensity of ScalarImages — for testing IntensityTransform."""

    def apply_transform(self, batch: Any, params: dict) -> Any:
        for _name, img_batch in self._get_images(batch).items():
            img_batch.data = img_batch.data * 2
        return batch


class _FlipSpatial(tio.SpatialTransform):
    """Flips along axis 0 — for testing SpatialTransform."""

    def apply_transform(self, batch: Any, params: dict) -> Any:
        for _name, img_batch in self._get_images(batch).items():
            img_batch.data = torch.flip(img_batch.data, [-3])
        return batch


# ── Transform base ───────────────────────────────────────────────────


class TestTransformBase:
    def test_forward_returns_subject(self) -> None:
        subject = _make_subject()
        result = _IdentityTransform()(subject)
        assert isinstance(result, tio.Subject)

    def test_forward_accepts_image(self) -> None:
        image = tio.ScalarImage(torch.rand(1, 8, 8, 8))
        result = _IdentityTransform()(image)
        assert isinstance(result, tio.Image)

    def test_forward_accepts_tensor(self) -> None:
        tensor = torch.rand(1, 8, 8, 8)
        result = _IdentityTransform()(tensor)
        assert isinstance(result, torch.Tensor)
        assert result.shape == (1, 8, 8, 8)

    def test_forward_accepts_ndarray(self) -> None:
        array = np.random.rand(1, 8, 8, 8).astype(np.float32)
        result = _IdentityTransform()(array)
        assert isinstance(result, np.ndarray)
        assert result.shape == (1, 8, 8, 8)

    def test_forward_accepts_ndarray_3d(self) -> None:
        array = np.random.rand(8, 8, 8).astype(np.float32)
        result = _IdentityTransform()(array)
        assert isinstance(result, np.ndarray)

    def test_forward_accepts_sitk(self) -> None:
        sitk_image = sitk.Image(8, 8, 8, sitk.sitkFloat32)
        result = _IdentityTransform()(sitk_image)
        assert isinstance(result, sitk.Image)

    def test_forward_accepts_nifti(self) -> None:
        nifti = nib.Nifti1Image(np.zeros((8, 8, 8)), np.eye(4))
        result = _IdentityTransform()(nifti)
        assert isinstance(result, nib.Nifti1Image)

    def test_sitk_preserves_spacing(self) -> None:
        sitk_image = sitk.Image(8, 8, 8, sitk.sitkFloat32)
        sitk_image.SetSpacing((2.0, 2.0, 2.0))
        result = _IdentityTransform()(sitk_image)
        assert result.GetSpacing() == pytest.approx((2.0, 2.0, 2.0))

    def test_nifti_preserves_affine(self) -> None:
        affine = np.diag([2.0, 2.0, 2.0, 1.0])
        nifti = nib.Nifti1Image(np.zeros((8, 8, 8)), affine)
        result = _IdentityTransform()(nifti)
        np.testing.assert_array_almost_equal(result.affine, affine)

    def test_forward_accepts_dict(self) -> None:
        data = {
            "t1": torch.rand(1, 8, 8, 8),
            "seg": torch.randint(0, 3, (1, 8, 8, 8)),
        }
        result = _IdentityTransform()(data)
        assert isinstance(result, dict)
        assert set(result.keys()) == {"t1", "seg"}
        assert isinstance(result["t1"], torch.Tensor)

    def test_dict_transform_modifies_data(self) -> None:
        data = {
            "t1": torch.rand(1, 8, 8, 8),
        }
        original = data["t1"].clone()
        result = _DoubleIntensity()(data)
        torch.testing.assert_close(result["t1"], original * 2)

    def test_dict_metadata_passthrough(self) -> None:
        data = {
            "t1": torch.rand(1, 8, 8, 8),
            "age": 42,
        }
        result = _IdentityTransform()(data)
        assert result["age"] == 42

    def test_probability_zero_skips(self) -> None:
        subject = _make_subject()
        original_data = subject.t1.data.clone()
        result = _DoubleIntensity(p=0.0)(subject)
        torch.testing.assert_close(result.t1.data, original_data)

    def test_probability_one_applies(self) -> None:
        subject = _make_subject()
        original_data = subject.t1.data.clone()
        result = _DoubleIntensity(p=1.0)(subject)
        torch.testing.assert_close(result.t1.data, original_data * 2)

    def test_history_recorded(self) -> None:
        subject = _make_subject()
        result = _IdentityTransform()(subject)
        assert len(result.applied_transforms) == 1
        assert result.applied_transforms[0].name == "_IdentityTransform"

    def test_history_has_params(self) -> None:
        subject = _make_subject()
        result = _IdentityTransform()(subject)
        trace = result.applied_transforms[0]
        assert isinstance(trace.params, dict)

    def test_history_serializable(self) -> None:
        subject = _make_subject()
        result = _IdentityTransform()(subject)
        d = asdict(result.applied_transforms[0])
        assert "name" in d
        assert "params" in d

    def test_is_nn_module(self) -> None:
        t = _IdentityTransform()
        assert isinstance(t, torch.nn.Module)

    def test_invalid_input_type(self) -> None:
        with pytest.raises(TypeError):
            _IdentityTransform()("not a valid input")


# ── include/exclude ──────────────────────────────────────────────────


class TestIncludeExclude:
    def test_include_filters(self) -> None:
        subject = _make_subject()
        t = _DoubleIntensity(include=["t1"])
        result = t(subject)
        # t1 should be doubled
        assert result.t1.data.mean() > 0
        # seg should be unchanged (it's a LabelMap so IntensityTransform
        # skips it anyway, but include also restricts)

    def test_exclude_filters(self) -> None:
        subject = _make_subject()
        original_t1 = subject.t1.data.clone()
        t = _DoubleIntensity(exclude=["t1"])
        result = t(subject)
        # t1 should be unchanged (excluded)
        torch.testing.assert_close(result.t1.data, original_t1)


# ── IntensityTransform ───────────────────────────────────────────────


class TestIntensityTransform:
    def test_only_scalar_images(self) -> None:
        subject = _make_subject()
        original_seg = subject.seg.data.clone()
        result = _DoubleIntensity()(subject)
        # seg (LabelMap) should not be modified
        torch.testing.assert_close(result.seg.data, original_seg)

    def test_scalar_image_modified(self) -> None:
        subject = _make_subject()
        original_t1 = subject.t1.data.clone()
        result = _DoubleIntensity()(subject)
        torch.testing.assert_close(result.t1.data, original_t1 * 2)


# ── SpatialTransform ─────────────────────────────────────────────────


class TestSpatialTransform:
    def test_all_images_modified(self) -> None:
        subject = _make_subject()
        result = _FlipSpatial()(subject)
        # Both t1 and seg should be flipped
        assert result.t1.data.shape == (1, 8, 8, 8)
        assert result.seg.data.shape == (1, 8, 8, 8)


# ── Compose ──────────────────────────────────────────────────────────


class TestCompose:
    def test_sequential_application(self) -> None:
        subject = _make_subject()
        original = subject.t1.data.clone()
        composed = tio.Compose([_DoubleIntensity(), _DoubleIntensity()])
        result = composed(subject)
        torch.testing.assert_close(result.t1.data, original * 4)

    def test_copy_true_preserves_original(self) -> None:
        subject = _make_subject()
        original = subject.t1.data.clone()
        composed = tio.Compose([_DoubleIntensity()], copy=True)
        composed(subject)
        # Original should be unchanged
        torch.testing.assert_close(subject.t1.data, original)

    def test_copy_false_no_deepcopy(self) -> None:
        """copy=False skips deepcopy — used inside Compose."""
        subject = _make_subject()
        composed = tio.Compose([_DoubleIntensity()], copy=False)
        result = composed(subject)
        # Result should be transformed
        assert result.t1.data.mean() > 0

    def test_empty_compose(self) -> None:
        subject = _make_subject()
        composed = tio.Compose([])
        result = composed(subject)
        assert isinstance(result, tio.Subject)

    def test_history_from_children(self) -> None:
        subject = _make_subject()
        composed = tio.Compose([_IdentityTransform(), _DoubleIntensity()])
        result = composed(subject)
        assert len(result.applied_transforms) == 2

    def test_accepts_image(self) -> None:
        image = tio.ScalarImage(torch.rand(1, 8, 8, 8))
        composed = tio.Compose([_IdentityTransform()])
        result = composed(image)
        assert isinstance(result, tio.Image)

    def test_accepts_tensor(self) -> None:
        tensor = torch.rand(1, 8, 8, 8)
        composed = tio.Compose([_IdentityTransform()])
        result = composed(tensor)
        assert isinstance(result, torch.Tensor)


# ── OneOf ─────────────────────────────────────────────────────────────


class TestOneOf:
    def test_applies_exactly_one(self) -> None:
        subject = _make_subject()
        one_of = tio.OneOf([_DoubleIntensity(), _IdentityTransform()])
        result = one_of(subject)
        # Exactly one transform should be in history
        assert len(result.applied_transforms) == 1

    def test_with_weights(self) -> None:
        subject = _make_subject()
        # Weight 1.0 on identity, 0.0 on double — should always pick identity
        one_of = tio.OneOf(
            {_IdentityTransform(): 1.0, _DoubleIntensity(): 0.0},
        )
        result = one_of(subject)
        assert result.applied_transforms[0].name == "_IdentityTransform"


# ── SomeOf ────────────────────────────────────────────────────────────


class TestSomeOf:
    def test_applies_n_transforms(self) -> None:
        subject = _make_subject()
        some_of = tio.SomeOf(
            [_IdentityTransform(), _DoubleIntensity(), _IdentityTransform()],
            num_transforms=2,
        )
        result = some_of(subject)
        assert len(result.applied_transforms) == 2

    def test_num_transforms_range(self) -> None:
        subject = _make_subject()
        some_of = tio.SomeOf(
            [_IdentityTransform(), _DoubleIntensity(), _IdentityTransform()],
            num_transforms=(1, 3),
        )
        result = some_of(subject)
        assert 1 <= len(result.applied_transforms) <= 3


# ── Operator sugar ───────────────────────────────────────────────────


class TestAddOperator:
    def test_add_creates_compose(self) -> None:
        t1 = tio.Flip(axes=(0,))
        t2 = tio.Noise(std=0.1)
        result = t1 + t2
        assert isinstance(result, tio.Compose)
        assert len(result.transforms) == 2

    def test_add_flattens_compose(self) -> None:
        t1 = tio.Flip(axes=(0,))
        t2 = tio.Noise(std=0.1)
        t3 = tio.BiasField()
        result = t1 + t2 + t3
        assert isinstance(result, tio.Compose)
        assert len(result.transforms) == 3

    def test_add_compose_plus_transform(self) -> None:
        c = tio.Compose([tio.Flip(axes=(0,)), tio.Noise(std=0.1)])
        t = tio.BiasField()
        result = c + t
        assert isinstance(result, tio.Compose)
        assert len(result.transforms) == 3

    def test_add_not_implemented_for_non_transform(self) -> None:
        with pytest.raises(TypeError):
            tio.Flip(axes=(0,)) + 42  # type: ignore[operator]

    def test_add_produces_working_pipeline(self) -> None:
        subject = _make_subject()
        pipeline = tio.Flip(axes=(0,)) + tio.Noise(std=0.01)
        result = pipeline(subject)
        assert result.t1.shape == subject.t1.shape


class TestOrOperator:
    def test_or_creates_oneof(self) -> None:
        t1 = tio.Flip(axes=(0,))
        t2 = tio.Noise(std=0.1)
        result = t1 | t2
        assert isinstance(result, tio.OneOf)
        assert len(result.transforms) == 2

    def test_or_flattens_oneof(self) -> None:
        t1 = tio.Flip(axes=(0,))
        t2 = tio.Noise(std=0.1)
        t3 = tio.BiasField()
        result = t1 | t2 | t3
        assert isinstance(result, tio.OneOf)
        assert len(result.transforms) == 3

    def test_or_not_implemented_for_non_transform(self) -> None:
        with pytest.raises(TypeError):
            tio.Flip(axes=(0,)) | "bad"  # type: ignore[operator]

    def test_or_produces_working_pipeline(self) -> None:
        subject = _make_subject()
        pipeline = tio.Flip(axes=(0,)) | tio.Noise(std=0.01)
        result = pipeline(subject)
        assert result.t1.shape == subject.t1.shape
