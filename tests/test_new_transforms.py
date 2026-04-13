"""Tests for Gamma, Blur, Clamp, Mask, and OneHot transforms."""

from __future__ import annotations

import numpy as np
import pytest
import torch

import torchio as tio


def _make_subject(with_label: bool = True) -> tio.Subject:
    data = torch.rand(1, 10, 10, 10) * 100
    kwargs: dict = {"t1": tio.ScalarImage(data)}
    if with_label:
        seg = torch.zeros(1, 10, 10, 10, dtype=torch.float32)
        seg[0, 2:5, 2:5, 2:5] = 1
        seg[0, 6:9, 6:9, 6:9] = 2
        kwargs["seg"] = tio.LabelMap(seg)
    return tio.Subject(**kwargs)


# ── Gamma ────────────────────────────────────────────────────────────


class TestGamma:
    def test_changes_data(self) -> None:
        subject = _make_subject(with_label=False)
        original = subject.t1.data.clone()
        result = tio.Gamma(log_gamma=0.3)(subject)
        assert not torch.allclose(result.t1.data, original)

    def test_zero_log_gamma_is_identity(self) -> None:
        subject = _make_subject(with_label=False)
        original = subject.t1.data.clone()
        result = tio.Gamma(log_gamma=0.0)(subject)
        torch.testing.assert_close(result.t1.data, original)

    def test_inverse(self) -> None:
        subject = _make_subject(with_label=False)
        original = subject.t1.data.clone()
        transformed = tio.Gamma(log_gamma=0.2)(subject)
        restored = transformed.apply_inverse_transform()
        np.testing.assert_allclose(
            restored.t1.data.numpy(),
            original.numpy(),
            atol=1e-4,
        )

    def test_leaves_labels_unchanged(self) -> None:
        subject = _make_subject()
        original_seg = subject.seg.data.clone()
        result = tio.Gamma()(subject)
        torch.testing.assert_close(result.seg.data, original_seg)


# ── Blur ─────────────────────────────────────────────────────────────


class TestBlur:
    def test_changes_data(self) -> None:
        subject = _make_subject(with_label=False)
        original = subject.t1.data.clone()
        result = tio.Blur(std=2.0)(subject)
        assert not torch.allclose(result.t1.data, original)

    def test_zero_std_is_identity(self) -> None:
        subject = _make_subject(with_label=False)
        original = subject.t1.data.clone()
        result = tio.Blur(std=0.0)(subject)
        torch.testing.assert_close(result.t1.data, original)

    def test_leaves_labels_unchanged(self) -> None:
        subject = _make_subject()
        original_seg = subject.seg.data.clone()
        result = tio.Blur(std=1.0)(subject)
        torch.testing.assert_close(result.seg.data, original_seg)


# ── Clamp ────────────────────────────────────────────────────────────


class TestClamp:
    def test_clamps_values(self) -> None:
        data = torch.tensor([-10.0, 0.0, 50.0, 200.0]).reshape(1, 1, 1, 4)
        subject = tio.Subject(t1=tio.ScalarImage(data))
        result = tio.Clamp(out_min=0, out_max=100)(subject)
        assert result.t1.data.min() >= 0
        assert result.t1.data.max() <= 100

    def test_clamp_min_only(self) -> None:
        data = torch.tensor([-5.0, 0.0, 5.0]).reshape(1, 1, 1, 3)
        subject = tio.Subject(t1=tio.ScalarImage(data))
        result = tio.Clamp(out_min=0)(subject)
        assert result.t1.data.min() >= 0

    def test_invalid_range_raises(self) -> None:
        with pytest.raises(ValueError, match="out_min"):
            tio.Clamp(out_min=100, out_max=0)


# ── Mask ─────────────────────────────────────────────────────────────


class TestMask:
    def test_mask_with_label_key(self) -> None:
        subject = _make_subject()
        result = tio.Mask(masking_method="seg")(subject)
        # Voxels outside the seg should be zero
        outside = subject.seg.data == 0
        assert (result.t1.data[outside.expand_as(result.t1.data)] == 0).all()

    def test_mask_with_callable(self) -> None:
        subject = _make_subject(with_label=False)
        result = tio.Mask(masking_method=lambda x: x > 50)(subject)
        assert result.t1.data.shape == subject.t1.data.shape

    def test_mask_with_labels(self) -> None:
        subject = _make_subject()
        result = tio.Mask(masking_method="seg", labels=[1])(subject)
        # Only label 1 region should be nonzero
        assert result.t1.data.shape == subject.t1.data.shape

    def test_mask_key_not_found(self) -> None:
        subject = _make_subject(with_label=False)
        with pytest.raises(KeyError, match="brain"):
            tio.Mask(masking_method="brain")(subject)

    def test_outside_value(self) -> None:
        subject = _make_subject()
        result = tio.Mask(masking_method="seg", outside_value=-1)(subject)
        outside = subject.seg.data == 0
        assert (result.t1.data[outside.expand_as(result.t1.data)] == -1).all()


# ── OneHot ───────────────────────────────────────────────────────────


class TestOneHot:
    def test_one_hot_encoding(self) -> None:
        subject = _make_subject()
        result = tio.OneHot()(subject)
        # seg had values 0, 1, 2 → 3 channels
        assert result.seg.data.shape[0] == 3  # batch dim stripped by unbatch
        # Each voxel should have exactly one 1 across channels
        assert (result.seg.data.sum(dim=0) == 1).all()

    def test_num_classes(self) -> None:
        subject = _make_subject()
        result = tio.OneHot(num_classes=5)(subject)
        assert result.seg.data.shape[0] == 5

    def test_inverse(self) -> None:
        subject = _make_subject()
        original = subject.seg.data.clone()
        transformed = tio.OneHot()(subject)
        restored = transformed.apply_inverse_transform()
        torch.testing.assert_close(restored.seg.data, original)

    def test_leaves_scalar_images_unchanged(self) -> None:
        subject = _make_subject()
        original = subject.t1.data.clone()
        result = tio.OneHot()(subject)
        torch.testing.assert_close(result.t1.data, original)


# ── Exports ──────────────────────────────────────────────────────────


class TestExports:
    def test_all_at_top_level(self) -> None:
        assert hasattr(tio, "Gamma")
        assert hasattr(tio, "Blur")
        assert hasattr(tio, "Clamp")
        assert hasattr(tio, "Mask")
        assert hasattr(tio, "OneHot")
