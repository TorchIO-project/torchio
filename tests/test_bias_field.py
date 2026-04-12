"""Tests for BiasField transform."""

from __future__ import annotations

import numpy as np
import pytest
import torch

import torchio as tio


def _make_subject() -> tio.Subject:
    data = torch.rand(1, 16, 16, 16) + 1.0  # positive intensities
    return tio.Subject(
        t1=tio.ScalarImage(data),
        seg=tio.LabelMap(torch.zeros(1, 16, 16, 16)),
    )


class TestBasic:
    def test_changes_data(self) -> None:
        subject = _make_subject()
        original = subject.t1.data.clone()
        result = tio.BiasField()(subject)
        assert not torch.allclose(result.t1.data, original)

    def test_field_is_multiplicative(self) -> None:
        subject = _make_subject()
        result = tio.BiasField(std=0.3)(subject)
        # All values should remain positive (exp is always > 0)
        assert result.t1.data.min() > 0

    def test_zero_std_is_identity(self) -> None:
        subject = _make_subject()
        original = subject.t1.data.clone()
        result = tio.BiasField(std=0.0)(subject)
        torch.testing.assert_close(result.t1.data, original)

    def test_leaves_label_maps_unchanged(self) -> None:
        subject = _make_subject()
        original_seg = subject.seg.data.clone()
        result = tio.BiasField()(subject)
        torch.testing.assert_close(result.seg.data, original_seg)

    def test_random_std(self) -> None:
        subject = _make_subject()
        transform = tio.BiasField(std=(0.0, 1.0))
        results = [transform(subject).t1.data.mean().item() for _ in range(5)]
        assert len({f"{v:.4f}" for v in results}) > 1

    def test_custom_scale(self) -> None:
        subject = _make_subject()
        result = tio.BiasField(std=0.5, scale=0.1)(subject)
        assert result.t1.data.shape == subject.t1.data.shape


class TestValidation:
    def test_negative_scale_raises(self) -> None:
        with pytest.raises(ValueError, match="scale"):
            tio.BiasField(scale=-0.1)

    def test_scale_above_one_raises(self) -> None:
        with pytest.raises(ValueError, match="scale"):
            tio.BiasField(scale=1.5)


class TestInverse:
    def test_inverse_restores_values(self) -> None:
        subject = _make_subject()
        original = subject.t1.data.clone()
        transformed = tio.BiasField(std=0.3)(subject)
        restored = transformed.apply_inverse_transform()
        np.testing.assert_allclose(
            restored.t1.data.numpy(),
            original.numpy(),
            atol=1e-5,
        )


class TestExports:
    def test_available_at_top_level(self) -> None:
        assert hasattr(tio, "BiasField")
