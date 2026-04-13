"""Tests for Swap transform."""

from __future__ import annotations

import warnings

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


class TestSwap:
    def test_changes_data(self) -> None:
        subject = _make_subject(with_label=False)
        original = subject.t1.data.clone()
        result = tio.Swap(patch_size=3, num_iterations=10)(subject)
        assert not torch.allclose(result.t1.data, original)

    def test_preserves_shape(self) -> None:
        subject = _make_subject(with_label=False)
        result = tio.Swap(patch_size=3, num_iterations=5)(subject)
        assert result.t1.data.shape == subject.t1.data.shape

    def test_warns_with_labels(self) -> None:
        subject = _make_subject()
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            tio.Swap(patch_size=3, num_iterations=1)(subject)
            assert any("LabelMap" in str(warning.message) for warning in w)

    def test_patch_too_large_raises(self) -> None:
        subject = _make_subject(with_label=False)
        with pytest.raises(ValueError, match="cannot be larger"):
            tio.Swap(patch_size=100, num_iterations=1)(subject)

    def test_single_iteration(self) -> None:
        subject = _make_subject(with_label=False)
        result = tio.Swap(patch_size=3, num_iterations=1)(subject)
        assert result.t1.data.shape == subject.t1.data.shape
