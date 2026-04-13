"""Tests for Motion transform."""

from __future__ import annotations

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


class TestMotion:
    def test_changes_data(self) -> None:
        subject = _make_subject(with_label=False)
        original = subject.t1.data.clone()
        result = tio.Motion(degrees=15, translation=10)(subject)
        assert not torch.allclose(result.t1.data, original)

    def test_num_transforms_validation(self) -> None:
        with pytest.raises(ValueError, match="num_transforms"):
            tio.Motion(num_transforms=0)

    def test_leaves_labels_unchanged(self) -> None:
        subject = _make_subject()
        original_seg = subject.seg.data.clone()
        result = tio.Motion()(subject)
        torch.testing.assert_close(result.seg.data, original_seg)

    def test_preserves_shape(self) -> None:
        subject = _make_subject(with_label=False)
        result = tio.Motion()(subject)
        assert result.t1.data.shape == subject.t1.data.shape

    def test_single_transform(self) -> None:
        subject = _make_subject(with_label=False)
        result = tio.Motion(num_transforms=1)(subject)
        assert result.t1.data.shape == subject.t1.data.shape
