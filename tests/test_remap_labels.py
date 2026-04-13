"""Tests for RemapLabels transform."""

from __future__ import annotations

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


class TestRemapLabels:
    def test_basic_remap(self) -> None:
        subject = _make_subject()
        result = tio.RemapLabels({1: 10, 2: 20})(subject)
        assert 10 in result.seg.data.unique().tolist()
        assert 20 in result.seg.data.unique().tolist()
        assert 1 not in result.seg.data.unique().tolist()
        assert 2 not in result.seg.data.unique().tolist()

    def test_merge_labels(self) -> None:
        subject = _make_subject()
        result = tio.RemapLabels({2: 1})(subject)
        assert 2 not in result.seg.data.unique().tolist()
        assert 1 in result.seg.data.unique().tolist()

    def test_swap_labels(self) -> None:
        subject = _make_subject()
        original_1_count = (subject.seg.data == 1).sum().item()
        original_2_count = (subject.seg.data == 2).sum().item()
        result = tio.RemapLabels({1: 2, 2: 1})(subject)
        assert (result.seg.data == 1).sum().item() == original_2_count
        assert (result.seg.data == 2).sum().item() == original_1_count

    def test_leaves_unlisted_labels(self) -> None:
        subject = _make_subject()
        original_0_count = (subject.seg.data == 0).sum().item()
        result = tio.RemapLabels({1: 10})(subject)
        assert (result.seg.data == 0).sum().item() == original_0_count

    def test_inverse(self) -> None:
        subject = _make_subject()
        original = subject.seg.data.clone()
        transformed = tio.RemapLabels({1: 10, 2: 20})(subject)
        restored = transformed.apply_inverse_transform()
        torch.testing.assert_close(restored.seg.data, original)

    def test_leaves_scalar_unchanged(self) -> None:
        subject = _make_subject()
        original = subject.t1.data.clone()
        result = tio.RemapLabels({1: 10})(subject)
        torch.testing.assert_close(result.t1.data, original)
