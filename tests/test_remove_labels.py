"""Tests for RemoveLabels transform."""

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


class TestRemoveLabels:
    def test_removes_specified_labels(self) -> None:
        subject = _make_subject()
        result = tio.RemoveLabels([2])(subject)
        assert 2 not in result.seg.data.unique().tolist()
        assert 1 in result.seg.data.unique().tolist()

    def test_removes_multiple_labels(self) -> None:
        subject = _make_subject()
        result = tio.RemoveLabels([1, 2])(subject)
        unique = result.seg.data.unique().tolist()
        assert unique == [0.0]

    def test_custom_background(self) -> None:
        subject = _make_subject()
        result = tio.RemoveLabels([1], background_label=99)(subject)
        assert 1 not in result.seg.data.unique().tolist()
        assert 99 in result.seg.data.unique().tolist()

    def test_leaves_scalar_unchanged(self) -> None:
        subject = _make_subject()
        original = subject.t1.data.clone()
        result = tio.RemoveLabels([1])(subject)
        torch.testing.assert_close(result.t1.data, original)

    def test_noop_when_label_absent(self) -> None:
        subject = _make_subject()
        original = subject.seg.data.clone()
        result = tio.RemoveLabels([99])(subject)
        torch.testing.assert_close(result.seg.data, original)
