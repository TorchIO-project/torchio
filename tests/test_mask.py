"""Tests for Mask transform."""

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


class TestMask:
    def test_mask_with_label_key(self) -> None:
        subject = _make_subject()
        result = tio.Mask(masking_method="seg")(subject)
        outside = subject.seg.data == 0
        assert (result.t1.data[outside.expand_as(result.t1.data)] == 0).all()

    def test_mask_with_callable(self) -> None:
        subject = _make_subject(with_label=False)
        result = tio.Mask(masking_method=lambda x: x > 50)(subject)
        assert result.t1.data.shape == subject.t1.data.shape

    def test_mask_with_labels(self) -> None:
        subject = _make_subject()
        result = tio.Mask(masking_method="seg", labels=[1])(subject)
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

    def test_batch_uses_each_element_label_map(self) -> None:
        first = _make_subject()
        second = _make_subject()
        second.seg.set_data(second.seg.data.flip(1))
        batch = tio.SubjectsBatch.from_subjects([first, second])
        result = tio.Mask(masking_method="seg")(batch)
        for i, subject in enumerate((first, second)):
            expected = tio.Mask(masking_method="seg")(subject)
            torch.testing.assert_close(result.images["t1"].data[i], expected.t1.data)

    def test_batch_uses_each_element_callable_mask(self) -> None:
        first = _make_subject(with_label=False)
        second = _make_subject(with_label=False)
        batch = tio.SubjectsBatch.from_subjects([first, second])
        transform = tio.Mask(masking_method=lambda x: x > 50, outside_value=-1)
        result = transform(batch)
        for i, subject in enumerate((first, second)):
            expected = transform(subject)
            torch.testing.assert_close(result.images["t1"].data[i], expected.t1.data)
