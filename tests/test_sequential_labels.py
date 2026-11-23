"""Tests for SequentialLabels transform."""

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


class TestSequentialLabels:
    def test_basic_sequential(self) -> None:
        seg = torch.zeros(1, 5, 5, 5, dtype=torch.float32)
        seg[0, 0:2, :, :] = 5
        seg[0, 3:5, :, :] = 10
        subject = tio.Subject(seg=tio.LabelMap(seg))
        result = tio.SequentialLabels()(subject)
        unique = sorted(result.seg.data.unique().tolist())
        assert unique == [0.0, 1.0, 2.0]

    def test_already_sequential(self) -> None:
        subject = _make_subject()
        result = tio.SequentialLabels()(subject)
        torch.testing.assert_close(result.seg.data, subject.seg.data)

    def test_inverse(self) -> None:
        seg = torch.zeros(1, 5, 5, 5, dtype=torch.float32)
        seg[0, 0:2, :, :] = 5
        seg[0, 3:5, :, :] = 10
        subject = tio.Subject(
            t1=tio.ScalarImage(torch.rand(1, 5, 5, 5)),
            seg=tio.LabelMap(seg),
        )
        original = subject.seg.data.clone()
        transformed = tio.SequentialLabels()(subject)
        restored = transformed.apply_inverse_transform()
        torch.testing.assert_close(restored.seg.data, original)

    def test_leaves_scalar_unchanged(self) -> None:
        subject = _make_subject()
        original = subject.t1.data.clone()
        result = tio.SequentialLabels()(subject)
        torch.testing.assert_close(result.t1.data, original)

    def test_batch_uses_each_samples_own_labels(self) -> None:
        """Each sample must be renumbered from its own labels, not sample 0's."""
        data0 = torch.zeros(1, 4, 4, 4, dtype=torch.float32)
        data0[0, 0, 0, 0] = 5
        data1 = torch.zeros(1, 4, 4, 4, dtype=torch.float32)
        data1[0, 0, 0, 0] = 7

        sub0 = tio.Subject(seg=tio.LabelMap(data0))
        sub1 = tio.Subject(seg=tio.LabelMap(data1))
        batch = tio.SubjectsBatch.from_subjects([sub0, sub1])

        out = tio.SequentialLabels()(batch)

        out_data = out.images["seg"].data
        assert sorted(out_data[0].unique().tolist()) == [0.0, 1.0]
        assert sorted(out_data[1].unique().tolist()) == [0.0, 1.0]
        # Sample 1's own label (7) must survive as non-background, not be
        # zeroed out because it was absent from sample 0's label set.
        assert out_data[1, 0, 0, 0, 0].item() != 0.0

    def test_batch_inverse_restores_each_sample(self) -> None:
        data0 = torch.zeros(1, 4, 4, 4, dtype=torch.float32)
        data0[0, 0, 0, 0] = 5
        data1 = torch.zeros(1, 4, 4, 4, dtype=torch.float32)
        data1[0, 0, 0, 0] = 7

        sub0 = tio.Subject(seg=tio.LabelMap(data0))
        sub1 = tio.Subject(seg=tio.LabelMap(data1))
        batch = tio.SubjectsBatch.from_subjects([sub0, sub1])
        original = batch.images["seg"].data.clone()

        transformed = tio.SequentialLabels()(batch)
        restored = transformed.apply_inverse_transform()

        torch.testing.assert_close(restored.images["seg"].data, original)
