"""Tests for Spike transform."""

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


class TestSpike:
    def test_changes_data(self) -> None:
        subject = _make_subject(with_label=False)
        original = subject.t1.data.clone()
        result = tio.Spike(num_spikes=3, intensity=2.0)(subject)
        assert not torch.allclose(result.t1.data, original)

    def test_zero_intensity_is_identity(self) -> None:
        subject = _make_subject(with_label=False)
        original = subject.t1.data.clone()
        result = tio.Spike(intensity=0.0)(subject)
        torch.testing.assert_close(result.t1.data, original)

    def test_leaves_labels_unchanged(self) -> None:
        subject = _make_subject()
        original_seg = subject.seg.data.clone()
        result = tio.Spike(num_spikes=3, intensity=2.0)(subject)
        torch.testing.assert_close(result.seg.data, original_seg)

    def test_single_spike(self) -> None:
        subject = _make_subject(with_label=False)
        original = subject.t1.data.clone()
        result = tio.Spike(num_spikes=1, intensity=1.0)(subject)
        assert not torch.allclose(result.t1.data, original)


class TestSpikePerInstance:
    def _batch(self, batch_size: int = 6) -> tio.SubjectsBatch:
        data = torch.rand(1, 12, 12, 12)
        subjects = [
            tio.Subject(t1=tio.ScalarImage(data.clone())) for _ in range(batch_size)
        ]
        return tio.SubjectsBatch.from_subjects(subjects)

    def test_per_instance_differs_across_batch(self) -> None:
        torch.manual_seed(0)
        batch = self._batch()
        result = tio.Spike(intensity=(1.0, 3.0))(batch)
        params = [history[-1].params for history in result.histories]
        assert all("_batched_keys" not in item for item in params)
        assert len({item["intensity"] for item in params}) > 1
        assert not torch.allclose(result.t1.data[0], result.t1.data[1])

    def test_per_instance_false_is_shared(self) -> None:
        torch.manual_seed(0)
        batch = self._batch()
        result = tio.Spike(intensity=(1.0, 3.0), per_instance=False)(batch)
        torch.testing.assert_close(result.t1.data[0], result.t1.data[1])

    def test_single_subject_keeps_scalar_params(self) -> None:
        subject = tio.Subject(t1=tio.ScalarImage(torch.rand(1, 12, 12, 12)))
        result = tio.Spike(intensity=(1.0, 3.0))(subject)
        assert "_batched_keys" not in result.applied_transforms[-1].params
