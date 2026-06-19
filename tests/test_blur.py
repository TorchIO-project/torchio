"""Tests for Blur transform."""

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


class TestBlurPerInstance:
    def _batch(self, batch_size: int = 5) -> tio.SubjectsBatch:
        data = torch.rand(1, 10, 10, 10)
        subjects = [
            tio.Subject(t1=tio.ScalarImage(data.clone())) for _ in range(batch_size)
        ]
        return tio.SubjectsBatch.from_subjects(subjects)

    def test_per_instance_differs_across_batch(self) -> None:
        torch.manual_seed(0)
        batch = self._batch()
        result = tio.Blur(std=(1.0, 4.0))(batch)
        params = result.applied_transforms[-1].params
        assert "_batched_keys" in params
        assert len(params["std"]) == batch.batch_size
        data = result.t1.data
        assert not torch.allclose(data[0], data[1])

    def test_per_instance_false_is_shared(self) -> None:
        torch.manual_seed(0)
        batch = self._batch()
        result = tio.Blur(std=(1.0, 4.0), per_instance=False)(batch)
        data = result.t1.data
        torch.testing.assert_close(data[0], data[1])

    def test_single_subject_keeps_scalar_params(self) -> None:
        subject = tio.Subject(t1=tio.ScalarImage(torch.rand(1, 8, 8, 8)))
        result = tio.Blur(std=(1.0, 4.0))(subject)
        assert "_batched_keys" not in result.applied_transforms[-1].params

    def test_per_instance_p_gates_some_elements(self) -> None:
        torch.manual_seed(0)
        batch = self._batch(batch_size=64)
        original = batch.t1.data.clone()
        result = tio.Blur(std=(2.0, 4.0), p=0.5)(batch)
        changed = [
            not torch.allclose(result.t1.data[i], original[i])
            for i in range(batch.batch_size)
        ]
        assert any(changed)
        assert not all(changed)

    def test_per_instance_p_masked_float64_elements_unchanged(self) -> None:
        """Zero-sigma elements must be exact, even for float64 data."""
        torch.manual_seed(0)
        data = (torch.rand(1, 8, 8, 8) + 0.1).double()
        subjects = [tio.Subject(t1=tio.ScalarImage(data.clone())) for _ in range(32)]
        batch = tio.SubjectsBatch.from_subjects(subjects)
        original = batch.t1.data.clone()
        result = tio.Blur(std=(2.0, 4.0), p=0.5)(batch)
        unchanged = [
            torch.equal(result.t1.data[i], original[i]) for i in range(batch.batch_size)
        ]
        assert any(unchanged)
        assert not all(unchanged)
