"""Tests for OneOf transform."""

from __future__ import annotations

import pytest
import torch

import torchio as tio


def _make_subject() -> tio.Subject:
    return tio.Subject(
        t1=tio.ScalarImage(torch.rand(1, 10, 10, 10)),
    )


class TestOneOf:
    def test_applies_one(self) -> None:
        subject = _make_subject()
        transform = tio.OneOf(
            [
                tio.Flip(axes=(0,)),
                tio.Gamma(log_gamma=0.3),
            ]
        )
        result = transform(subject)
        assert result.t1.data.shape == subject.t1.data.shape

    def test_single_transform(self) -> None:
        subject = _make_subject()
        result = tio.OneOf([tio.Flip(axes=(0,))])(subject)
        assert result.t1.data.shape == subject.t1.data.shape

    def test_with_weights(self) -> None:
        subject = _make_subject()
        transform = tio.OneOf(
            {
                tio.Flip(axes=(0,)): 1.0,
                tio.Gamma(log_gamma=0.0): 0.0,
            }
        )
        result = transform(subject)
        assert result.t1.data.shape == subject.t1.data.shape

    def test_history_recorded(self) -> None:
        subject = _make_subject()
        result = tio.OneOf([tio.Flip(axes=(0,))])(subject)
        assert len(result.applied_transforms) > 0


class TestOneOfPerInstance:
    def _batch(self, batch_size: int = 32) -> tio.SubjectsBatch:
        data = torch.rand(1, 8, 8, 8)
        subjects = [
            tio.Subject(t1=tio.ScalarImage(data.clone())) for _ in range(batch_size)
        ]
        return tio.SubjectsBatch.from_subjects(subjects)

    def test_per_element_chooses_different_transforms(self) -> None:
        torch.manual_seed(0)
        batch = self._batch()
        transform = tio.OneOf([tio.Gamma(log_gamma=0.5), tio.Flip(axes=(0,))])
        result = transform(batch)
        names = set()
        for subject in result.unbatch():
            assert len(subject.applied_transforms) == 1
            names.add(subject.applied_transforms[-1].name)
        assert names == {"Gamma", "Flip"}

    def test_per_instance_false_is_batch_wide(self) -> None:
        torch.manual_seed(0)
        batch = self._batch(batch_size=8)
        transform = tio.OneOf(
            [tio.Gamma(log_gamma=0.5), tio.Flip(axes=(0,))],
            per_instance=False,
        )
        result = transform(batch)
        names = {subject.applied_transforms[-1].name for subject in result.unbatch()}
        assert len(names) == 1

    def test_single_subject_unaffected(self) -> None:
        subject = _make_subject()
        result = tio.OneOf([tio.Gamma(log_gamma=0.5), tio.Flip(axes=(0,))])(subject)
        assert len(result.applied_transforms) == 1

    def test_history_composes_after_oneof(self) -> None:
        torch.manual_seed(0)
        batch = self._batch(batch_size=16)
        pipeline = tio.Compose(
            [
                tio.OneOf([tio.Gamma(log_gamma=0.5), tio.Flip(axes=(0,))]),
                tio.Gamma(log_gamma=0.2),
            ]
        )
        result = pipeline(batch)
        for subject in result.unbatch():
            names = [trace.name for trace in subject.applied_transforms]
            assert len(names) == 2
            assert names[-1] == "Gamma"

    def test_p_gates_some_elements(self) -> None:
        torch.manual_seed(0)
        batch = self._batch(batch_size=64)
        transform = tio.OneOf([tio.Gamma(log_gamma=0.8)], p=0.5)
        result = transform(batch)
        applied = [len(subject.applied_transforms) == 1 for subject in result.unbatch()]
        assert any(applied)
        assert not all(applied)

    def test_per_element_inverse_restores(self) -> None:
        torch.manual_seed(0)
        batch = self._batch(batch_size=16)
        original = batch.t1.data.clone()
        transform = tio.OneOf([tio.Flip(axes=(0,)), tio.Flip(axes=(1,))])
        result = transform(batch)
        restored = result.apply_inverse_transform()
        torch.testing.assert_close(restored.t1.data, original)

    def test_functional_inverse_restores_per_element(self) -> None:
        torch.manual_seed(0)
        batch = self._batch(batch_size=16)
        original = batch.t1.data.clone()
        transform = tio.OneOf([tio.Flip(axes=(0,)), tio.Flip(axes=(1,))])
        result = transform(batch)
        restored = tio.apply_inverse_transform(result)
        torch.testing.assert_close(restored.t1.data, original)

    def test_get_inverse_transform_raises_for_per_element(self) -> None:
        torch.manual_seed(0)
        batch = self._batch(batch_size=4)
        result = tio.OneOf([tio.Flip(axes=(0,))])(batch)
        with pytest.raises(RuntimeError, match="per-element"):
            result.get_inverse_transform()

    def test_clear_history_clears_per_element(self) -> None:
        torch.manual_seed(0)
        batch = self._batch(batch_size=4)
        result = tio.OneOf([tio.Flip(axes=(0,))])(batch)
        result.clear_history()
        assert result._per_element_history is None
        for subject in result.unbatch():
            assert subject.applied_transforms == []

    def test_p_zero_is_noop_preserving_history(self) -> None:
        torch.manual_seed(0)
        batch = self._batch(batch_size=4)
        original = batch.t1.data.clone()
        # A prior shared, invertible transform.
        flipped = tio.Flip(axes=(0,))(batch)
        result = tio.OneOf([tio.Flip(axes=(1,))], p=0.0)(flipped)
        torch.testing.assert_close(result.t1.data, flipped.t1.data)
        assert result._per_element_history is None
        # The shared Flip history is intact and still invertible as a batch.
        restored = result.apply_inverse_transform()
        torch.testing.assert_close(restored.t1.data, original)
