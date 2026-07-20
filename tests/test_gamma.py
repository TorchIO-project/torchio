"""Tests for Gamma transform."""

from __future__ import annotations

import numpy as np
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


class TestGamma:
    def test_changes_data(self) -> None:
        subject = _make_subject(with_label=False)
        original = subject.t1.data.clone()
        result = tio.Gamma(log_gamma=0.3)(subject)
        assert not torch.allclose(result.t1.data, original)

    def test_zero_log_gamma_is_identity(self) -> None:
        subject = _make_subject(with_label=False)
        original = subject.t1.data.clone()
        result = tio.Gamma(log_gamma=0.0)(subject)
        torch.testing.assert_close(result.t1.data, original)

    def test_inverse(self) -> None:
        subject = _make_subject(with_label=False)
        original = subject.t1.data.clone()
        transformed = tio.Gamma(log_gamma=0.2)(subject)
        restored = transformed.apply_inverse_transform()
        np.testing.assert_allclose(
            restored.t1.data.numpy(),
            original.numpy(),
            atol=1e-4,
        )

    def test_inverse_respects_include_scope(self) -> None:
        a = torch.arange(8.0).reshape(1, 2, 2, 2)
        b = torch.arange(100.0, 108.0).reshape(1, 2, 2, 2)
        subject = tio.Subject(
            a=tio.ScalarImage(a.clone()),
            b=tio.ScalarImage(b.clone()),
        )

        transformed = tio.Gamma(log_gamma=0.5, include=["a"])(subject)
        restored = transformed.apply_inverse_transform()

        torch.testing.assert_close(restored.a.data, a)
        torch.testing.assert_close(restored.b.data, b)

    def test_inverse_respects_exclude_scope(self) -> None:
        a = torch.arange(8.0).reshape(1, 2, 2, 2)
        b = torch.arange(100.0, 108.0).reshape(1, 2, 2, 2)
        subject = tio.Subject(
            a=tio.ScalarImage(a.clone()),
            b=tio.ScalarImage(b.clone()),
        )

        transformed = tio.Gamma(log_gamma=0.5, exclude=["b"])(subject)
        restored = transformed.apply_inverse_transform()

        torch.testing.assert_close(restored.a.data, a)
        torch.testing.assert_close(restored.b.data, b)

    def test_leaves_labels_unchanged(self) -> None:
        subject = _make_subject()
        original_seg = subject.seg.data.clone()
        result = tio.Gamma(log_gamma=0.3)(subject)
        torch.testing.assert_close(result.seg.data, original_seg)


class TestGammaPerInstance:
    def _batch(self, batch_size: int = 6) -> tio.SubjectsBatch:
        subjects = [
            tio.Subject(t1=tio.ScalarImage(torch.rand(1, 8, 8, 8) + 0.1))
            for _ in range(batch_size)
        ]
        return tio.SubjectsBatch.from_subjects(subjects)

    def test_per_instance_default_differs_across_batch(self) -> None:
        """With a random range, each element gets its own gamma."""
        torch.manual_seed(0)
        batch = self._batch()
        transform = tio.Gamma(log_gamma=(0.2, 0.8))
        result = transform(batch)
        log_gammas = [history[-1].params["log_gamma"] for history in result.histories]
        assert all(isinstance(value, float) for value in log_gammas)
        assert len(set(log_gammas)) > 1

    def test_per_instance_false_is_shared(self) -> None:
        torch.manual_seed(0)
        batch = self._batch()
        transform = tio.Gamma(log_gamma=(0.2, 0.8), per_instance=False)
        result = transform(batch)
        params = result.applied_transforms[-1].params
        assert isinstance(params["log_gamma"], float)

    def test_single_subject_keeps_scalar_params(self) -> None:
        subject = tio.Subject(t1=tio.ScalarImage(torch.rand(1, 8, 8, 8) + 0.1))
        result = tio.Gamma(log_gamma=(0.2, 0.8))(subject)
        assert isinstance(result.applied_transforms[-1].params["log_gamma"], float)

    def test_per_instance_values_applied(self) -> None:
        """Element-wise gamma is actually applied to each element."""
        torch.manual_seed(0)
        batch = self._batch(batch_size=4)
        original = batch.t1.data.clone()
        transform = tio.Gamma(log_gamma=(0.2, 0.8))
        result = transform(batch)
        log_gammas = [history[-1].params["log_gamma"] for history in result.histories]
        for i, log_gamma in enumerate(log_gammas):
            gamma = torch.tensor(log_gamma).exp()
            expected = original[i].sign() * original[i].abs().pow(gamma)
            torch.testing.assert_close(result.t1.data[i], expected)

    def test_per_instance_p_gates_some_elements(self) -> None:
        torch.manual_seed(0)
        batch = self._batch(batch_size=64)
        original = batch.t1.data.clone()
        transform = tio.Gamma(log_gamma=(0.5, 1.0), p=0.5)
        result = transform(batch)
        changed = [
            not torch.allclose(result.t1.data[i], original[i])
            for i in range(batch.batch_size)
        ]
        assert any(changed)
        assert not all(changed)

    def test_per_instance_p_masked_elements_have_no_history(self) -> None:
        torch.manual_seed(0)
        batch = self._batch(batch_size=32)
        original = batch.t1.data.clone()
        transform = tio.Gamma(log_gamma=(0.5, 1.0), p=0.5)
        result = transform(batch)
        subjects = result.unbatch()
        for i, subject in enumerate(subjects):
            changed = not torch.allclose(subject.t1.data, original[i])
            has_history = len(subject.applied_transforms) == 1
            assert changed == has_history

    def test_per_instance_inverse_round_trip(self) -> None:
        torch.manual_seed(0)
        batch = self._batch(batch_size=5)
        original = batch.t1.data.clone()
        transform = tio.Gamma(log_gamma=(0.2, 0.5))
        result = transform(batch)
        restored = result.apply_inverse_transform()
        torch.testing.assert_close(restored.t1.data, original, atol=1e-4, rtol=0)
