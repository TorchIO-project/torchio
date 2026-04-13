"""Tests for Compose transform."""

from __future__ import annotations

import torch

import torchio as tio


def _make_subject() -> tio.Subject:
    return tio.Subject(
        t1=tio.ScalarImage(torch.rand(1, 10, 10, 10)),
        seg=tio.LabelMap(torch.zeros(1, 10, 10, 10)),
    )


class TestCompose:
    def test_identity_compose(self) -> None:
        subject = _make_subject()
        original = subject.t1.data.clone()
        result = tio.Compose([])(subject)
        torch.testing.assert_close(result.t1.data, original)

    def test_single_transform(self) -> None:
        subject = _make_subject()
        result = tio.Compose([tio.Flip(axes=(0,))])(subject)
        assert result.t1.data.shape == subject.t1.data.shape

    def test_multiple_transforms(self) -> None:
        subject = _make_subject()
        pipeline = tio.Compose(
            [
                tio.Flip(axes=(0,)),
                tio.Gamma(log_gamma=0.0),
            ]
        )
        result = pipeline(subject)
        assert result.t1.data.shape == subject.t1.data.shape

    def test_nested_compose(self) -> None:
        subject = _make_subject()
        inner = tio.Compose([tio.Flip(axes=(0,))], copy=False)
        outer = tio.Compose([inner])
        result = outer(subject)
        assert result.t1.data.shape == subject.t1.data.shape

    def test_copy_default(self) -> None:
        """Compose deep-copies input by default."""
        subject = _make_subject()
        original_data = subject.t1.data.clone()
        tio.Compose([tio.Gamma(log_gamma=0.5)])(subject)
        torch.testing.assert_close(subject.t1.data, original_data)

    def test_no_copy(self) -> None:
        subject = _make_subject()
        result = tio.Compose([tio.Gamma(log_gamma=0.0)], copy=False)(subject)
        assert result.t1.data.shape == subject.t1.data.shape

    def test_history_recorded(self) -> None:
        subject = _make_subject()
        result = tio.Compose([tio.Flip(axes=(0,))])(subject)
        assert len(result.applied_transforms) > 0
