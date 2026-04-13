"""Tests for OneOf transform."""

from __future__ import annotations

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
