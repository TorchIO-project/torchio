"""Tests for SomeOf transform."""

from __future__ import annotations

import torch

import torchio as tio


def _make_subject() -> tio.Subject:
    return tio.Subject(
        t1=tio.ScalarImage(torch.rand(1, 10, 10, 10)),
    )


class TestSomeOf:
    def test_applies_subset(self) -> None:
        subject = _make_subject()
        transform = tio.SomeOf(
            [tio.Flip(axes=(0,)), tio.Gamma(log_gamma=0.0)],
            num_transforms=1,
        )
        result = transform(subject)
        assert result.t1.data.shape == subject.t1.data.shape

    def test_all_transforms(self) -> None:
        subject = _make_subject()
        transform = tio.SomeOf(
            [tio.Flip(axes=(0,)), tio.Gamma(log_gamma=0.0)],
            num_transforms=2,
        )
        result = transform(subject)
        assert result.t1.data.shape == subject.t1.data.shape

    def test_range_num_transforms(self) -> None:
        subject = _make_subject()
        transform = tio.SomeOf(
            [
                tio.Flip(axes=(0,)),
                tio.Gamma(log_gamma=0.0),
                tio.Blur(std=0.0),
            ],
            num_transforms=(1, 3),
        )
        result = transform(subject)
        assert result.t1.data.shape == subject.t1.data.shape

    def test_history_recorded(self) -> None:
        subject = _make_subject()
        transform = tio.SomeOf(
            [tio.Flip(axes=(0,))],
            num_transforms=1,
        )
        result = transform(subject)
        assert len(result.applied_transforms) > 0
