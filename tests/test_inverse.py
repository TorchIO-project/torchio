"""Tests for the inverse transform module."""

from __future__ import annotations

import torch

import torchio as tio


def _make_subject() -> tio.Subject:
    return tio.Subject(
        t1=tio.ScalarImage(torch.rand(1, 10, 10, 10)),
        seg=tio.LabelMap(torch.zeros(1, 10, 10, 10)),
    )


class TestApplyInverseTransform:
    def test_flip_inverse(self) -> None:
        subject = _make_subject()
        original = subject.t1.data.clone()
        transformed = tio.Flip(axes=(0,))(subject)
        restored = transformed.apply_inverse_transform()
        torch.testing.assert_close(restored.t1.data, original)

    def test_compose_inverse(self) -> None:
        subject = _make_subject()
        original = subject.t1.data.clone()
        pipeline = tio.Compose(
            [
                tio.Flip(axes=(0,)),
                tio.Flip(axes=(1,)),
            ]
        )
        transformed = pipeline(subject)
        restored = transformed.apply_inverse_transform()
        torch.testing.assert_close(restored.t1.data, original)

    def test_ignore_intensity(self) -> None:
        subject = _make_subject()
        original = subject.t1.data.clone()
        pipeline = tio.Compose(
            [
                tio.Flip(axes=(0,)),
                tio.Noise(std=0.1),
            ]
        )
        transformed = pipeline(subject)
        restored = transformed.apply_inverse_transform(
            ignore_intensity=True,
        )
        # Shape restored, flip inverted, noise skipped.
        assert restored.t1.data.shape == original.shape

    def test_get_inverse_transform(self) -> None:
        subject = _make_subject()
        transformed = tio.Flip(axes=(0,))(subject)
        inverse = transformed.get_inverse_transform()
        assert inverse is not None

    def test_standalone_function(self) -> None:
        subject = _make_subject()
        original = subject.t1.data.clone()
        transformed = tio.Flip(axes=(0,))(subject)
        restored = tio.apply_inverse_transform(transformed)
        torch.testing.assert_close(restored.t1.data, original)

    def test_no_history(self) -> None:
        """Subject with no transforms should return itself."""
        subject = _make_subject()
        original = subject.t1.data.clone()
        restored = subject.apply_inverse_transform()
        torch.testing.assert_close(restored.t1.data, original)
