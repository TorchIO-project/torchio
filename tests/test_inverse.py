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

    def test_does_not_mutate_subject(self) -> None:
        subject = _make_subject()
        transformed = tio.Flip(axes=(0,))(subject)
        snapshot = transformed.t1.data.clone()
        restored = transformed.apply_inverse_transform()
        # Inverting must not modify the transformed input in place.
        torch.testing.assert_close(transformed.t1.data, snapshot)
        assert restored is not transformed

    def test_does_not_mutate_batch(self) -> None:
        data = torch.rand(1, 16, 16, 16)
        batch = tio.SubjectsBatch.from_subjects(
            [tio.Subject(t1=tio.ScalarImage(data.clone())) for _ in range(3)]
        )
        transformed = tio.Affine(degrees=(0, 0, (10, 45)), default_pad_value=0.0)(batch)
        snapshot = transformed.t1.data.clone()
        restored = transformed.apply_inverse_transform()
        torch.testing.assert_close(transformed.t1.data, snapshot)
        assert not torch.allclose(restored.t1.data, transformed.t1.data)

    def test_does_not_mutate_per_element_batch(self) -> None:
        torch.manual_seed(0)
        data = torch.rand(1, 16, 16, 16)
        batch = tio.SubjectsBatch.from_subjects(
            [tio.Subject(t1=tio.ScalarImage(data.clone())) for _ in range(8)]
        )
        transformed = tio.OneOf([tio.Flip(axes=(0,)), tio.Flip(axes=(1,))])(batch)
        snapshot = transformed.t1.data.clone()
        transformed.apply_inverse_transform()
        torch.testing.assert_close(transformed.t1.data, snapshot)

    def test_standalone_function_does_not_mutate(self) -> None:
        data = torch.rand(1, 16, 16, 16)
        batch = tio.SubjectsBatch.from_subjects(
            [tio.Subject(t1=tio.ScalarImage(data.clone())) for _ in range(3)]
        )
        transformed = tio.Affine(degrees=(0, 0, (10, 45)), default_pad_value=0.0)(batch)
        snapshot = transformed.t1.data.clone()
        tio.apply_inverse_transform(transformed)
        torch.testing.assert_close(transformed.t1.data, snapshot)

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

    def test_missing_trace_image_skips_inverse(self) -> None:
        a = torch.arange(8.0).reshape(1, 2, 2, 2)
        b = torch.arange(100.0, 108.0).reshape(1, 2, 2, 2)
        subject = tio.Subject(
            a=tio.ScalarImage(a.clone()),
            b=tio.ScalarImage(b.clone()),
        )

        transformed = tio.Gamma(log_gamma=0.5, include=["a"])(subject)
        current = tio.Subject(b=transformed.b)
        current.applied_transforms = transformed.applied_transforms
        restored = current.apply_inverse_transform()

        torch.testing.assert_close(restored.b.data, b)
