"""Tests for IntensityInversion transform."""

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


class TestIntensityInversion:
    def test_changes_data(self) -> None:
        subject = _make_subject(with_label=False)
        original = subject.t1.data.clone()
        result = tio.IntensityInversion()(subject)
        assert not torch.allclose(result.t1.data, original)

    def test_double_application_is_identity(self) -> None:
        subject = _make_subject(with_label=False)
        original = subject.t1.data.clone()
        once = tio.IntensityInversion()(subject)
        twice = tio.IntensityInversion()(once)
        # Two `max - x + min` passes accumulate float32 rounding, so compare
        # with the same tolerance the codebase uses for inverse round-trips.
        torch.testing.assert_close(twice.t1.data, original, atol=1e-4, rtol=1e-5)

    def test_p_zero_is_noop(self) -> None:
        subject = _make_subject(with_label=False)
        original = subject.t1.data.clone()
        result = tio.IntensityInversion(p=0.0)(subject)
        torch.testing.assert_close(result.t1.data, original)

    def test_range_is_preserved(self) -> None:
        subject = _make_subject(with_label=False)
        original = subject.t1.data.clone()
        result = tio.IntensityInversion()(subject)
        torch.testing.assert_close(
            result.t1.data.min(), original.min(), atol=1e-4, rtol=1e-5
        )
        torch.testing.assert_close(
            result.t1.data.max(), original.max(), atol=1e-4, rtol=1e-5
        )

    def test_dark_and_bright_are_swapped(self) -> None:
        subject = _make_subject(with_label=False)
        original = subject.t1.data.clone()
        result = tio.IntensityInversion()(subject).t1.data
        argmin = original.flatten().argmin()
        argmax = original.flatten().argmax()
        # The brightest voxel becomes the darkest and vice versa.
        torch.testing.assert_close(
            result.flatten()[argmax], original.min(), atol=1e-4, rtol=1e-5
        )
        torch.testing.assert_close(
            result.flatten()[argmin], original.max(), atol=1e-4, rtol=1e-5
        )

    def test_per_channel(self) -> None:
        # Two channels with very different ranges, to prove per-channel scope.
        channel0 = torch.linspace(0, 100, 1000).reshape(1, 10, 10, 10)
        channel1 = torch.linspace(-5, 5, 1000).reshape(1, 10, 10, 10)
        data = torch.cat([channel0, channel1], dim=0)
        subject = tio.Subject(t1=tio.ScalarImage(data))
        result = tio.IntensityInversion()(subject).t1.data
        for channel in range(2):
            original_channel = data[channel]
            inverted_channel = result[channel]
            expected = (
                original_channel.max() - original_channel + original_channel.min()
            )
            # Same formula as apply_transform, so this is bit-exact.
            torch.testing.assert_close(inverted_channel, expected)
            # Each channel stays within its own original range.
            torch.testing.assert_close(
                inverted_channel.min(), original_channel.min(), atol=1e-4, rtol=1e-5
            )
            torch.testing.assert_close(
                inverted_channel.max(), original_channel.max(), atol=1e-4, rtol=1e-5
            )

    def test_inverse(self) -> None:
        subject = _make_subject(with_label=False)
        original = subject.t1.data.clone()
        transformed = tio.IntensityInversion()(subject)
        restored = transformed.apply_inverse_transform()
        np.testing.assert_allclose(
            restored.t1.data.numpy(),
            original.numpy(),
            atol=1e-4,
        )

    def test_inverse_respects_include(self) -> None:
        # Forward inverts only t1; the inverse must not touch t2 (which was
        # never inverted). Guards against the history losing include/exclude.
        a = torch.rand(1, 8, 8, 8) * 100
        b = torch.rand(1, 8, 8, 8) * 100
        subject = tio.Subject(t1=tio.ScalarImage(a), t2=tio.ScalarImage(b))
        original_a = subject.t1.data.clone()
        original_b = subject.t2.data.clone()
        transformed = tio.IntensityInversion(include=["t1"])(subject)
        # Only t1 changed in the forward pass.
        assert not torch.allclose(transformed.t1.data, original_a)
        torch.testing.assert_close(transformed.t2.data, original_b)
        restored = transformed.apply_inverse_transform()
        # t1 round-trips through two inversions (float32 rounding); t2 is an
        # untouched copy and must match exactly.
        torch.testing.assert_close(restored.t1.data, original_a, atol=1e-4, rtol=1e-5)
        torch.testing.assert_close(restored.t2.data, original_b)

    def test_leaves_labels_unchanged(self) -> None:
        subject = _make_subject()
        original_seg = subject.seg.data.clone()
        result = tio.IntensityInversion()(subject)
        torch.testing.assert_close(result.seg.data, original_seg)
