"""Tests for RemoveLabels, KeepLargestComponent, Swap, HistogramStandardization."""

from __future__ import annotations

import warnings

import pytest
import torch

import torchio as tio
from torchio.transforms.histogram_standardization import compute_histogram_landmarks


def _make_subject(with_label: bool = True) -> tio.Subject:
    data = torch.rand(1, 10, 10, 10) * 100
    kwargs: dict = {"t1": tio.ScalarImage(data)}
    if with_label:
        seg = torch.zeros(1, 10, 10, 10, dtype=torch.float32)
        seg[0, 2:5, 2:5, 2:5] = 1
        seg[0, 6:9, 6:9, 6:9] = 2
        kwargs["seg"] = tio.LabelMap(seg)
    return tio.Subject(**kwargs)


# ── RemoveLabels ─────────────────────────────────────────────────────


class TestRemoveLabels:
    def test_removes_specified_labels(self) -> None:
        subject = _make_subject()
        result = tio.RemoveLabels([2])(subject)
        assert 2 not in result.seg.data.unique().tolist()
        assert 1 in result.seg.data.unique().tolist()

    def test_removes_multiple_labels(self) -> None:
        subject = _make_subject()
        result = tio.RemoveLabels([1, 2])(subject)
        unique = result.seg.data.unique().tolist()
        assert unique == [0.0]

    def test_custom_background(self) -> None:
        subject = _make_subject()
        result = tio.RemoveLabels([1], background_label=99)(subject)
        assert 1 not in result.seg.data.unique().tolist()
        assert 99 in result.seg.data.unique().tolist()

    def test_leaves_scalar_unchanged(self) -> None:
        subject = _make_subject()
        original = subject.t1.data.clone()
        result = tio.RemoveLabels([1])(subject)
        torch.testing.assert_close(result.t1.data, original)

    def test_noop_when_label_absent(self) -> None:
        subject = _make_subject()
        original = subject.seg.data.clone()
        result = tio.RemoveLabels([99])(subject)
        torch.testing.assert_close(result.seg.data, original)


# ── KeepLargestComponent ─────────────────────────────────────────────


class TestKeepLargestComponent:
    def test_keeps_largest_binary(self) -> None:
        seg = torch.zeros(1, 20, 20, 20, dtype=torch.float32)
        seg[0, 1:3, 1:3, 1:3] = 1  # small component (8 voxels)
        seg[0, 10:18, 10:18, 10:18] = 1  # large component (512 voxels)
        subject = tio.Subject(seg=tio.LabelMap(seg))
        result = tio.KeepLargestComponent()(subject)
        # Small component should be removed.
        assert result.seg.data[0, 2, 2, 2] == 0
        # Large component should remain.
        assert result.seg.data[0, 14, 14, 14] == 1

    def test_multi_label(self) -> None:
        seg = torch.zeros(1, 20, 20, 20, dtype=torch.float32)
        # Label 1: two components (sizes 8 and 27)
        seg[0, 0:2, 0:2, 0:2] = 1
        seg[0, 10:13, 10:13, 10:13] = 1
        # Label 2: one component
        seg[0, 5:8, 5:8, 5:8] = 2
        subject = tio.Subject(seg=tio.LabelMap(seg))
        result = tio.KeepLargestComponent()(subject)
        # Small label-1 component removed.
        assert result.seg.data[0, 1, 1, 1] == 0
        # Large label-1 component kept.
        assert result.seg.data[0, 11, 11, 11] == 1
        # Label-2 (only component) kept.
        assert result.seg.data[0, 6, 6, 6] == 2

    def test_specific_labels(self) -> None:
        seg = torch.zeros(1, 20, 20, 20, dtype=torch.float32)
        seg[0, 0:2, 0:2, 0:2] = 1  # small
        seg[0, 10:13, 10:13, 10:13] = 1  # large
        seg[0, 5:7, 5:7, 5:7] = 2  # small
        seg[0, 15:19, 15:19, 15:19] = 2  # large
        subject = tio.Subject(seg=tio.LabelMap(seg))
        result = tio.KeepLargestComponent(labels=[1])(subject)
        # Label 1 small removed.
        assert result.seg.data[0, 1, 1, 1] == 0
        # Label 2 small kept (not in filter list).
        assert result.seg.data[0, 6, 6, 6] == 2

    def test_multichannel_raises(self) -> None:
        seg = torch.zeros(2, 10, 10, 10, dtype=torch.float32)
        subject = tio.Subject(seg=tio.LabelMap(seg))
        with pytest.raises(RuntimeError, match="single-channel"):
            tio.KeepLargestComponent()(subject)

    def test_leaves_scalar_unchanged(self) -> None:
        subject = _make_subject()
        original = subject.t1.data.clone()
        result = tio.KeepLargestComponent()(subject)
        torch.testing.assert_close(result.t1.data, original)

    def test_face_connectivity(self) -> None:
        # Two cubes touching at corners only (not face-connected).
        seg = torch.zeros(1, 10, 10, 10, dtype=torch.float32)
        seg[0, 0:3, 0:3, 0:3] = 1
        seg[0, 3:7, 3:7, 3:7] = 1
        subject = tio.Subject(seg=tio.LabelMap(seg))
        # With fully_connected=True (26-conn), they are one component.
        result_26 = tio.KeepLargestComponent(fully_connected=True)(subject)
        assert result_26.seg.data[0, 1, 1, 1] == 1
        assert result_26.seg.data[0, 5, 5, 5] == 1


# ── Swap ─────────────────────────────────────────────────────────────


class TestSwap:
    def test_changes_data(self) -> None:
        subject = _make_subject(with_label=False)
        original = subject.t1.data.clone()
        result = tio.Swap(patch_size=3, num_iterations=10)(subject)
        assert not torch.allclose(result.t1.data, original)

    def test_preserves_shape(self) -> None:
        subject = _make_subject(with_label=False)
        result = tio.Swap(patch_size=3, num_iterations=5)(subject)
        assert result.t1.data.shape == subject.t1.data.shape

    def test_warns_with_labels(self) -> None:
        subject = _make_subject()
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            tio.Swap(patch_size=3, num_iterations=1)(subject)
            assert any("LabelMap" in str(warning.message) for warning in w)

    def test_patch_too_large_raises(self) -> None:
        subject = _make_subject(with_label=False)
        with pytest.raises(ValueError, match="cannot be larger"):
            tio.Swap(patch_size=100, num_iterations=1)(subject)

    def test_single_iteration(self) -> None:
        subject = _make_subject(with_label=False)
        result = tio.Swap(patch_size=3, num_iterations=1)(subject)
        assert result.t1.data.shape == subject.t1.data.shape


# ── HistogramStandardization ─────────────────────────────────────────


class TestHistogramStandardization:
    @staticmethod
    def _make_images(n: int = 5) -> list[tio.ScalarImage]:
        """Create synthetic training images."""
        images = []
        for i in range(n):
            data = torch.randn(1, 10, 10, 10) * (10 + i) + (50 + i * 5)
            images.append(tio.ScalarImage(data))
        return images

    def test_compute_landmarks(self) -> None:
        images = self._make_images()
        landmarks = compute_histogram_landmarks(images)
        assert landmarks.ndim == 1
        assert len(landmarks) > 2

    def test_landmarks_monotonic(self) -> None:
        images = self._make_images(n=10)
        landmarks = compute_histogram_landmarks(images)
        diffs = torch.diff(landmarks)
        assert (diffs >= -1e-5).all(), "Landmarks should be approximately monotonic"

    def test_apply_changes_data(self) -> None:
        images = self._make_images()
        landmarks = compute_histogram_landmarks(images)
        subject = _make_subject(with_label=False)
        original = subject.t1.data.clone()
        result = tio.HistogramStandardization(landmarks)(subject)
        assert not torch.allclose(result.t1.data, original)

    def test_apply_with_masking(self) -> None:
        images = self._make_images()
        mask_fn = lambda x: x > x.median()  # noqa: E731
        landmarks = compute_histogram_landmarks(
            images,
            masking_method=mask_fn,
        )
        assert landmarks.ndim == 1

    def test_leaves_labels_unchanged(self) -> None:
        images = self._make_images()
        landmarks = compute_histogram_landmarks(images)
        subject = _make_subject()
        original_seg = subject.seg.data.clone()
        result = tio.HistogramStandardization(landmarks)(subject)
        torch.testing.assert_close(result.seg.data, original_seg)

    def test_landmark_count_mismatch_raises(self) -> None:
        landmarks = torch.linspace(0, 100, 5)  # Wrong count
        subject = _make_subject(with_label=False)
        with pytest.raises(ValueError, match="does not match"):
            tio.HistogramStandardization(landmarks)(subject)

    def test_custom_quantiles(self) -> None:
        images = self._make_images()
        quantiles = (0.01, 0.25, 0.5, 0.75, 0.99)
        landmarks = compute_histogram_landmarks(
            images,
            quantiles=quantiles,
        )
        assert len(landmarks) == 5

    def test_too_few_quantiles_raises(self) -> None:
        images = self._make_images()
        with pytest.raises(ValueError, match="at least 2"):
            compute_histogram_landmarks(images, quantiles=(0.5,))


# ── Exports ──────────────────────────────────────────────────────────


class TestBatch3Exports:
    @pytest.mark.parametrize(
        "name",
        [
            "HistogramStandardization",
            "KeepLargestComponent",
            "RemoveLabels",
            "Swap",
        ],
    )
    def test_top_level_export(self, name: str) -> None:
        assert hasattr(tio, name)
