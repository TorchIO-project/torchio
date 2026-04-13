"""Tests for HistogramStandardization transform."""

from __future__ import annotations

from pathlib import Path

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


class TestHistogramStandardization:
    @staticmethod
    def _make_images(n: int = 5) -> list[tio.ScalarImage]:
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
        assert (diffs >= -1e-5).all()

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
        landmarks = torch.linspace(0, 100, 5)
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


class TestHistogramStandardizationEdgeCases:
    def test_quantiles_out_of_range_raises(self) -> None:
        images = [tio.ScalarImage(torch.randn(1, 5, 5, 5)) for _ in range(3)]
        with pytest.raises(ValueError, match="\\[0, 1\\]"):
            compute_histogram_landmarks(images, quantiles=(-0.1, 0.5, 1.1))

    def test_cutoff_not_in_quantiles_raises(self) -> None:
        images = [tio.ScalarImage(torch.randn(1, 5, 5, 5)) for _ in range(3)]
        with pytest.raises(ValueError, match="Cutoff"):
            compute_histogram_landmarks(
                images, quantiles=(0.25, 0.5, 0.75), cutoff=(0.01, 0.99)
            )

    def test_load_landmarks_from_npy(self, tmp_path: Path) -> None:
        import numpy as np

        arr = np.linspace(0, 100, 13).astype(np.float32)
        npy_path = tmp_path / "landmarks.npy"
        np.save(npy_path, arr)
        subject = _make_subject(with_label=False)
        result = tio.HistogramStandardization(npy_path)(subject)
        assert result.t1.data.shape == subject.t1.data.shape

    def test_load_landmarks_from_pt(self, tmp_path: Path) -> None:
        landmarks = torch.linspace(0, 100, 13)
        pt_path = tmp_path / "landmarks.pt"
        torch.save(landmarks, pt_path)
        subject = _make_subject(with_label=False)
        result = tio.HistogramStandardization(pt_path)(subject)
        assert result.t1.data.shape == subject.t1.data.shape

    def test_unsupported_format_raises(self, tmp_path: Path) -> None:
        bad_path = tmp_path / "landmarks.csv"
        bad_path.write_text("1,2,3")
        with pytest.raises(ValueError, match="Unsupported"):
            tio.HistogramStandardization(bad_path)

    def test_pt_with_wrong_type_raises(self, tmp_path: Path) -> None:
        pt_path = tmp_path / "landmarks.pt"
        torch.save({"not": "a tensor"}, pt_path)
        with pytest.raises(TypeError, match="Expected a Tensor"):
            tio.HistogramStandardization(pt_path)

    def test_load_from_path_string(self) -> None:
        images = [tio.ScalarImage(torch.randn(1, 5, 5, 5)) for _ in range(3)]
        landmarks = compute_histogram_landmarks(images)
        assert landmarks.ndim == 1
