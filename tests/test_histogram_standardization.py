"""Tests for HistogramStandardization transform."""

from __future__ import annotations

from pathlib import Path
from threading import Barrier

import numpy as np
import pytest
import torch

import torchio as tio
from torchio.transforms.intensity.histogram_standardization import (
    compute_histogram_landmarks,
)


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


class TestParallelLandmarks:
    @pytest.mark.parametrize("num_workers", [0, 1, 2, 8])
    @pytest.mark.parametrize("masked", [False, True])
    def test_matches_numpy_reference(self, num_workers: int, masked: bool) -> None:
        generator = torch.Generator().manual_seed(970)
        tensors = [
            torch.randn(2, 4, 5, 6, generator=generator) * (i + 1) + i for i in range(4)
        ]
        original = [tensor.clone() for tensor in tensors]
        quantiles = (0.05, 0.25, 0.5, 0.75, 0.95)
        mask_fn = (lambda x: x > 0) if masked else None
        database = np.stack(
            [
                np.percentile(
                    tensor[tensor > 0].numpy() if masked else tensor.numpy(),
                    np.array(quantiles) * 100,
                )
                for tensor in tensors
            ]
        )
        # Independently map each training image's cutoff range to [0, 100].
        expected = np.mean(
            (database - database[:, :1]) * 100 / (database[:, -1:] - database[:, :1]),
            axis=0,
        )
        actual = compute_histogram_landmarks(
            [tio.ScalarImage(tensor) for tensor in tensors],
            quantiles=quantiles,
            cutoff=(0.05, 0.95),
            masking_method=mask_fn,
            num_workers=num_workers,
        )
        torch.testing.assert_close(actual, torch.tensor(expected, dtype=torch.float32))
        for tensor, before in zip(tensors, original, strict=True):
            torch.testing.assert_close(tensor, before, rtol=0, atol=0)

    @pytest.mark.parametrize("num_workers", [1, 2, 8])
    def test_mixed_sources_match_sequential(
        self, tmp_path: Path, num_workers: int
    ) -> None:
        images = TestHistogramStandardization._make_images(n=4)
        expected = compute_histogram_landmarks(images)
        paths = [
            tmp_path / "plain.nii",
            tmp_path / "compressed.nii.gz",
            tmp_path / "lazy.nii",
        ]
        for image, path in zip(images[:3], paths, strict=True):
            image.save(path)
        lazy_image = tio.ScalarImage(paths[2])
        assert not lazy_image.is_loaded
        sources = [paths[0], str(paths[1]), lazy_image, images[3]]
        actual = compute_histogram_landmarks(sources, num_workers=num_workers)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        # The trained landmarks also produce identical standardized images.
        sequential = tio.HistogramStandardization(expected)(images[0])
        parallel = tio.HistogramStandardization(actual)(images[0])
        torch.testing.assert_close(parallel.data, sequential.data, rtol=0, atol=0)

    def test_masking_runs_concurrently(self) -> None:
        barrier = Barrier(2)
        images = TestHistogramStandardization._make_images(n=2)
        expected = compute_histogram_landmarks(images)

        def mask(tensor: torch.Tensor) -> torch.Tensor:
            # Both workers must enter the callable before either can finish.
            barrier.wait(timeout=10)
            return torch.ones_like(tensor, dtype=torch.bool)

        actual = compute_histogram_landmarks(images, masking_method=mask, num_workers=2)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    def test_single_image_with_multiple_workers(self) -> None:
        images = TestHistogramStandardization._make_images(n=1)
        expected = compute_histogram_landmarks(images)
        actual = compute_histogram_landmarks(images, num_workers=4)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    @pytest.mark.parametrize("num_workers", [0, 2])
    def test_reader_errors_propagate(self, tmp_path: Path, num_workers: int) -> None:
        with pytest.raises(FileNotFoundError):
            compute_histogram_landmarks(
                [tmp_path / "missing.nii"], num_workers=num_workers
            )

    def test_masking_errors_propagate(self) -> None:
        def broken_mask(tensor: torch.Tensor) -> torch.Tensor:
            raise RuntimeError("Failed to compute training mask")

        with pytest.raises(RuntimeError, match="Failed to compute training mask"):
            compute_histogram_landmarks(
                TestHistogramStandardization._make_images(n=2),
                masking_method=broken_mask,
                num_workers=2,
            )

    def test_negative_worker_count(self) -> None:
        with pytest.raises(ValueError, match="num_workers must be non-negative"):
            compute_histogram_landmarks([], num_workers=-1)
