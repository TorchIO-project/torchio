"""Tests for remote/URL/file-like Image loading via fsspec."""

from __future__ import annotations

import io
from pathlib import Path

import nibabel as nib
import numpy as np
import pytest
import torch

import torchio as tio


@pytest.fixture
def nifti_path(tmp_path: Path) -> Path:
    path = tmp_path / "test.nii.gz"
    data = np.random.rand(8, 8, 8).astype(np.float32)
    nib.save(nib.Nifti1Image(data, np.eye(4)), path)
    return path


class TestLocalPath:
    def test_str_path(self, nifti_path: Path) -> None:
        image = tio.ScalarImage(str(nifti_path))
        assert image.shape == (1, 8, 8, 8)

    def test_path_object(self, nifti_path: Path) -> None:
        image = tio.ScalarImage(nifti_path)
        assert image.shape == (1, 8, 8, 8)


class TestFileUrl:
    def test_file_uri(self, nifti_path: Path) -> None:
        """file:// URIs should work via fsspec."""
        uri = f"file://{nifti_path}"
        image = tio.ScalarImage(uri)
        assert image.shape == (1, 8, 8, 8)


class TestFileLike:
    def test_bytes_io(self, nifti_path: Path) -> None:
        """BytesIO should work by materializing to a temp file."""
        buf = io.BytesIO(nifti_path.read_bytes())
        image = tio.ScalarImage(buf, suffix=".nii.gz")
        assert image.shape == (1, 8, 8, 8)

    def test_open_file(self, nifti_path: Path) -> None:
        """An open binary file should work."""
        with open(nifti_path, "rb") as f:
            image = tio.ScalarImage(f, suffix=".nii.gz")
            assert image.shape == (1, 8, 8, 8)


class TestFsspec:
    def test_local_fsspec(self, nifti_path: Path) -> None:
        """fsspec local filesystem should work."""
        import fsspec

        of = fsspec.open(str(nifti_path), mode="rb")
        image = tio.ScalarImage(of)
        assert image.shape == (1, 8, 8, 8)
