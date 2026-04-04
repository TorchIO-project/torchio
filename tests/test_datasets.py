"""Tests for built-in datasets and new Image features."""

from __future__ import annotations

import pytest
import torch

import torchio as tio
from torchio.datasets.zone_plate import ZonePlate
from torchio.download import check_integrity
from torchio.download import compress
from torchio.download import get_torchio_cache_dir
from torchio.io import read_matrix
from torchio.io import write_matrix

# --- ZonePlate ---------------------------------------------------------------


class TestZonePlate:
    def test_default_size(self):
        zp = ZonePlate(size=11)
        assert zp.size == 11
        img = zp.image
        assert img.data.shape == (1, 11, 11, 11)

    def test_custom_size(self):
        zp = ZonePlate(size=11)
        img = zp.image
        assert img.spatial_shape == (11, 11, 11)

    def test_even_size(self):
        zp = ZonePlate(size=10)
        img = zp.image
        assert img.spatial_shape == (10, 10, 10)

    def test_minimum_size(self):
        zp = ZonePlate(size=3)
        img = zp.image
        assert img.spatial_shape == (3, 3, 3)

    def test_too_small(self):
        with pytest.raises(ValueError, match="at least 3"):
            ZonePlate(size=2)

    def test_is_subject(self):
        zp = ZonePlate(size=5)
        assert isinstance(zp, tio.Subject)

    def test_has_scalar_image(self):
        zp = ZonePlate(size=5)
        assert isinstance(zp.image, tio.ScalarImage)

    def test_affine_origin(self):
        zp = ZonePlate(size=11)
        img = zp.image
        origin = img.affine.origin
        assert origin == pytest.approx((-5.0, -5.0, -5.0))


# --- Download utilities ------------------------------------------------------


class TestDownloadUtils:
    def test_cache_dir(self):
        d = get_torchio_cache_dir()
        assert d.name == "torchio"
        assert d.is_absolute()

    def test_compress(self, tmp_path):
        inp = tmp_path / "test.nii"
        inp.write_bytes(b"fake nifti content " * 100)
        out = compress(inp)
        assert out.suffix == ".gz"
        assert out.exists()
        assert out.stat().st_size < inp.stat().st_size

    def test_compress_custom_output(self, tmp_path):
        inp = tmp_path / "test.nii"
        inp.write_bytes(b"hello" * 50)
        custom = tmp_path / "custom.nii.gz"
        result = compress(inp, custom)
        assert result == custom
        assert custom.exists()

    def test_check_integrity_missing(self, tmp_path):
        assert not check_integrity(tmp_path / "nonexistent.bin")

    def test_check_integrity_exists_no_md5(self, tmp_path):
        f = tmp_path / "file.bin"
        f.write_bytes(b"data")
        assert check_integrity(f)


# --- read_matrix / write_matrix ----------------------------------------------


class TestMatrixIO:
    def test_roundtrip_tfm(self, tmp_path):
        matrix = torch.eye(4, dtype=torch.float64)
        matrix[0, 3] = 10.0
        matrix[1, 3] = -5.0
        path = tmp_path / "transform.tfm"
        write_matrix(matrix, path)
        loaded = read_matrix(path)
        torch.testing.assert_close(loaded, matrix, atol=1e-6, rtol=1e-6)

    def test_roundtrip_txt(self, tmp_path):
        matrix = torch.eye(4, dtype=torch.float64)
        matrix[2, 3] = 7.0
        path = tmp_path / "transform.txt"
        write_matrix(matrix, path)
        loaded = read_matrix(path)
        torch.testing.assert_close(loaded, matrix, atol=1e-6, rtol=1e-6)

    def test_unsupported_suffix(self, tmp_path):
        with pytest.raises(ValueError, match="Unknown suffix"):
            read_matrix(tmp_path / "bad.xyz")

    def test_write_unsupported_suffix(self, tmp_path):
        with pytest.raises(ValueError, match="Unknown suffix"):
            write_matrix(torch.eye(4), tmp_path / "bad.xyz")


# --- channels_last -----------------------------------------------------------


class TestChannelsLast:
    def test_from_tensor_channels_last(self):
        data = torch.randn(10, 20, 30, 3)  # (I, J, K, C)
        img = tio.ScalarImage.from_tensor(data, channels_last=True)
        assert img.data.shape == (3, 10, 20, 30)

    def test_from_tensor_channels_first(self):
        data = torch.randn(3, 10, 20, 30)  # (C, I, J, K)
        img = tio.ScalarImage.from_tensor(data, channels_last=False)
        assert img.data.shape == (3, 10, 20, 30)

    def test_channels_last_load(self, tmp_path):
        # Create a standard 3D NIfTI image, then the reader returns (1,I,J,K)
        # We test that channels_last permutes from (I,J,K,C) -> (C,I,J,K)
        # by using from_tensor with explicit channels_last data
        data = torch.randn(5, 6, 7, 3)  # (I, J, K, C)
        img = tio.LabelMap.from_tensor(data, channels_last=True)
        assert img.data.shape == (3, 5, 6, 7)
