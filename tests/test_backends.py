"""Tests for lazy image data backends."""

from __future__ import annotations

from pathlib import Path

import nibabel as nib
import numpy as np
import pytest
import torch

from torchio import ScalarImage
from torchio.data.backends import NibabelBackend
from torchio.data.backends import NumpyBackend
from torchio.data.backends import TensorBackend


class TestNumpyBackend:
    def test_shape(self):
        data = np.random.randn(1, 10, 12, 14).astype(np.float32)
        backend = NumpyBackend(data)
        assert backend.shape == (1, 10, 12, 14)

    def test_getitem(self):
        data = np.random.randn(1, 10, 12, 14).astype(np.float32)
        backend = NumpyBackend(data)
        sliced = backend[0, 2:5, 3:7, 4:8]
        np.testing.assert_array_equal(sliced, data[0, 2:5, 3:7, 4:8])

    def test_to_tensor(self):
        data = np.random.randn(1, 10, 12, 14).astype(np.float32)
        backend = NumpyBackend(data)
        tensor = backend.to_tensor()
        assert isinstance(tensor, torch.Tensor)
        np.testing.assert_array_equal(tensor.numpy(), data)

    def test_from_tensor(self):
        tensor = torch.randn(1, 10, 12, 14)
        backend = NumpyBackend(tensor.numpy())
        result = backend.to_tensor()
        assert torch.equal(result, tensor)


class TestNibabelBackend:
    @pytest.fixture
    def nifti_path(self, tmp_path: Path) -> Path:
        data = np.random.randn(10, 12, 14).astype(np.float32)
        nii = nib.Nifti1Image(data, np.diag([2.0, 2.0, 2.0, 1.0]))
        path = tmp_path / "test.nii.gz"
        nib.save(nii, path)
        return path

    @pytest.fixture
    def multichannel_nifti_path(self, tmp_path: Path) -> Path:
        data = np.random.randn(10, 12, 14, 3).astype(np.float32)
        nii = nib.Nifti1Image(data, np.eye(4))
        path = tmp_path / "multi.nii.gz"
        nib.save(nii, path)
        return path

    def test_shape_3d(self, nifti_path: Path):
        nii = nib.load(nifti_path)
        backend = NibabelBackend(nii)
        assert backend.shape == (1, 10, 12, 14)

    def test_shape_4d(self, multichannel_nifti_path: Path):
        nii = nib.load(multichannel_nifti_path)
        backend = NibabelBackend(nii)
        assert backend.shape == (3, 10, 12, 14)

    def test_affine(self, nifti_path: Path):
        nii = nib.load(nifti_path)
        backend = NibabelBackend(nii)
        np.testing.assert_array_equal(
            backend.affine,
            np.diag([2.0, 2.0, 2.0, 1.0]),
        )

    def test_to_tensor_3d(self, nifti_path: Path):
        nii = nib.load(nifti_path)
        backend = NibabelBackend(nii)
        tensor = backend.to_tensor()
        assert tensor.shape == (1, 10, 12, 14)
        assert tensor.dtype == torch.float32

    def test_to_tensor_4d(self, multichannel_nifti_path: Path):
        nii = nib.load(multichannel_nifti_path)
        backend = NibabelBackend(nii)
        tensor = backend.to_tensor()
        assert tensor.shape == (3, 10, 12, 14)

    def test_getitem_3d(self, nifti_path: Path):
        nii = nib.load(nifti_path)
        backend = NibabelBackend(nii)
        # Spatial slice in (I, J, K) space
        sliced = backend[:, 2:5, 3:7, 4:8]
        assert sliced.shape == (1, 3, 4, 4)

    def test_does_not_load_full_data_for_shape(self, nifti_path: Path):
        nii = nib.load(nifti_path)
        backend = NibabelBackend(nii)
        _ = backend.shape
        # dataobj should still be a proxy, not a loaded array
        assert not isinstance(nii.dataobj, np.ndarray)

    def test_invalid_ndim_raises(self, tmp_path: Path):
        data = np.random.randn(10, 12, 14, 3, 2).astype(np.float32)
        nii = nib.Nifti1Image(data, np.eye(4))
        path = tmp_path / "bad.nii.gz"
        nib.save(nii, path)
        nii = nib.load(path)
        with pytest.raises(ValueError, match="3D or 4D"):
            NibabelBackend(nii)


class TestImageWithBackends:
    def test_from_tensor_uses_tensor_backend(self):
        image = ScalarImage(torch.randn(1, 10, 10, 10))
        assert isinstance(image._backend, TensorBackend)

    def test_nifti_uses_nibabel_backend(self, tmp_path: Path):
        data = np.random.randn(10, 10, 10).astype(np.float32)
        nii = nib.Nifti1Image(data, np.eye(4))
        path = tmp_path / "test.nii.gz"
        nib.save(nii, path)

        image = ScalarImage(path)
        assert not image.is_loaded
        _ = image.data  # trigger load
        assert isinstance(image._backend, NibabelBackend)

    def test_shape_without_loading_uses_backend(self, tmp_path: Path):
        data = np.random.randn(10, 12, 14).astype(np.float32)
        nii = nib.Nifti1Image(data, np.eye(4))
        path = tmp_path / "test.nii.gz"
        nib.save(nii, path)

        image = ScalarImage(path)
        assert image.shape == (1, 10, 12, 14)
        # Backend should be created but data not materialized
        assert image._backend is not None
        assert image._data is None

    def test_dataobj_returns_backend(self, tmp_path: Path):
        data = np.random.randn(10, 10, 10).astype(np.float32)
        nii = nib.Nifti1Image(data, np.eye(4))
        path = tmp_path / "test.nii.gz"
        nib.save(nii, path)

        image = ScalarImage(path)
        dataobj = image.dataobj
        assert isinstance(dataobj, NibabelBackend)

    def test_dataobj_from_tensor(self):
        image = ScalarImage(torch.randn(1, 10, 10, 10))
        dataobj = image.dataobj
        assert isinstance(dataobj, TensorBackend)

    def test_data_caches_tensor(self, tmp_path: Path):
        data = np.random.randn(10, 10, 10).astype(np.float32)
        nii = nib.Nifti1Image(data, np.eye(4))
        path = tmp_path / "test.nii.gz"
        nib.save(nii, path)

        image = ScalarImage(path)
        tensor1 = image.data
        tensor2 = image.data
        assert tensor1 is tensor2  # same object, cached

    def test_lazy_slice_via_dataobj(self, tmp_path: Path):
        data = np.random.randn(10, 12, 14).astype(np.float32)
        nii = nib.Nifti1Image(data, np.eye(4))
        path = tmp_path / "test.nii.gz"
        nib.save(nii, path)

        image = ScalarImage(path)
        backend = image.dataobj
        sliced = backend[:, 2:5, 3:7, 4:8]
        assert sliced.shape == (1, 3, 4, 4)
        assert image._data is None  # full tensor never materialized


class TestZarrBackend:
    """Tests for ZarrBackend using nifti-zarr."""

    @pytest.fixture(scope="class")
    def zarr_path(self, tmp_path_factory: pytest.TempPathFactory) -> Path:
        try:
            import niizarr
        except ImportError:
            pytest.skip("nifti-zarr not installed")
        tmp_path = tmp_path_factory.mktemp("zarr")
        data = np.random.rand(16, 16, 16).astype(np.float32)
        nii = nib.Nifti1Image(data, np.eye(4))
        nii_path = tmp_path / "test.nii"
        nib.save(nii, nii_path)
        zarr_path = tmp_path / "test.nii.zarr"
        niizarr.nii2zarr(str(nii_path), str(zarr_path))
        return zarr_path

    def test_zarr_image_shape(self, zarr_path: Path):
        image = ScalarImage(zarr_path)
        assert image.shape == (1, 16, 16, 16)

    def test_zarr_lazy_load(self, zarr_path: Path):
        from torchio.data.backends import ZarrBackend

        image = ScalarImage(zarr_path)
        backend = image.dataobj
        assert isinstance(backend, (NibabelBackend, ZarrBackend))
        assert image._data is None

    def test_zarr_slice(self, zarr_path: Path):
        image = ScalarImage(zarr_path)
        backend = image.dataobj
        sliced = backend[:, 2:12, 2:12, 2:12]
        assert sliced.shape == (1, 10, 10, 10)
        assert image._data is None

    def test_zarr_materialize(self, zarr_path: Path):
        image = ScalarImage(zarr_path)
        tensor = image.data
        assert isinstance(tensor, torch.Tensor)
        assert tensor.shape == (1, 16, 16, 16)

    def test_read_nifti_zarr(self, zarr_path: Path):
        from torchio.data.io import read_nifti_zarr

        tensor, affine = read_nifti_zarr(zarr_path)
        assert tensor.shape == (1, 16, 16, 16)
        assert affine.shape == (4, 4)
