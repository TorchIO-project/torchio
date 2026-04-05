"""Tests for Image, ScalarImage, and LabelMap."""

from __future__ import annotations

from pathlib import Path

import nibabel as nib
import numpy as np
import pytest
import SimpleITK as sitk
import torch
from einops import rearrange

from torchio import Image
from torchio import LabelMap
from torchio import ScalarImage


class TestImageCreationFromPath:
    def test_from_path_positional(self, tmp_path: Path):
        tensor = torch.randn(1, 10, 10, 10)
        array = rearrange(tensor.numpy(), "c i j k -> i j k c")
        nii = nib.Nifti1Image(array, np.eye(4))
        path = tmp_path / "test.nii.gz"
        nib.save(nii, path)
        image = ScalarImage(path)
        assert not image.is_loaded
        _ = image.data
        assert image.is_loaded

    def test_from_path_keyword(self, tmp_path: Path):
        tensor = torch.randn(1, 10, 10, 10)
        nii = nib.Nifti1Image(tensor.numpy()[0], np.eye(4))
        path = tmp_path / "test.nii.gz"
        nib.save(nii, path)
        image = ScalarImage(path=path)
        assert image.path == path

    def test_from_path_string(self, tmp_path: Path):
        tensor = torch.randn(1, 10, 10, 10)
        nii = nib.Nifti1Image(tensor.numpy()[0], np.eye(4))
        path = tmp_path / "test.nii.gz"
        nib.save(nii, path)
        image = ScalarImage(str(path))
        assert image.path == path

    def test_path_with_affine(self, tmp_path: Path):
        tensor = torch.randn(1, 10, 10, 10)
        nii = nib.Nifti1Image(tensor.numpy()[0], np.eye(4))
        path = tmp_path / "test.nii.gz"
        nib.save(nii, path)
        custom_affine = np.diag([2.0, 2.0, 2.0, 1.0])
        image = ScalarImage(path, affine=custom_affine)
        assert not image.is_loaded
        np.testing.assert_array_equal(image.affine, custom_affine)

    def test_path_property(self, tmp_path: Path):
        tensor = torch.randn(1, 10, 10, 10)
        nii = nib.Nifti1Image(tensor.numpy()[0], np.eye(4))
        path = tmp_path / "test.nii.gz"
        nib.save(nii, path)
        image = ScalarImage(path)
        assert image.path == path

    def test_no_path_raises(self):
        with pytest.raises(TypeError):
            ScalarImage()


class TestImageCreationFromTensor:
    def test_from_tensor(self):
        tensor = torch.randn(1, 10, 10, 10)
        image = ScalarImage.from_tensor(tensor)
        assert isinstance(image, ScalarImage)
        assert torch.equal(image.data, tensor)

    def test_from_tensor_numpy(self):
        array = np.random.randn(1, 10, 10, 10).astype(np.float32)
        image = ScalarImage.from_tensor(array)
        assert torch.equal(image.data, torch.from_numpy(array))

    def test_from_tensor_default_affine(self):
        image = ScalarImage.from_tensor(torch.randn(1, 10, 10, 10))
        np.testing.assert_array_equal(image.affine, np.eye(4))

    def test_from_tensor_custom_affine(self):
        affine = np.diag([2.0, 2.0, 2.0, 1.0])
        image = ScalarImage.from_tensor(
            torch.randn(1, 10, 10, 10),
            affine=affine,
        )
        np.testing.assert_array_equal(image.affine, affine)

    def test_from_tensor_affine_object(self):
        from torchio import Affine

        aff = Affine(np.diag([2.0, 2.0, 2.0, 1.0]))
        image = ScalarImage.from_tensor(
            torch.randn(1, 10, 10, 10),
            affine=aff,
        )
        assert image.affine == aff

    def test_from_tensor_metadata(self):
        image = ScalarImage.from_tensor(
            torch.randn(1, 10, 10, 10),
            scan_id="abc123",
        )
        assert image.metadata == {"scan_id": "abc123"}

    def test_from_tensor_must_be_4d(self):
        with pytest.raises(ValueError, match="4D"):
            ScalarImage.from_tensor(torch.randn(10, 10, 10))

    def test_from_tensor_affine_must_be_4x4(self):
        with pytest.raises(ValueError, match=r"4.*4"):
            ScalarImage.from_tensor(
                torch.randn(1, 10, 10, 10),
                affine=np.eye(3),
            )

    def test_from_tensor_path_is_none(self):
        image = ScalarImage.from_tensor(torch.randn(1, 10, 10, 10))
        assert image.path is None

    def test_from_tensor_preserves_subclass(self):
        image = LabelMap.from_tensor(torch.randint(0, 5, (1, 10, 10, 10)))
        assert isinstance(image, LabelMap)

    def test_from_tensor_is_loaded(self):
        image = ScalarImage.from_tensor(torch.randn(1, 10, 10, 10))
        assert image.is_loaded


class TestImageProperties:
    @pytest.fixture
    def image(self) -> ScalarImage:
        return ScalarImage.from_tensor(torch.randn(2, 10, 20, 30))

    def test_shape(self, image: ScalarImage):
        assert image.shape == (2, 10, 20, 30)

    def test_spatial_shape(self, image: ScalarImage):
        assert image.spatial_shape == (10, 20, 30)

    def test_num_channels(self, image: ScalarImage):
        assert image.num_channels == 2

    def test_spacing(self, image: ScalarImage):
        assert image.spacing == (1.0, 1.0, 1.0)

    def test_spacing_with_custom_affine(self):
        affine = np.diag([0.5, 0.8, 1.2, 1.0])
        image = ScalarImage.from_tensor(
            torch.randn(1, 10, 10, 10),
            affine=affine,
        )
        np.testing.assert_allclose(image.spacing, (0.5, 0.8, 1.2))

    def test_origin(self, image: ScalarImage):
        assert image.origin == (0.0, 0.0, 0.0)

    def test_is_loaded_from_tensor(self, image: ScalarImage):
        assert image.is_loaded

    def test_memory(self, image: ScalarImage):
        # 2 * 10 * 20 * 30 * 4 bytes (float32)
        assert image.memory == 2 * 10 * 20 * 30 * 4


class TestLabelMap:
    def test_is_label_map(self):
        label = LabelMap.from_tensor(torch.randint(0, 5, (1, 10, 10, 10)))
        assert isinstance(label, LabelMap)
        assert isinstance(label, Image)

    def test_is_not_scalar_image(self):
        label = LabelMap.from_tensor(torch.randint(0, 5, (1, 10, 10, 10)))
        assert not isinstance(label, ScalarImage)

    def test_is_label_subclass(self):
        assert issubclass(LabelMap, Image)


class TestScalarImage:
    def test_is_scalar_image(self):
        image = ScalarImage.from_tensor(torch.randn(1, 10, 10, 10))
        assert isinstance(image, ScalarImage)
        assert isinstance(image, Image)

    def test_is_not_label_map(self):
        image = ScalarImage.from_tensor(torch.randn(1, 10, 10, 10))
        assert not isinstance(image, LabelMap)

    def test_is_image_subclass(self):
        assert issubclass(ScalarImage, Image)


class TestNewLike:
    def test_new_like_preserves_type(self):
        image = ScalarImage.from_tensor(torch.randn(1, 10, 10, 10))
        new = image.new_like(data=torch.randn(1, 5, 5, 5))
        assert isinstance(new, ScalarImage)

    def test_new_like_preserves_affine(self):
        affine = np.diag([2.0, 2.0, 2.0, 1.0])
        image = ScalarImage.from_tensor(
            torch.randn(1, 10, 10, 10),
            affine=affine,
        )
        new = image.new_like(data=torch.randn(1, 5, 5, 5))
        np.testing.assert_array_equal(new.affine, affine)

    def test_new_like_with_new_affine(self):
        image = ScalarImage.from_tensor(torch.randn(1, 10, 10, 10))
        new_affine = np.diag([3.0, 3.0, 3.0, 1.0])
        new = image.new_like(data=torch.randn(1, 5, 5, 5), affine=new_affine)
        np.testing.assert_array_equal(new.affine, new_affine)

    def test_new_like_preserves_metadata(self):
        image = ScalarImage.from_tensor(
            torch.randn(1, 10, 10, 10),
            scan_id="abc123",
        )
        new = image.new_like(data=torch.randn(1, 5, 5, 5))
        assert new.metadata == {"scan_id": "abc123"}

    def test_new_like_on_custom_subclass(self):
        class MyImage(ScalarImage):
            pass

        image = MyImage.from_tensor(torch.randn(1, 10, 10, 10))
        new = image.new_like(data=torch.randn(1, 5, 5, 5))
        assert isinstance(new, MyImage)

    def test_new_like_label_map(self):
        label = LabelMap.from_tensor(torch.randint(0, 5, (1, 10, 10, 10)))
        new = label.new_like(data=torch.randint(0, 5, (1, 5, 5, 5)))
        assert isinstance(new, LabelMap)
        assert not isinstance(new, ScalarImage)


class TestSetData:
    def test_set_data(self):
        image = ScalarImage.from_tensor(torch.randn(1, 10, 10, 10))
        new_data = torch.randn(1, 5, 5, 5)
        image.set_data(new_data)
        assert torch.equal(image.data, new_data)

    def test_set_data_must_be_4d(self):
        image = ScalarImage.from_tensor(torch.randn(1, 10, 10, 10))
        with pytest.raises(ValueError, match="4D"):
            image.set_data(torch.randn(10, 10, 10))


class TestImageRepr:
    def test_loaded_repr(self):
        image = ScalarImage.from_tensor(torch.randn(1, 10, 10, 10))
        r = repr(image)
        assert "ScalarImage" in r
        assert "10" in r

    def test_unloaded_repr(self, tmp_path: Path):
        tensor = torch.randn(1, 10, 10, 10)
        array = rearrange(tensor.numpy(), "c i j k -> i j k c")
        nii = nib.Nifti1Image(array, np.eye(4))
        path = tmp_path / "test.nii.gz"
        nib.save(nii, path)
        image = ScalarImage(path)
        r = repr(image)
        assert "ScalarImage" in r
        # Now shows shape/dtype/memory even without loading
        assert "shape" in r or "spatial" in r
        assert "dtype" in r
        assert "memory" in r

    def test_repr_tensor_only_no_path(self):
        image = Image.from_tensor(torch.randn(1, 4, 4, 4))
        image._data = None
        r = repr(image)
        # No data and no path — falls back to minimal repr
        assert "Image" in r


class TestImageLoad:
    def test_load_already_loaded_is_noop(self):
        image = ScalarImage.from_tensor(torch.randn(1, 10, 10, 10))
        original_data = image.data
        image.load()
        assert image.data is original_data

    def test_load_no_path_raises(self):
        image = ScalarImage.from_tensor(torch.randn(1, 10, 10, 10))
        image._data = None
        image._path = None
        with pytest.raises(RuntimeError, match="no path"):
            image.load()

    def test_shape_no_data_no_path_raises(self):
        image = ScalarImage.from_tensor(torch.randn(1, 10, 10, 10))
        image._data = None
        image._path = None
        image._backend = None
        with pytest.raises(RuntimeError, match="Cannot determine shape"):
            image.shape


class TestImageCopy:
    def test_copy(self):
        import copy

        image = ScalarImage.from_tensor(torch.randn(1, 10, 10, 10))
        copied = copy.copy(image)
        assert isinstance(copied, ScalarImage)
        assert torch.equal(copied.data, image.data)
        copied.set_data(torch.zeros(1, 10, 10, 10))
        assert not torch.equal(image.data, copied.data)

    def test_deepcopy_tensor_based(self):
        import copy

        image = ScalarImage.from_tensor(torch.randn(1, 10, 10, 10))
        copied = copy.deepcopy(image)
        assert isinstance(copied, ScalarImage)
        assert torch.equal(copied.data, image.data)
        copied.set_data(torch.zeros(1, 10, 10, 10))
        assert not torch.equal(image.data, copied.data)

    def test_deepcopy_path_based_unloaded(self, tmp_path: Path):
        import copy

        tensor = torch.randn(1, 10, 10, 10)
        array = rearrange(tensor.numpy(), "c i j k -> i j k c")
        nii = nib.Nifti1Image(array, np.eye(4))
        path = tmp_path / "test.nii.gz"
        nib.save(nii, path)

        image = ScalarImage(path)
        copied = copy.deepcopy(image)
        assert isinstance(copied, ScalarImage)
        assert not copied.is_loaded
        assert copied.path == path

    def test_deepcopy_path_based_loaded(self, tmp_path: Path):
        import copy

        tensor = torch.randn(1, 10, 10, 10)
        array = rearrange(tensor.numpy(), "c i j k -> i j k c")
        nii = nib.Nifti1Image(array, np.eye(4))
        path = tmp_path / "test.nii.gz"
        nib.save(nii, path)

        image = ScalarImage(path)
        _ = image.data  # trigger load
        copied = copy.deepcopy(image)
        assert isinstance(copied, ScalarImage)
        assert copied.is_loaded
        assert copied.path == path

    def test_deepcopy_degenerate_state(self):
        import copy

        image = ScalarImage.from_tensor(torch.randn(1, 4, 4, 4))
        image._data = None
        image._path = None
        copied = copy.deepcopy(image)
        assert isinstance(copied, ScalarImage)
        assert not copied.is_loaded
        assert copied.path is None


class TestNibabelReader:
    def test_4d_nifti(self, tmp_path: Path):
        data = np.random.randn(10, 10, 10, 3).astype(np.float32)
        nii = nib.Nifti1Image(data, np.eye(4))
        path = tmp_path / "multichannel.nii.gz"
        nib.save(nii, path)
        image = ScalarImage(path)
        assert image.shape == (3, 10, 10, 10)

    def test_4d_nifti_shape_from_header(self, tmp_path: Path):
        data = np.random.randn(10, 10, 10, 3).astype(np.float32)
        nii = nib.Nifti1Image(data, np.eye(4))
        path = tmp_path / "multichannel.nii.gz"
        nib.save(nii, path)
        image = ScalarImage(path)
        assert not image.is_loaded
        shape = image.shape
        assert shape == (3, 10, 10, 10)
        assert not image.is_loaded

    def test_invalid_ndim_raises(self, tmp_path: Path):
        data = np.random.randn(10, 10, 10, 3, 2).astype(np.float32)
        nii = nib.Nifti1Image(data, np.eye(4))
        path = tmp_path / "bad.nii.gz"
        nib.save(nii, path)
        image = ScalarImage(path)
        with pytest.raises(ValueError, match="3D or 4D"):
            image.data

    def test_invalid_ndim_shape_from_header(self, tmp_path: Path):
        data = np.random.randn(10, 10, 10, 3, 2).astype(np.float32)
        nii = nib.Nifti1Image(data, np.eye(4))
        path = tmp_path / "bad.nii.gz"
        nib.save(nii, path)
        image = ScalarImage(path)
        with pytest.raises(ValueError, match="3D or 4D"):
            image.shape


class TestSimpleITKReader:
    def test_read_nrrd(self, tmp_path: Path):
        data = np.random.randn(10, 12, 14).astype(np.float32)
        sitk_image = sitk.GetImageFromArray(
            rearrange(data, "i j k -> k j i"),
        )
        sitk_image.SetSpacing((0.5, 0.8, 1.2))
        path = tmp_path / "test.nrrd"
        sitk.WriteImage(sitk_image, str(path))

        image = ScalarImage(path)
        assert image.shape == (1, 10, 12, 14)
        np.testing.assert_allclose(image.spacing, (0.5, 0.8, 1.2))

    def test_shape_from_header_nrrd(self, tmp_path: Path):
        data = np.random.randn(10, 12, 14).astype(np.float32)
        sitk_image = sitk.GetImageFromArray(
            rearrange(data, "i j k -> k j i"),
        )
        path = tmp_path / "test.nrrd"
        sitk.WriteImage(sitk_image, str(path))

        image = ScalarImage(path)
        assert not image.is_loaded
        shape = image.shape
        assert shape == (1, 10, 12, 14)
        assert not image.is_loaded

    def test_read_multichannel_nrrd(self, tmp_path: Path):
        data = np.random.randn(10, 12, 14, 3).astype(np.float32)
        sitk_image = sitk.GetImageFromArray(
            rearrange(data, "i j k c -> k j i c"),
            isVector=True,
        )
        path = tmp_path / "multi.nrrd"
        sitk.WriteImage(sitk_image, str(path))

        image = ScalarImage(path)
        assert image.shape == (3, 10, 12, 14)


class TestReaderErrors:
    def test_sitk_reader_invalid_ndim(self, tmp_path: Path):
        """SimpleITK reader raises for unexpected dimensions."""
        from torchio.data.io import read_sitk

        data_2d = np.zeros((10, 10), dtype=np.float32)
        sitk_image = sitk.GetImageFromArray(data_2d)
        path = tmp_path / "flat.nrrd"
        sitk.WriteImage(sitk_image, str(path))

        with pytest.raises(ValueError, match="Expected 3D"):
            read_sitk(path)

    def test_sitk_shape_reader_invalid_ndim(self, tmp_path: Path):
        """SimpleITK shape reader raises for non-3D images."""
        data_2d = np.zeros((10, 10), dtype=np.float32)
        sitk_image = sitk.GetImageFromArray(data_2d)
        path = tmp_path / "flat.nrrd"
        sitk.WriteImage(sitk_image, str(path))

        image = ScalarImage(path)
        with pytest.raises(ValueError, match="Expected 3D"):
            image.shape


class TestSimpleITKReaderEdgeCases:
    def test_multichannel_nrrd_loads_data(self, tmp_path: Path):
        data = np.random.randn(10, 12, 14, 3).astype(np.float32)
        sitk_image = sitk.GetImageFromArray(
            rearrange(data, "i j k c -> k j i c"),
            isVector=True,
        )
        path = tmp_path / "multi.nrrd"
        sitk.WriteImage(sitk_image, str(path))

        image = ScalarImage(path)
        _ = image.data  # trigger actual data load
        assert image.shape == (3, 10, 12, 14)

    def test_5d_vector_nifti_loads_data(self, tmp_path: Path):
        # SimpleITK writes multichannel NIfTI as 5D vector: (I, J, K, 1, C)
        tensor = torch.randn(3, 10, 10, 10)
        image = ScalarImage.from_tensor(tensor)
        path = tmp_path / "vector.nii.gz"
        image.save(path)

        loaded = ScalarImage(path)
        _ = loaded.data  # trigger nibabel load of 5D vector NIfTI
        assert loaded.shape == (3, 10, 10, 10)


class TestImageIO:
    def test_save_and_load_nifti(self, tmp_path: Path):
        tensor = torch.randn(1, 10, 10, 10)
        affine = np.diag([2.0, 2.0, 2.0, 1.0])
        image = ScalarImage.from_tensor(tensor, affine=affine)
        path = tmp_path / "output.nii.gz"
        image.save(path)

        loaded = ScalarImage(path)
        assert loaded.shape == (1, 10, 10, 10)
        np.testing.assert_allclose(loaded.spacing, (2.0, 2.0, 2.0))

    def test_save_and_load_nrrd(self, tmp_path: Path):
        tensor = torch.randn(1, 10, 12, 14)
        affine = np.diag([0.5, 0.8, 1.2, 1.0])
        image = ScalarImage.from_tensor(tensor, affine=affine)
        path = tmp_path / "output.nrrd"
        image.save(path)

        loaded = ScalarImage(path)
        assert loaded.shape == (1, 10, 12, 14)
        np.testing.assert_allclose(loaded.spacing, (0.5, 0.8, 1.2))

    @pytest.mark.parametrize("extension", [".nii.gz", ".nrrd"])
    def test_save_preserves_affine(self, tmp_path: Path, extension: str):
        """Round-trip save/load must preserve the full affine matrix."""
        tensor = torch.randn(1, 8, 10, 12)
        affine = np.array(
            [
                [0.0, 0.0, 1.5, -10.0],
                [0.5, 0.0, 0.0, -20.0],
                [0.0, 0.8, 0.0, 30.0],
                [0.0, 0.0, 0.0, 1.0],
            ]
        )
        image = ScalarImage.from_tensor(tensor, affine=affine)
        path = tmp_path / f"output{extension}"
        image.save(path)

        loaded = ScalarImage(path)
        np.testing.assert_allclose(
            loaded.affine.numpy(),
            affine,
            atol=1e-6,
        )
        torch.testing.assert_close(loaded.data, tensor, atol=1e-4, rtol=1e-4)

    def test_save_preserves_lps_orientation(self, tmp_path: Path):
        """NIfTI with LPS+ orientation must survive a save/load round-trip."""
        tensor = torch.randn(1, 8, 10, 12)
        affine = np.array(
            [
                [-0.5, 0.0, 0.0, 90.0],
                [0.0, -0.5, 0.0, 126.0],
                [0.0, 0.0, 0.5, -72.0],
                [0.0, 0.0, 0.0, 1.0],
            ]
        )
        image = ScalarImage.from_tensor(tensor, affine=affine)
        path = tmp_path / "lps.nii.gz"
        image.save(path)

        loaded = ScalarImage(path)
        np.testing.assert_allclose(
            loaded.affine.numpy(),
            affine,
            atol=1e-6,
        )
        assert loaded.affine.orientation == ("L", "P", "S")

    def test_save_multichannel(self, tmp_path: Path):
        tensor = torch.randn(3, 10, 10, 10)
        image = ScalarImage.from_tensor(tensor)
        path = tmp_path / "multi.nii.gz"
        image.save(path)

        loaded = ScalarImage(path)
        assert loaded.shape == (3, 10, 10, 10)

    def test_custom_reader(self, tmp_path: Path):
        path = tmp_path / "test.npy"
        data = np.random.randn(1, 10, 10, 10).astype(np.float32)
        np.save(path, data)

        def npy_reader(p):
            arr = np.load(p)
            return torch.from_numpy(arr), np.eye(4)

        image = ScalarImage(path, reader=npy_reader)
        assert image.shape == (1, 10, 10, 10)

    def test_save_nii_zarr(self, tmp_path: Path):
        tensor = torch.randn(1, 10, 12, 14)
        affine = np.diag([2.0, 3.0, 4.0, 1.0])
        image = ScalarImage.from_tensor(tensor, affine=affine)
        path = tmp_path / "output.nii.zarr"
        image.save(path)

        loaded = ScalarImage(path)
        assert loaded.shape == (1, 10, 12, 14)
        np.testing.assert_allclose(loaded.spacing, (2.0, 3.0, 4.0), atol=1e-5)
        np.testing.assert_allclose(
            loaded.data.numpy(),
            tensor.numpy(),
            atol=1e-5,
        )

    def test_save_nii_zarr_multichannel(self, tmp_path: Path):
        tensor = torch.randn(3, 8, 8, 8)
        image = ScalarImage.from_tensor(tensor)
        path = tmp_path / "multi.nii.zarr"
        image.save(path)

        loaded = ScalarImage(path)
        assert loaded.shape == (3, 8, 8, 8)


class TestImageSlicing:
    def test_slice_channel_int(self):
        tensor = torch.randn(3, 20, 20, 20)
        image = ScalarImage.from_tensor(tensor)
        sliced = image[0]
        assert sliced.shape == (1, 20, 20, 20)
        torch.testing.assert_close(sliced.data, tensor[0:1])

    def test_slice_channel_range(self):
        tensor = torch.randn(5, 20, 20, 20)
        image = ScalarImage.from_tensor(tensor)
        sliced = image[1:3]
        assert sliced.shape == (2, 20, 20, 20)
        torch.testing.assert_close(sliced.data, tensor[1:3])

    def test_slice_spatial_via_tuple(self):
        tensor = torch.randn(1, 20, 20, 20)
        image = ScalarImage.from_tensor(tensor)
        sliced = image[:, 5:10]
        assert sliced.shape == (1, 5, 20, 20)
        torch.testing.assert_close(sliced.data, tensor[:, 5:10, :, :])

    def test_slice_all_four_dims(self):
        tensor = torch.randn(3, 20, 20, 20)
        image = ScalarImage.from_tensor(tensor)
        sliced = image[0:2, 2:8, 3:7, 4:10]
        assert sliced.shape == (2, 6, 4, 6)
        torch.testing.assert_close(sliced.data, tensor[0:2, 2:8, 3:7, 4:10])

    def test_slice_preserves_class(self):
        tensor = torch.randn(1, 20, 20, 20)
        image = LabelMap.from_tensor(tensor)
        sliced = image[:, 5:10]
        assert isinstance(sliced, LabelMap)

    def test_slice_updates_affine_origin(self):
        affine = np.diag([2.0, 3.0, 4.0, 1.0])
        tensor = torch.randn(1, 20, 20, 20)
        image = ScalarImage.from_tensor(tensor, affine=affine)
        sliced = image[:, 5:10]
        # New origin should be shifted by 5 voxels * 2mm spacing in I direction
        expected_origin = (10.0, 0.0, 0.0)
        np.testing.assert_allclose(sliced.origin, expected_origin)

    def test_slice_channel_does_not_affect_origin(self):
        affine = np.diag([2.0, 3.0, 4.0, 1.0])
        tensor = torch.randn(5, 20, 20, 20)
        image = ScalarImage.from_tensor(tensor, affine=affine)
        sliced = image[1:3]
        np.testing.assert_allclose(sliced.origin, (0.0, 0.0, 0.0))

    def test_slice_partial_dims(self):
        tensor = torch.randn(1, 20, 20, 20)
        image = ScalarImage.from_tensor(tensor)
        sliced = image[:, 5:10, 3:7]
        assert sliced.shape == (1, 5, 4, 20)

    def test_slice_negative_indices(self):
        tensor = torch.randn(1, 20, 20, 20)
        image = ScalarImage.from_tensor(tensor)
        sliced = image[:, -5:]
        assert sliced.shape == (1, 5, 20, 20)
        torch.testing.assert_close(sliced.data, tensor[:, 15:, :, :])

    def test_slice_with_step(self):
        tensor = torch.randn(1, 20, 20, 20)
        image = ScalarImage.from_tensor(tensor)
        sliced = image[:, ::2]
        assert sliced.shape == (1, 10, 20, 20)
        torch.testing.assert_close(sliced.data, tensor[:, ::2, :, :])

    def test_slice_ellipsis_trailing(self):
        tensor = torch.randn(3, 20, 20, 20)
        image = ScalarImage.from_tensor(tensor)
        sliced = image[..., 5:10]
        assert sliced.shape == (3, 20, 20, 5)
        torch.testing.assert_close(sliced.data, tensor[..., 5:10])

    def test_slice_ellipsis_leading(self):
        tensor = torch.randn(3, 20, 20, 20)
        image = ScalarImage.from_tensor(tensor)
        sliced = image[0, ...]
        assert sliced.shape == (1, 20, 20, 20)
        torch.testing.assert_close(sliced.data, tensor[0:1])

    def test_slice_ellipsis_middle(self):
        tensor = torch.randn(3, 20, 20, 20)
        image = ScalarImage.from_tensor(tensor)
        sliced = image[0:2, ..., 5:10]
        assert sliced.shape == (2, 20, 20, 5)
        torch.testing.assert_close(sliced.data, tensor[0:2, :, :, 5:10])

    def test_slice_bare_ellipsis(self):
        tensor = torch.randn(1, 20, 20, 20)
        image = ScalarImage.from_tensor(tensor)
        sliced = image[...]
        assert sliced.shape == (1, 20, 20, 20)
        torch.testing.assert_close(sliced.data, tensor)

    def test_slice_double_ellipsis_raises(self):
        image = ScalarImage.from_tensor(torch.randn(1, 20, 20, 20))
        with pytest.raises(IndexError, match="one ellipsis"):
            image[..., ...]

    def test_slice_float_raises(self):
        image = ScalarImage.from_tensor(torch.randn(1, 20, 20, 20))
        with pytest.raises(TypeError, match="not understood"):
            image[1.5]

    def test_slice_too_many_dims_raises(self):
        image = ScalarImage.from_tensor(torch.randn(1, 20, 20, 20))
        with pytest.raises(IndexError, match="Too many"):
            image[:, :, :, :, :]

    def test_slice_lazy_does_not_load(self, tmp_path: Path):
        data = np.random.randn(10, 12, 14).astype(np.float32)
        nii = nib.Nifti1Image(data, np.eye(4))
        path = tmp_path / "test.nii.gz"
        nib.save(nii, path)

        image = ScalarImage(path)
        sliced = image[:, 2:5, 3:7, 4:8]
        assert not image.is_loaded  # parent not loaded
        assert sliced.shape == (1, 3, 4, 4)

    def test_slice_preserves_metadata(self):
        image = ScalarImage.from_tensor(
            torch.randn(1, 20, 20, 20),
            modality="T1",
        )
        sliced = image[:, 5:10]
        assert sliced.metadata["modality"] == "T1"


class TestReaderWriterKwargs:
    def _make_nifti(self, tmp_path: Path) -> Path:
        """Helper: write a small NIfTI file and return its path."""
        data = np.random.randn(10, 12, 14).astype(np.float32)
        nii = nib.Nifti1Image(data, np.eye(4))
        path = tmp_path / "test.nii.gz"
        nib.save(nii, path)
        return path

    def test_reader_kwargs_passed(self, tmp_path: Path):
        """reader_kwargs are forwarded to the reader function."""
        from unittest.mock import patch

        path = self._make_nifti(tmp_path)
        image = ScalarImage(path, reader_kwargs={"keep_file_open": True})
        with patch("torchio.data.image.nib.load", wraps=nib.load) as mock_load:
            image.load()
            mock_load.assert_called_once_with(path, keep_file_open=True)

    def test_writer_kwargs_passed(self, tmp_path: Path):
        """writer_kwargs are forwarded to SimpleITK.WriteImage."""
        from unittest.mock import patch

        image = ScalarImage.from_tensor(torch.randn(1, 4, 4, 4))
        out = tmp_path / "out.nii.gz"
        with patch("torchio.data.image.sitk.WriteImage") as mock_write:
            image.save(out, writer_kwargs={"useCompression": True})
            mock_write.assert_called_once()
            _, call_kwargs = mock_write.call_args
            assert call_kwargs["useCompression"] is True

    def test_reader_kwargs_deepcopy(self, tmp_path: Path):
        """reader_kwargs survive deepcopy."""
        import copy

        path = self._make_nifti(tmp_path)
        kw = {"keep_file_open": True}
        image = ScalarImage(path, reader_kwargs=kw)
        copied = copy.deepcopy(image)
        assert copied._reader_kwargs == kw
        assert copied._reader_kwargs is not image._reader_kwargs
