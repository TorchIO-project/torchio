import tempfile
from pathlib import Path
from unittest.mock import patch as mock_patch

import nibabel as nib
import numpy as np
import pytest
import SimpleITK as sitk
import torch

from torchio.data import ScalarImage
from torchio.data import io

from ..utils import TorchioTestCase


class TestIO(TorchioTestCase):
    """Tests for `io` module."""

    def setUp(self):
        super().setUp()
        self.nii_path = self.get_image_path('read_image')
        self.dicom_dir = self.get_tests_data_dir() / 'dicom'
        self.dicom_path = self.dicom_dir / 'IMG0001.dcm'
        string = (
            '1.5 0.18088 -0.124887 0.65072 '
            '-0.20025 0.965639 -0.165653 -11.6452 '
            '0.0906326 0.18661 0.978245 11.4002 '
            '0 0 0 1 '
        )
        tensor = torch.as_tensor(np.fromstring(string, sep=' ').reshape(4, 4))
        self.matrix = tensor

    def test_read_image(self):
        # I need to find something readable by nib but not sitk
        io.read_image(self.nii_path)

    def test_save_rgb(self):
        im = ScalarImage(tensor=torch.rand(1, 4, 5, 1))
        with pytest.warns(RuntimeWarning):
            im.save(self.dir / 'test.jpg')

    def test_read_dicom_file(self):
        tensor, _ = io.read_image(self.dicom_path)
        assert tuple(tensor.shape) == (1, 88, 128, 1)

    def test_read_dicom_dir(self):
        tensor, _ = io.read_image(self.dicom_dir)
        assert tuple(tensor.shape) == (1, 88, 128, 17)

    def test_dicom_dir_missing(self):
        with pytest.raises(FileNotFoundError):
            io._read_dicom('missing')

    def test_dicom_dir_no_files(self):
        empty = self.dir / 'empty'
        empty.mkdir()
        sitk.ProcessObject_SetGlobalWarningDisplay(False)
        with pytest.raises(FileNotFoundError):
            io._read_dicom(empty)
        sitk.ProcessObject_SetGlobalWarningDisplay(True)

    def write_read_matrix(self, suffix):
        out_path = self.dir / f'matrix{suffix}'
        io.write_matrix(self.matrix, out_path)
        matrix = io.read_matrix(out_path)
        assert torch.allclose(matrix, self.matrix)

    def test_matrix_itk(self):
        self.write_read_matrix('.tfm')
        self.write_read_matrix('.h5')

    def test_matrix_txt(self):
        self.write_read_matrix('.txt')

    def test_ensure_4d_5d(self):
        tensor = torch.rand(3, 4, 5, 1, 2)
        assert io.ensure_4d(tensor).shape == (2, 3, 4, 5)

    def test_ensure_4d_5d_t_gt_1(self):
        tensor = torch.rand(3, 4, 5, 2, 2)
        with pytest.raises(ValueError):
            io.ensure_4d(tensor)

    def test_ensure_4d_2d(self):
        tensor = torch.rand(4, 5)
        assert io.ensure_4d(tensor).shape == (1, 4, 5, 1)

    def test_ensure_4d_2d_3dims_rgb_first(self):
        tensor = torch.rand(3, 4, 5)
        assert io.ensure_4d(tensor).shape == (3, 4, 5, 1)

    def test_ensure_4d_2d_3dims_rgb_last(self):
        tensor = torch.rand(4, 5, 3)
        assert io.ensure_4d(tensor).shape == (3, 4, 5, 1)

    def test_ensure_4d_3d(self):
        tensor = torch.rand(4, 5, 6)
        assert io.ensure_4d(tensor).shape == (1, 4, 5, 6)

    def test_ensure_4d_2_spatial_dims(self):
        tensor = torch.rand(4, 5, 6)
        assert io.ensure_4d(tensor, num_spatial_dims=2).shape == (4, 5, 6, 1)

    def test_ensure_4d_3_spatial_dims(self):
        tensor = torch.rand(4, 5, 6)
        assert io.ensure_4d(tensor, num_spatial_dims=3).shape == (1, 4, 5, 6)

    def test_ensure_4d_nd_not_supported(self):
        tensor = torch.rand(1, 2, 3, 4, 5)
        with pytest.raises(ValueError):
            io.ensure_4d(tensor)

    def test_sitk_to_nib(self):
        data = np.random.rand(10, 12)
        image = sitk.GetImageFromArray(data)
        tensor, _ = io.sitk_to_nib(image)
        assert data.sum() == pytest.approx(tensor.sum())

    def test_sitk_to_affine(self):
        spacing = 1, 2, 3
        direction_lps = -1, 0, 0, 0, -1, 0, 0, 0, 1
        origin_lps = left, posterior, superior = -10, -20, 30
        image = sitk.GetImageFromArray(np.random.rand(10, 20, 30))
        image.SetDirection(direction_lps)
        image.SetSpacing(spacing)
        image.SetOrigin(origin_lps)
        origin_ras = -left, -posterior, superior
        fixture = np.diag((*spacing, 1))
        fixture[:3, 3] = origin_ras
        affine = io.get_ras_affine_from_sitk(image)
        self.assert_tensor_almost_equal(fixture, affine)


# This doesn't work as a method of the class
libs = 'sitk', 'nibabel'
parameters = []
for save_lib in libs:
    for load_lib in libs:
        for dims in 2, 3, 4:
            parameters.append((save_lib, load_lib, dims))


@pytest.mark.parametrize(('save_lib', 'load_lib', 'dims'), parameters)
def test_write_nd_with_a_read_it_with_b(save_lib, load_lib, dims):
    shape = [1, 4, 5, 6]
    if dims == 2:
        shape[-1] = 1
    elif dims == 4:
        shape[0] = 2
    tensor = torch.randn(*shape)
    affine = np.eye(4)
    tempdir = Path(tempfile.gettempdir()) / '.torchio_tests'
    tempdir.mkdir(exist_ok=True)
    path = tempdir / 'test_io.nii'
    save_function = getattr(io, f'_write_{save_lib}')
    load_function = getattr(io, f'_read_{save_lib}')
    save_function(tensor, affine, path)
    loaded_tensor, loaded_affine = load_function(path)
    TorchioTestCase.assert_tensor_equal(
        tensor.squeeze(),
        loaded_tensor.squeeze(),
        msg=f'Save lib: {save_lib}; load lib: {load_lib}; dims: {dims}',
        check_stride=False,
    )
    TorchioTestCase.assert_tensor_equal(affine, loaded_affine)


class TestNibabelToSimpleITK(TorchioTestCase):
    def setUp(self):
        super().setUp()
        self.affine = np.eye(4)

    def test_wrong_num_dims(self):
        with pytest.raises(ValueError):
            io.nib_to_sitk(np.random.rand(10, 10), self.affine)

    def test_2d_single(self):
        data = np.random.rand(1, 10, 12, 1)
        image = io.nib_to_sitk(data, self.affine)
        assert image.GetDimension() == 2
        assert image.GetSize() == (10, 12)
        assert image.GetNumberOfComponentsPerPixel() == 1

    def test_2d_multi(self):
        data = np.random.rand(5, 10, 12, 1)
        image = io.nib_to_sitk(data, self.affine)
        assert image.GetDimension() == 2
        assert image.GetSize() == (10, 12)
        assert image.GetNumberOfComponentsPerPixel() == 5

    def test_2d_3d_single(self):
        data = np.random.rand(1, 10, 12, 1)
        image = io.nib_to_sitk(data, self.affine, force_3d=True)
        assert image.GetDimension() == 3
        assert image.GetSize() == (10, 12, 1)
        assert image.GetNumberOfComponentsPerPixel() == 1

    def test_2d_3d_multi(self):
        data = np.random.rand(5, 10, 12, 1)
        image = io.nib_to_sitk(data, self.affine, force_3d=True)
        assert image.GetDimension() == 3
        assert image.GetSize() == (10, 12, 1)
        assert image.GetNumberOfComponentsPerPixel() == 5

    def test_3d_single(self):
        data = np.random.rand(1, 8, 10, 12)
        image = io.nib_to_sitk(data, self.affine)
        assert image.GetDimension() == 3
        assert image.GetSize() == (8, 10, 12)
        assert image.GetNumberOfComponentsPerPixel() == 1

    def test_3d_multi(self):
        data = np.random.rand(5, 8, 10, 12)
        image = io.nib_to_sitk(data, self.affine)
        assert image.GetDimension() == 3
        assert image.GetSize() == (8, 10, 12)
        assert image.GetNumberOfComponentsPerPixel() == 5


class TestIOCoverage(TorchioTestCase):
    """Additional coverage tests for io module."""

    def test_read_shape_2d(self):
        """read_shape handles 2D images by setting depth to 1."""
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'img2d.png'
            image_2d = sitk.Image(10, 12, sitk.sitkUInt8)
            sitk.WriteImage(image_2d, str(path))
            shape = io.read_shape(path)
            assert shape == (1, 10, 12, 1)

    def test_write_nibabel_hdr_img(self):
        """_write_nibabel writes Nifti1Pair for .img extension."""
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'test.img'
            tensor = torch.rand(1, 5, 5, 5)
            affine = np.eye(4)
            io._write_nibabel(tensor, affine, path)
            assert path.exists()
            hdr_path = path.with_suffix('.hdr')
            assert hdr_path.exists()

    def test_write_nibabel_unsupported_suffix(self):
        """_write_nibabel raises error for unsupported extensions."""
        from nibabel.filebasedimages import ImageFileError

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'test.xyz'
            tensor = torch.rand(1, 5, 5, 5)
            affine = np.eye(4)
            with pytest.raises(ImageFileError):
                io._write_nibabel(tensor, affine, path)

    def test_write_image_nibabel_fallback(self):
        """write_image falls back to nibabel when sitk write fails."""
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'test.nii.gz'
            tensor = torch.rand(1, 5, 5, 5)
            affine = np.eye(4)
            with mock_patch('torchio.data.io._write_sitk', side_effect=RuntimeError):
                io.write_image(tensor, affine, path)
            assert path.exists()

    def test_write_sitk_squeeze(self):
        """_write_sitk respects explicit squeeze parameter."""
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'test.nii.gz'
            tensor = torch.rand(1, 5, 5, 1)
            affine = np.eye(4)
            io._write_sitk(tensor, affine, path, squeeze=False)
            reader = sitk.ImageFileReader()
            reader.SetFileName(str(path))
            reader.ReadImageInformation()
            assert reader.GetDimension() == 3

    def test_read_matrix_unsupported_suffix(self):
        """read_matrix raises ValueError for unsupported file extension."""
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'matrix.mat'
            path.touch()
            with pytest.raises(ValueError, match='Unknown suffix'):
                io.read_matrix(path)

    def test_ensure_4d_1d_raises(self):
        """ensure_4d raises ValueError for 1D tensors."""
        with pytest.raises(ValueError, match='not supported'):
            io.ensure_4d(torch.tensor([1, 2, 3]))

    def test_ensure_4d_6d_raises(self):
        """ensure_4d raises ValueError for 6D tensors."""
        with pytest.raises(ValueError, match='not supported'):
            io.ensure_4d(torch.randn(1, 2, 3, 4, 5, 6))

    def test_sitk_to_nib_4d_bad_nifti(self):
        """sitk_to_nib handles 4D images (bad NIfTI encoding)."""
        with tempfile.TemporaryDirectory() as tmp:
            # Create a 4D NIfTI with channels as 4th spatial dim
            array = np.random.rand(5, 5, 5, 3).astype(np.float32)
            nii = nib.nifti1.Nifti1Image(array, np.eye(4))
            path = Path(tmp) / 'bad4d.nii.gz'
            nib.save(nii, str(path))
            sitk_image = sitk.ReadImage(str(path))
            data, affine = io.sitk_to_nib(sitk_image)
            assert data.shape[0] == 3

    def test_get_ras_affine_4d_direction(self):
        """get_ras_affine_from_sitk handles 16-element direction (4D)."""
        with tempfile.TemporaryDirectory() as tmp:
            # Create a 4D NIfTI to get a 4D SITK image
            array = np.random.rand(5, 5, 5, 2).astype(np.float32)
            nii = nib.nifti1.Nifti1Image(array, np.eye(4))
            path = Path(tmp) / 'dir4d.nii.gz'
            nib.save(nii, str(path))
            sitk_image = sitk.ReadImage(str(path))
            if sitk_image.GetDimension() == 4:
                affine = io.get_ras_affine_from_sitk(sitk_image)
                assert affine.shape == (4, 4)


# Affine with non-orthonormal direction cosines (e.g. some oblique MRI exports).
# SimpleITK/ITK refuses to read NIfTI files with such an affine, so torchio
# must fall back to NiBabel for shape and affine metadata.
NON_ORTHONORMAL_AFFINE = np.array(
    [
        [4.13e-02, 2.07e-02, 7.913e-01, -30.26],
        [-3.098e-01, 2.80e-03, 1.055e-01, 124.6],
        [0.0, -3.118e-01, 5.35e-02, 54.08],
        [0.0, 0.0, 0.0, 1.0],
    ]
)


def _save_non_orthonormal_nifti(path, shape=(8, 8, 4)):
    data = np.zeros(shape, dtype=np.float32)
    nib.save(nib.Nifti1Image(data, NON_ORTHONORMAL_AFFINE), str(path))


class TestNonOrthonormalFallback(TorchioTestCase):
    """`read_shape`/`read_affine` fall back to NiBabel when SimpleITK fails."""

    def test_sitk_cannot_read_fixture(self):
        """The fixture must trigger a SimpleITK RuntimeError."""
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'oblique.nii.gz'
            _save_non_orthonormal_nifti(path)
            reader = sitk.ImageFileReader()
            reader.SetFileName(str(path))
            with pytest.raises(RuntimeError):
                reader.ReadImageInformation()

    def test_read_shape_nibabel_fallback(self):
        """read_shape returns (C, W, H, D) using NiBabel when SimpleITK fails."""
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'oblique.nii.gz'
            _save_non_orthonormal_nifti(path, shape=(8, 9, 4))
            with pytest.warns(UserWarning):
                shape = io.read_shape(path)
            assert shape == (1, 8, 9, 4)

    def test_read_affine_nibabel_fallback(self):
        """read_affine returns the NiBabel affine when SimpleITK fails."""
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'oblique.nii.gz'
            _save_non_orthonormal_nifti(path)
            with pytest.warns(UserWarning):
                affine = io.read_affine(path)
            self.assert_tensor_almost_equal(affine, NON_ORTHONORMAL_AFFINE)

    def test_read_shape_nibabel_4d(self):
        """read_shape maps a 4D NiBabel image to (C, W, H, D)."""
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'oblique4d.nii.gz'
            data = np.zeros((8, 9, 4, 3), dtype=np.float32)
            nib.save(nib.Nifti1Image(data, NON_ORTHONORMAL_AFFINE), str(path))
            with pytest.warns(UserWarning):
                shape = io.read_shape(path)
            assert shape == (3, 8, 9, 4)

    def test_read_shape_matches_loaded_tensor_4d(self):
        """read_shape and the NiBabel data fallback agree for 4D images."""
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'oblique4d.nii.gz'
            data = np.zeros((8, 9, 4, 3), dtype=np.float32)
            nib.save(nib.Nifti1Image(data, NON_ORTHONORMAL_AFFINE), str(path))
            with pytest.warns(UserWarning):
                shape = io.read_shape(path)
            tensor, _ = io._read_nibabel(path)
            assert shape == tuple(tensor.shape)
            assert shape == (3, 8, 9, 4)

    def test_scalar_image_metadata(self):
        """ScalarImage exposes shape/affine/spacing/origin without loading."""
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'oblique.nii.gz'
            _save_non_orthonormal_nifti(path, shape=(8, 9, 4))
            image = ScalarImage(path)
            with pytest.warns(UserWarning):
                assert image.shape == (1, 8, 9, 4)
            with pytest.warns(UserWarning):
                self.assert_tensor_almost_equal(
                    image.affine,
                    NON_ORTHONORMAL_AFFINE,
                )
            with pytest.warns(UserWarning):
                spacing = image.spacing
            expected = np.sqrt((NON_ORTHONORMAL_AFFINE[:3, :3] ** 2).sum(axis=0))
            self.assert_tensor_almost_equal(np.array(spacing), expected)

    def test_normal_file_does_not_use_fallback(self):
        """Orthonormal files keep using SimpleITK (no warning, same result)."""
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'normal.nii.gz'
            data = np.zeros((8, 9, 4), dtype=np.float32)
            affine = np.diag([1.5, 2.0, 3.0, 1.0])
            affine[:3, 3] = [10.0, 20.0, 30.0]
            nib.save(nib.Nifti1Image(data, affine), str(path))
            import warnings

            with warnings.catch_warnings():
                warnings.simplefilter('error')
                assert io.read_shape(path) == (1, 8, 9, 4)
                io.read_affine(path)
