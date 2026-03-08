import tempfile
from pathlib import Path
from unittest.mock import patch

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

    def test_read_shape_2d(self):
        class DummyReader:
            def SetFileName(self, filename):
                self.filename = filename

            def ReadImageInformation(self):
                pass

            def GetNumberOfComponents(self):
                return 1

            def GetDimension(self):
                return 2

            def GetSize(self):
                return 4, 5

        with patch('torchio.data.io.sitk.ImageFileReader', return_value=DummyReader()):
            assert io.read_shape('test.png') == (1, 4, 5, 1)

    def test_read_shape_4d(self):
        class DummyReader:
            def SetFileName(self, filename):
                self.filename = filename

            def ReadImageInformation(self):
                pass

            def GetNumberOfComponents(self):
                return 1

            def GetDimension(self):
                return 4

            def GetSize(self):
                return 4, 5, 6, 7

        with patch('torchio.data.io.sitk.ImageFileReader', return_value=DummyReader()):
            assert io.read_shape('bad.nii.gz') == (7, 4, 5, 6)

    def test_write_image_falls_back_to_nibabel(self):
        tensor = torch.rand(1, 4, 5, 6)
        affine = np.eye(4)
        path = self.dir / 'fallback.nii.gz'

        with (
            patch('torchio.data.io._write_sitk', side_effect=RuntimeError),
            patch('torchio.data.io._write_nibabel') as write_nibabel,
        ):
            io.write_image(tensor, affine, path)

        write_nibabel.assert_called_once_with(tensor, affine, path)

    def test_write_nibabel_img_pair(self):
        tensor = torch.rand(1, 4, 5, 6)
        affine = np.eye(4)
        path = self.dir / 'pair.img'

        io._write_nibabel(tensor, affine, path)

        assert path.is_file()
        assert path.with_suffix('.hdr').is_file()

    def test_write_nibabel_invalid_suffix(self):
        tensor = torch.rand(1, 4, 5, 6)
        affine = np.eye(4)

        with pytest.raises(io.ImageFileError):
            io._write_nibabel(tensor, affine, self.dir / 'pair.xyz')

    def test_write_sitk_with_explicit_squeeze(self):
        tensor = torch.rand(1, 4, 5, 1)
        affine = np.eye(4)
        path = self.dir / 'squeezed.nii.gz'

        with (
            patch('torchio.data.io.nib_to_sitk', return_value=object()) as nib_to_sitk,
            patch('torchio.data.io.sitk.WriteImage'),
        ):
            io._write_sitk(tensor, affine, path, squeeze=True)

        assert nib_to_sitk.call_args.kwargs['force_3d'] is False

    def test_read_matrix_unknown_suffix(self):
        with pytest.raises(ValueError, match='Unknown suffix'):
            io.read_matrix(self.dir / 'matrix.unknown')

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

    def test_ensure_4d_6d_not_supported(self):
        tensor = torch.rand(1, 2, 3, 4, 5, 6)
        with pytest.raises(ValueError, match='6D images not supported yet'):
            io.ensure_4d(tensor)

    def test_sitk_to_nib(self):
        data = np.random.rand(10, 12)
        image = sitk.GetImageFromArray(data)
        tensor, _ = io.sitk_to_nib(image)
        assert data.sum() == pytest.approx(tensor.sum())

    def test_sitk_to_nib_bad_4d_image(self):
        class DummyImage:
            def GetNumberOfComponentsPerPixel(self):
                return 2

            def GetDimension(self):
                return 4

        array = np.arange(2 * 4 * 3 * 5).reshape(2, 4, 3, 5, 1)
        with (
            patch('torchio.data.io.sitk.GetArrayFromImage', return_value=array),
            patch('torchio.data.io.get_ras_affine_from_sitk', return_value=np.eye(4)),
        ):
            tensor, affine = io.sitk_to_nib(DummyImage())

        assert tensor.shape == (2, 5, 3, 4)
        np.testing.assert_array_equal(affine, np.eye(4))

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

    def test_get_ras_affine_from_sitk_2d(self):
        class DummyImage:
            def GetSpacing(self):
                return 1, 2

            def GetDirection(self):
                return -1, 0, 0, -1

            def GetOrigin(self):
                return -5, -6

        affine = io.get_ras_affine_from_sitk(DummyImage())
        fixture = np.diag((1, 2, 1, 1))
        fixture[:3, 3] = 5, 6, 0
        np.testing.assert_array_equal(affine, fixture)

    def test_get_ras_affine_from_sitk_4d_direction(self):
        class DummyImage:
            def GetSpacing(self):
                return 1, 2, 3, 4

            def GetDirection(self):
                return tuple(np.diag((-1, -1, 1, 1)).flatten())

            def GetOrigin(self):
                return -5, -6, 7, 8

        affine = io.get_ras_affine_from_sitk(DummyImage())
        fixture = np.diag((1, 2, 3, 1))
        fixture[:3, 3] = 5, 6, 7
        np.testing.assert_array_equal(affine, fixture)


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
