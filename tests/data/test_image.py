#!/usr/bin/env python
"""Tests for Image."""

import copy
import sys
import tempfile
from pathlib import Path
from unittest.mock import patch

import nibabel as nib
import numpy as np
import pytest
import torch

import torchio as tio

from ..utils import TorchioTestCase


class TestImage(TorchioTestCase):
    """Tests for `Image`."""

    def test_image_not_found(self):
        with pytest.raises(FileNotFoundError):
            tio.ScalarImage('nopath')

    @pytest.mark.skipif(sys.platform == 'win32', reason='Path not valid')
    def test_wrong_path_value(self):
        with pytest.raises(RuntimeError):
            tio.ScalarImage('~&./@#"!?X7=+')

    def test_wrong_path_type(self):
        with pytest.raises(TypeError):
            tio.ScalarImage(5)

    def test_wrong_affine(self):
        with pytest.raises(TypeError):
            tio.ScalarImage(5, affine=1)

    def test_tensor_flip(self):
        sample_input = torch.ones((4, 30, 30, 30))
        tio.RandomFlip()(sample_input)

    def test_tensor_affine(self):
        sample_input = torch.ones((4, 10, 10, 10))
        tio.RandomAffine()(sample_input)

    def test_wrong_scalar_image_type(self):
        data = torch.ones((1, 10, 10, 10))
        with pytest.raises(ValueError):
            tio.ScalarImage(tensor=data, type=tio.LABEL)

    def test_wrong_label_map_type(self):
        data = torch.ones((1, 10, 10, 10))
        with pytest.raises(ValueError):
            tio.LabelMap(tensor=data, type=tio.INTENSITY)

    def test_no_input(self):
        with pytest.raises(ValueError):
            tio.ScalarImage()

    def test_bad_key(self):
        with pytest.raises(ValueError):
            tio.ScalarImage(path='', data=5)

    def test_repr(self):
        subject = tio.Subject(
            t1=tio.ScalarImage(self.get_image_path('repr_test')),
        )
        assert 'memory' not in repr(subject['t1'])
        subject.load()
        assert 'memory' in repr(subject['t1'])

    def test_data_tensor(self):
        subject = copy.deepcopy(self.sample_subject)
        subject.load()
        assert subject.t1.data is subject.t1.tensor

    def test_bad_affine(self):
        with pytest.raises(ValueError):
            tio.ScalarImage(tensor=torch.rand(1, 2, 3, 4), affine=np.eye(3))

    def test_nans_tensor(self):
        tensor = np.random.rand(1, 2, 3, 4)
        tensor[0, 0, 0, 0] = np.nan
        with pytest.warns(RuntimeWarning):
            image = tio.ScalarImage(tensor=tensor, check_nans=True)
        image.set_check_nans(False)

    def test_get_center(self):
        tensor = torch.rand(1, 3, 3, 3)
        image = tio.ScalarImage(tensor=tensor)
        ras = image.get_center()
        lps = image.get_center(lps=True)
        assert ras == (1, 1, 1)
        assert lps == (-1, -1, 1)

    def test_with_list_of_missing_files(self):
        with pytest.raises(FileNotFoundError):
            tio.ScalarImage(path=['nopath', 'error'])

    def test_with_sequences_of_paths(self):
        shape = (5, 5, 5)
        path1 = self.get_image_path('path1', shape=shape)
        path2 = self.get_image_path('path2', shape=shape)
        paths_tuple = path1, path2
        paths_list = list(paths_tuple)
        for sequence in (paths_tuple, paths_list):
            image = tio.ScalarImage(path=sequence)
            assert image.shape == (2, 5, 5, 5)
            assert image[tio.STEM] == ['path1', 'path2']

    def test_with_a_list_of_images_with_different_shapes(self):
        path1 = self.get_image_path('path1', shape=(5, 5, 5))
        path2 = self.get_image_path('path2', shape=(7, 5, 5))
        image = tio.ScalarImage(path=[path1, path2])
        with pytest.raises(RuntimeError):
            image.load()

    def test_with_a_list_of_images_with_different_affines(self):
        path1 = self.get_image_path('path1', spacing=(1, 1, 1))
        path2 = self.get_image_path('path2', spacing=(1, 2, 1))
        image = tio.ScalarImage(path=[path1, path2])
        with pytest.warns(RuntimeWarning):
            image.load()

    def test_with_a_list_of_2d_paths(self):
        shape = (5, 6)
        path1 = self.get_image_path('path1', shape=shape, suffix='.nii')
        path2 = self.get_image_path('path2', shape=shape, suffix='.img')
        path3 = self.get_image_path('path3', shape=shape, suffix='.hdr')
        image = tio.ScalarImage(path=[path1, path2, path3])
        assert image.shape == (3, 5, 6, 1)
        assert image[tio.STEM] == ['path1', 'path2', 'path3']

    def test_axis_name_2d(self):
        path = self.get_image_path('im2d', shape=(5, 6))
        image = tio.ScalarImage(path)
        height_idx = image.axis_name_to_index('t')
        width_idx = image.axis_name_to_index('l')
        assert image.height == image.shape[height_idx]
        assert image.width == image.shape[width_idx]

    def test_different_shape(self):
        path_1 = self.get_image_path('im_shape1', shape=(5, 5, 5))
        path_2 = self.get_image_path('im_shape2', shape=(7, 5, 5))

        image = tio.ScalarImage([path_1, path_2])
        with pytest.raises(RuntimeError):
            image.load()

    @pytest.mark.slow
    @pytest.mark.skipif(sys.platform == 'win32', reason='Unstable on Windows')
    def test_plot(self):
        image = self.sample_subject.t1
        image.plot(show=False, output_path=self.dir / 'image.png')

    def test_data_type_uint16_array(self):
        tensor = np.random.rand(1, 3, 3, 3).astype(np.uint16)
        image = tio.ScalarImage(tensor=tensor)
        assert image.data.dtype == torch.int32

    def test_data_type_uint32_array(self):
        tensor = np.random.rand(1, 3, 3, 3).astype(np.uint32)
        image = tio.ScalarImage(tensor=tensor)
        assert image.data.dtype == torch.int64

    def test_save_image_with_data_type_boolean(self):
        tensor = np.random.rand(1, 3, 3, 3).astype(bool)
        image = tio.ScalarImage(tensor=tensor)
        image.save(self.dir / 'image.nii')

    def test_load_uint(self):
        affine = np.eye(4)
        for dtype in np.uint16, np.uint32:
            data = np.ones((3, 3, 3), dtype=dtype)
            img = nib.Nifti1Image(data, affine)
            with tempfile.NamedTemporaryFile(suffix='.nii', delete=False) as f:
                nib.save(img, f.name)
                tio.ScalarImage(f.name).load()

    def test_pil_3d(self):
        with pytest.raises(RuntimeError):
            tio.ScalarImage(tensor=torch.rand(1, 2, 3, 4)).as_pil()

    def test_pil_1(self):
        tio.ScalarImage(tensor=torch.rand(1, 2, 3, 1)).as_pil()

    def test_pil_2(self):
        with pytest.raises(RuntimeError):
            tio.ScalarImage(tensor=torch.rand(2, 2, 3, 1)).as_pil()

    def test_pil_3(self):
        tio.ScalarImage(tensor=torch.rand(3, 2, 3, 1)).as_pil()

    def test_set_data(self):
        im = self.sample_subject.t1
        with pytest.deprecated_call():
            im.data = im.data

    def test_no_type(self):
        with pytest.warns(FutureWarning):
            tio.Image(tensor=torch.rand(1, 2, 3, 4))

    def test_custom_reader(self):
        path = self.dir / 'im.npy'

        def numpy_reader(path):
            return np.load(path), np.eye(4)

        def assert_shape(shape_in, shape_out):
            np.save(path, np.random.rand(*shape_in))
            image = tio.ScalarImage(path, reader=numpy_reader)
            assert image.shape == shape_out

        assert_shape((5, 5), (1, 5, 5, 1))
        assert_shape((5, 5, 3), (3, 5, 5, 1))
        assert_shape((3, 5, 5), (3, 5, 5, 1))
        assert_shape((5, 5, 5), (1, 5, 5, 5))
        assert_shape((1, 5, 5, 5), (1, 5, 5, 5))
        assert_shape((4, 5, 5, 5), (4, 5, 5, 5))

    def test_fast_gif(self):
        with pytest.warns(RuntimeWarning):
            with tempfile.NamedTemporaryFile(suffix='.gif', delete=False) as f:
                self.sample_subject.t1.to_gif(0, 0.0001, f.name)

    def test_gif_rgb(self):
        with tempfile.NamedTemporaryFile(suffix='.gif', delete=False) as f:
            tio.ScalarImage(tensor=torch.rand(3, 4, 5, 6)).to_gif(0, 1, f.name)

    @pytest.mark.slow
    def test_hist(self):
        self.sample_subject.t1.hist(density=False, show=False)
        self.sample_subject.t1.hist(density=True, show=False)

    def test_count(self):
        image = self.sample_subject.label
        max_n = image.data.numel()
        nonzero = image.count_nonzero()
        assert 0 <= nonzero <= max_n
        counts = image.count_labels()
        assert tuple(counts) == (0, 1)
        assert 0 <= counts[0] <= max_n
        assert 0 <= counts[1] <= max_n

    def test_affine_multipath(self):
        # https://github.com/TorchIO-project/torchio/issues/762
        path1 = self.get_image_path('multi1')
        path2 = self.get_image_path('multi2')
        paths = path1, path2
        image = tio.ScalarImage(paths)
        self.assert_tensor_equal(image.affine, np.eye(4))

    def test_bad_numpy_type_reader(self):
        # https://github.com/TorchIO-project/torchio/issues/764
        def numpy_reader(path):
            return np.load(path), np.eye(4)

        tensor = np.random.rand(1, 2, 3, 4).astype(np.uint16)
        test_path = self.dir / 'test_image.npy'
        np.save(test_path, tensor)
        image = tio.ScalarImage(test_path, reader=numpy_reader)
        image.load()

    def test_load_unload(self):
        path = self.get_image_path('unload')
        image = tio.ScalarImage(path)
        with self.assertRaises(RuntimeError):
            image.unload()
        image.load()
        assert image._loaded
        image.unload()
        assert not image._loaded
        assert image[tio.DATA] is None
        assert image[tio.AFFINE] is None
        assert not image._loaded

    def test_unload_no_path(self):
        tensor = torch.rand(1, 2, 3, 4)
        image = tio.ScalarImage(tensor=tensor)
        with self.assertRaises(RuntimeError):
            image.unload()

    def test_copy_no_data(self):
        # https://github.com/TorchIO-project/torchio/issues/974
        path = self.get_image_path('im_copy')
        my_image = tio.LabelMap(path)
        assert not my_image._loaded
        new_image = copy.copy(my_image)
        assert not my_image._loaded
        assert not new_image._loaded

        my_image.load()
        new_image = copy.copy(my_image)
        assert my_image._loaded
        assert new_image._loaded

    def test_slicing(self):
        path = self.get_image_path('im_slicing')
        image = tio.ScalarImage(path)

        assert image.shape == (1, 10, 20, 30)

        cropped = image[0]
        assert cropped.shape == (1, 1, 20, 30)

        cropped = image[:, 2:-3]
        assert cropped.shape == (1, 10, 15, 30)

        cropped = image[-5:, 5:]
        assert cropped.shape == (1, 5, 15, 30)

        with pytest.raises(NotImplementedError):
            image[..., 5]

        with pytest.raises(ValueError):
            image[0:8:-1]

        with pytest.raises(ValueError):
            image[3::-1]

    def test_verify_path(self):
        path = Path(self.get_image_path('im_verify'))

        image = tio.ScalarImage(path, verify_path=False)
        assert image.path == path

        image = tio.ScalarImage(path, verify_path=True)
        assert image.path == path

        fake_path = Path('fake_path.nii')

        image = tio.ScalarImage(fake_path, verify_path=False)
        assert image.path == fake_path

        with pytest.raises(FileNotFoundError):
            tio.ScalarImage(fake_path, verify_path=True)

    def test_channels_last_deprecation_warning(self):
        """Passing channels_last kwarg emits a FutureWarning."""
        tensor = torch.rand(1, 10, 10, 10)
        with pytest.warns(FutureWarning, match='channels_last'):
            tio.ScalarImage(tensor=tensor, channels_last=True)

    def test_repr_html_returns_html(self):
        """_repr_html_ returns HTML string when matplotlib is available."""
        image = self.sample_subject.get_scalar_image('t1')
        html = image._repr_html_()
        assert isinstance(html, str)
        assert '<' in html

    def test_repr_html_fallback_without_matplotlib(self):
        """_repr_html_ falls back to __repr__ when matplotlib is absent."""
        image = self.sample_subject.get_scalar_image('t1')
        with patch.dict('sys.modules', {'matplotlib': None, 'matplotlib.figure': None}):
            result = image._repr_html_()
        assert 'ScalarImage' in result

    def test_lazy_load_data(self):
        """Accessing .data triggers lazy loading from path."""
        path = self.get_image_path('lazy_data')
        image = tio.ScalarImage(path)
        assert not image._loaded
        _ = image.data
        assert image._loaded

    def test_lazy_load_affine(self):
        """Accessing .affine when using custom reader triggers full load."""
        path = self.get_image_path('lazy_affine')

        def custom_reader(p):
            img = tio.ScalarImage(path=p)
            img.load()
            return img.data, np.eye(4)

        image = tio.ScalarImage(path, reader=custom_reader)
        assert not image._loaded
        _ = image.affine
        assert image._loaded

    def test_bounds_property(self):
        """bounds returns world positions of corner voxels."""
        image = self.sample_subject.get_scalar_image('t1')
        bounds = image.bounds
        assert bounds.shape == (2, 3)
        np.testing.assert_array_equal(bounds[0], [0, 0, 0])

    def test_get_bounds(self):
        """get_bounds returns min/max world coordinates with half-voxel offset."""
        image = self.sample_subject.get_scalar_image('t1')
        bounds = image.get_bounds()
        assert len(bounds) == 3
        for axis_bounds in bounds:
            assert len(axis_bounds) == 2

    def test_axis_name_to_index_non_string(self):
        """axis_name_to_index raises ValueError for non-string input."""
        image = self.sample_subject.get_scalar_image('t1')
        with pytest.raises(ValueError, match='string'):
            image.axis_name_to_index(123)

    def test_flip_axis_invalid_label(self):
        """flip_axis raises ValueError for unrecognized axis label."""
        with pytest.raises(ValueError, match='Axis not understood'):
            tio.ScalarImage.flip_axis('X')

    def test_tensor_as_path_raises(self):
        """Passing tensor as positional path argument gives helpful error."""
        with pytest.raises(TypeError, match='tensor=your_tensor'):
            tio.ScalarImage(torch.rand(1, 10, 10, 10))

    def test_numpy_as_path_raises(self):
        """Passing numpy array as path gives helpful error."""
        with pytest.raises(TypeError, match='tensor=your_tensor'):
            tio.ScalarImage(np.random.rand(1, 10, 10, 10))

    def test_bad_path_type_int(self):
        """Non-string/non-Path path types raise TypeError."""
        with pytest.raises(TypeError, match='path'):
            tio.ScalarImage(path=12345)

    def test_dict_as_path_raises(self):
        """Passing dict as path raises TypeError."""
        with pytest.raises(TypeError, match='cannot be a dictionary'):
            tio.ScalarImage(path={'file': 'a.nii'})

    def test_none_tensor_raises(self):
        """None tensor when not allowed raises RuntimeError."""
        image = tio.ScalarImage(tensor=torch.rand(1, 10, 10, 10))
        with pytest.raises(RuntimeError, match='cannot be None'):
            image._parse_tensor(None, none_ok=False)

    def test_invalid_tensor_type_raises(self):
        """Non-tensor/non-array types raise TypeError."""
        image = tio.ScalarImage(tensor=torch.rand(1, 10, 10, 10))
        with pytest.raises(TypeError, match='PyTorch tensor or NumPy'):
            image._parse_tensor([1, 2, 3])

    def test_non_4d_tensor_raises(self):
        """Non-4D tensors raise ValueError."""
        image = tio.ScalarImage(tensor=torch.rand(1, 10, 10, 10))
        with pytest.raises(ValueError, match='4D'):
            image._parse_tensor(torch.rand(10, 10, 10))

    def test_bool_tensor_converted_to_uint8(self):
        """Boolean tensors are automatically converted to uint8."""
        tensor = torch.randint(0, 2, (1, 5, 5, 5), dtype=torch.bool)
        image = tio.ScalarImage(tensor=tensor)
        assert image.data.dtype == torch.uint8

    def test_nan_warning_from_tensor(self):
        """NaN tensor with check_nans=True emits RuntimeWarning."""
        tensor = torch.full((1, 5, 5, 5), float('nan'))
        with pytest.warns(RuntimeWarning, match='NaNs found in tensor'):
            tio.ScalarImage(tensor=tensor, check_nans=True)

    def test_path_not_available_raises(self):
        """_as_path_list raises RuntimeError for tensor-only image."""
        image = tio.ScalarImage(tensor=torch.rand(1, 5, 5, 5))
        with pytest.raises(RuntimeError, match='path is not available'):
            image._as_path_list()

    def test_is_dir_none_path(self):
        """_is_dir returns False for tensor-only image."""
        image = tio.ScalarImage(tensor=torch.rand(1, 5, 5, 5))
        assert image._is_dir() is False

    def test_as_pil_no_transpose(self):
        """as_pil with transpose=False permutes differently from default."""
        tensor = torch.randint(0, 255, (1, 10, 10, 1), dtype=torch.float32)
        image = tio.ScalarImage(tensor=tensor)
        pil_image = image.as_pil(transpose=False)
        assert pil_image.size == (10, 10)

    def test_slice_type_error(self):
        """Unsupported slice types raise TypeError."""
        image = self.sample_subject.get_scalar_image('t1')
        with pytest.raises(TypeError, match='Slice type not understood'):
            image[0.5, :]  # float is invalid

    def test_multipath_as_path_list(self):
        """_as_path_list returns list for multipath images."""
        path1 = Path(self.get_image_path('multi1'))
        path2 = Path(self.get_image_path('multi2'))
        image = tio.ScalarImage(path=[path1, path2])
        paths = image._as_path_list()
        assert len(paths) == 2
