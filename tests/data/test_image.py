#!/usr/bin/env python
"""Tests for Image."""

import builtins
import copy
import sys
import tempfile
from pathlib import Path
from unittest.mock import MagicMock
from unittest.mock import patch

import nibabel as nib
import numpy as np
import pytest
import torch
from matplotlib.figure import Figure

import torchio as tio

from ..utils import TorchioTestCase


def _raise_import_error(module_name: str):
    real_import = builtins.__import__

    def side_effect(name, *args, **kwargs):
        if name == module_name:
            raise ModuleNotFoundError(module_name)
        return real_import(name, *args, **kwargs)

    return side_effect


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

    def test_repr_html(self):
        image = self.sample_subject.t1
        with (
            patch.object(tio.ScalarImage, 'plot', return_value=Figure()),
            patch('torchio.visualization._figure_to_html', return_value='<figure>'),
        ):
            assert image._repr_html_() == '<figure>'

        with patch(
            'builtins.__import__',
            side_effect=_raise_import_error('matplotlib.figure'),
        ):
            assert image._repr_html_() == repr(image)

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

    def test_channels_last(self):
        with pytest.warns(FutureWarning, match='channels_last'):
            tio.ScalarImage(
                tensor=torch.rand(1, 2, 3, 4),
                channels_last=True,
            )

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

    def test_reload_data_and_affine_after_unload(self):
        path_1 = self.get_image_path('reload_1')
        path_2 = self.get_image_path('reload_2')

        image = tio.ScalarImage([path_1, path_2])
        image.load()
        image.unload()

        assert image.data.shape == (2, 10, 20, 30)
        image.unload()
        np.testing.assert_array_equal(image.affine, np.eye(4))

    def test_unload_no_path(self):
        tensor = torch.rand(1, 2, 3, 4)
        image = tio.ScalarImage(tensor=tensor)
        with self.assertRaises(RuntimeError):
            image.unload()

    def test_image_helpers_without_path(self):
        image = tio.ScalarImage(tensor=torch.rand(1, 2, 3, 4))

        with pytest.raises(RuntimeError, match='Image path is not available'):
            image._as_path_list()

        assert not image._is_dir()

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

    def test_bad_slice_type(self):
        image = tio.ScalarImage(tensor=torch.rand(1, 2, 3, 4))

        with pytest.raises(TypeError, match='Slice type not understood'):
            image._crop_from_slices((object(),))

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

    def test_private_path_parsing_errors(self):
        image = tio.ScalarImage(tensor=torch.rand(1, 2, 3, 4))

        with pytest.raises(TypeError, match='tensor=your_tensor'):
            image._parse_single_path(torch.rand(1))

        with pytest.raises(TypeError, match='Expected type str or Path'):
            image._parse_single_path(object())

        with pytest.raises(TypeError, match='cannot be a dictionary'):
            image._parse_path({})

    def test_private_tensor_parsing_errors(self):
        image = tio.ScalarImage(tensor=torch.rand(1, 2, 3, 4))

        with pytest.raises(RuntimeError, match='Input tensor cannot be None'):
            image._parse_tensor(None, none_ok=False)

        with pytest.raises(TypeError, match='PyTorch tensor or NumPy array'):
            image._parse_tensor('bad tensor')

        with pytest.raises(ValueError, match='Input tensor must be 4D'):
            image._parse_tensor(torch.rand(1, 2, 3))

    def test_read_and_check_warns_on_nans(self):
        path = self.dir / 'nan_image.npy'
        np.save(path, np.full((1, 2, 3, 4), np.nan))

        def numpy_reader(path):
            return np.load(path), np.eye(4)

        image = tio.ScalarImage(path, check_nans=True, reader=numpy_reader)

        with pytest.warns(RuntimeWarning, match='NaNs found in file'):
            image.load()

    def test_bounds_and_axis_validation(self):
        image = tio.ScalarImage(tensor=torch.rand(1, 2, 3, 4))

        np.testing.assert_array_equal(
            image.bounds,
            np.array(((0, 0, 0), (1, 2, 3))),
        )
        assert image.get_bounds() == ((-0.5, 1.5), (-0.5, 2.5), (-0.5, 3.5))

        with pytest.raises(ValueError, match='Axis must be a string'):
            image.axis_name_to_index(0)

        with pytest.raises(ValueError, match='Axis not understood'):
            image.flip_axis('x')

    def test_direction_must_be_3d(self):
        image = tio.ScalarImage(tensor=torch.rand(1, 2, 3, 4))

        with (
            patch(
                'torchio.data.image.get_sitk_metadata_from_ras_affine',
                return_value=((0, 0, 0), (1, 1, 1), (1, 0, 0, 1)),
            ),
            pytest.raises(RuntimeError, match='Expected a 3D direction'),
        ):
            _ = image.direction

    def test_as_pil_without_pillow(self):
        image = tio.ScalarImage(tensor=torch.rand(1, 2, 3, 1))

        with (
            patch('builtins.__import__', side_effect=_raise_import_error('PIL')),
            pytest.raises(RuntimeError, match='Please install Pillow'),
        ):
            image.as_pil()

    def test_as_pil_without_transpose(self):
        image = tio.ScalarImage(tensor=torch.rand(4, 2, 3, 1))
        pil_image = image.as_pil(transpose=False)
        assert pil_image.size == (3, 2)

    def test_to_ras(self):
        image = tio.ScalarImage(tensor=torch.rand(1, 2, 3, 4))
        assert image.to_ras() is image

        non_ras = tio.ScalarImage(
            tensor=torch.rand(1, 2, 3, 4),
            affine=np.diag((-1, 1, 1, 1)),
        )
        assert non_ras.orientation_str != 'RAS'
        assert non_ras.to_ras().orientation_str == 'RAS'

    def test_plot_branches(self):
        image_2d = tio.ScalarImage(tensor=torch.rand(1, 2, 3, 1))
        pil_image = MagicMock()
        with patch.object(tio.ScalarImage, 'as_pil', return_value=pil_image):
            assert image_2d.plot() is None
        pil_image.show.assert_called_once_with()

        image_3d = tio.ScalarImage(tensor=torch.rand(1, 2, 3, 4))
        with patch('torchio.visualization.plot_volume', return_value='figure'):
            assert image_3d.plot(return_fig=True) == 'figure'

    def test_show_with_explicit_viewer(self):
        class DummyViewer:
            def __init__(self):
                self.application = None
                self.extension = None
                self.calls = 0

            def SetFileExtension(self, extension):
                self.extension = extension

            def SetApplication(self, application):
                self.application = application

            def Execute(self, image):
                self.calls += 1

        viewer = DummyViewer()
        viewer_path = self.dir / 'viewer'
        image = tio.LabelMap(tensor=torch.rand(1, 2, 3, 4) > 0.5)

        with patch('torchio.data.image.sitk.ImageViewer', return_value=viewer):
            image.show(viewer_path=viewer_path)

        assert viewer.extension == '.seg.nrrd'
        assert viewer.application == str(viewer_path)
        assert viewer.calls == 1

    def test_show_with_guessed_viewer(self):
        class DummyViewer:
            def __init__(self):
                self.application = None
                self.calls = 0

            def SetFileExtension(self, extension):
                pass

            def SetApplication(self, application):
                self.application = application

            def Execute(self, image):
                self.calls += 1
                if self.calls == 1:
                    raise RuntimeError('viewer failed')

        viewer = DummyViewer()
        guessed_viewer = self.dir / 'guessed-viewer'

        with (
            patch('torchio.data.image.sitk.ImageViewer', return_value=viewer),
            patch(
                'torchio.data.image.guess_external_viewer',
                return_value=guessed_viewer,
            ),
        ):
            tio.ScalarImage(tensor=torch.rand(1, 2, 3, 4)).show()

        assert viewer.application == str(guessed_viewer)
        assert viewer.calls == 2

    def test_show_without_available_viewer(self):
        class DummyViewer:
            def SetFileExtension(self, extension):
                pass

            def SetApplication(self, application):
                pass

            def Execute(self, image):
                raise RuntimeError('viewer failed')

        with (
            patch('torchio.data.image.sitk.ImageViewer', return_value=DummyViewer()),
            patch('torchio.data.image.guess_external_viewer', return_value=None),
            pytest.raises(RuntimeError, match='No external viewer has been found'),
        ):
            tio.ScalarImage(tensor=torch.rand(1, 2, 3, 4)).show()

    def test_to_video(self):
        image = tio.ScalarImage(tensor=torch.rand(1, 2, 3, 4))

        with patch('torchio.visualization.make_video') as make_video:
            image.to_video(self.dir / 'video.mp4')

        make_video.assert_called_once()
