import copy
import os
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest
import torch

import torchio as tio

from .utils import TorchioTestCase


class TestUtils(TorchioTestCase):
    """Tests for `utils` module."""

    def test_to_tuple(self):
        assert tio.utils.to_tuple(1) == (1,)
        assert tio.utils.to_tuple((1,)) == (1,)
        assert tio.utils.to_tuple(1, length=3) == (1, 1, 1)
        assert tio.utils.to_tuple((1, 2)) == (1, 2)
        assert tio.utils.to_tuple((1, 2), length=3) == (1, 2)
        assert tio.utils.to_tuple([1, 2], length=3) == (1, 2)

    def test_get_stem(self):
        assert tio.utils.get_stem('/home/image.nii.gz') == 'image'
        assert tio.utils.get_stem('/home/image.nii') == 'image'
        assert tio.utils.get_stem('/home/image.nrrd') == 'image'

    def test_guess_type(self):
        assert tio.utils.guess_type('None') is None
        assert isinstance(tio.utils.guess_type('1'), int)
        assert isinstance(tio.utils.guess_type('1.5'), float)
        assert isinstance(tio.utils.guess_type('(1, 3, 5)'), tuple)
        assert isinstance(tio.utils.guess_type('(1,3,5)'), tuple)
        assert isinstance(tio.utils.guess_type('[1,3,5]'), list)
        assert isinstance(tio.utils.guess_type('test'), str)

    def test_apply_transform_to_file(self):
        transform = tio.RandomFlip()
        tio.utils.apply_transform_to_file(
            self.get_image_path('input'),
            transform,
            self.get_image_path('output'),
            verbose=True,
        )

    def test_subjects_from_batch(self):
        dataset = tio.SubjectsDataset(4 * [self.sample_subject])
        loader = tio.SubjectsLoader(dataset, batch_size=4)
        batch = tio.utils.get_first_item(loader)
        subjects = tio.utils.get_subjects_from_batch(batch)
        assert isinstance(subjects[0], tio.Subject)

    def test_subjects_from_batch_with_string_metadata(self):
        subject_c_with_string_metadata = tio.Subject(
            name='John Doe',
            label=tio.LabelMap(self.get_image_path('label_c', binary=True)),
        )

        dataset = tio.SubjectsDataset(4 * [subject_c_with_string_metadata])
        loader = tio.SubjectsLoader(dataset, batch_size=4)
        batch = tio.utils.get_first_item(loader)
        subjects = tio.utils.get_subjects_from_batch(batch)
        assert isinstance(subjects[0], tio.Subject)
        assert 'label' in subjects[0]
        assert 'name' in subjects[0]

    def test_subjects_from_batch_with_int_metadata(self):
        subject_c_with_int_metadata = tio.Subject(
            age=45,
            label=tio.LabelMap(self.get_image_path('label_c', binary=True)),
        )
        dataset = tio.SubjectsDataset(4 * [subject_c_with_int_metadata])
        loader = tio.SubjectsLoader(dataset, batch_size=4)
        batch = tio.utils.get_first_item(loader)
        subjects = tio.utils.get_subjects_from_batch(batch)
        assert isinstance(subjects[0], tio.Subject)
        assert 'label' in subjects[0]
        assert 'age' in subjects[0]

    def test_add_images_from_batch(self):
        subject = copy.deepcopy(self.sample_subject)
        subjects = 4 * [subject]
        preds = torch.rand(4, *subject.shape)
        tio.utils.add_images_from_batch(subjects, preds)

    def test_empty_batch(self):
        with pytest.raises(RuntimeError):
            tio.utils.get_batch_images_and_size({})

    def test_compress_default_output(self):
        """compress() creates .nii.gz from input with default output."""
        with tempfile.TemporaryDirectory() as tmp:
            input_path = Path(tmp) / 'test.nii'
            input_path.write_bytes(b'fake nifti data')
            result = tio.utils.compress(input_path)
            assert result.exists()
            assert result.suffix == '.gz'

    def test_compress_explicit_output(self):
        """compress() writes to the specified output path."""
        with tempfile.TemporaryDirectory() as tmp:
            input_path = Path(tmp) / 'test.nii'
            input_path.write_bytes(b'fake nifti data')
            output_path = Path(tmp) / 'compressed.nii.gz'
            result = tio.utils.compress(input_path, output_path)
            assert result == output_path
            assert result.exists()

    def test_history_collate_non_subject(self):
        """history_collate returns empty dict for non-Subject batches."""
        result = tio.utils.history_collate([{'a': 1}])
        assert result == {}

    def test_create_dummy_dataset_force(self):
        """create_dummy_dataset with force=True recreates directories."""
        with tempfile.TemporaryDirectory() as tmp:
            subjects = tio.utils.create_dummy_dataset(
                num_images=2,
                size_range=(5, 10),
                directory=tmp,
            )
            assert len(subjects) == 2
            # Call again with force to recreate
            subjects = tio.utils.create_dummy_dataset(
                num_images=2,
                size_range=(5, 10),
                directory=tmp,
                force=True,
            )
            assert len(subjects) == 2

    def test_create_dummy_dataset_existing(self):
        """create_dummy_dataset loads from existing dirs without force."""
        with tempfile.TemporaryDirectory() as tmp:
            tio.utils.create_dummy_dataset(
                num_images=2,
                size_range=(5, 10),
                directory=tmp,
            )
            # Call again without force — loads from existing paths
            subjects = tio.utils.create_dummy_dataset(
                num_images=2,
                size_range=(5, 10),
                directory=tmp,
            )
            assert len(subjects) == 2

    def test_create_dummy_dataset_verbose(self):
        """create_dummy_dataset with verbose=True prints a message."""
        with tempfile.TemporaryDirectory() as tmp:
            subjects = tio.utils.create_dummy_dataset(
                num_images=1,
                size_range=(5, 10),
                directory=tmp,
                verbose=True,
            )
            assert len(subjects) == 1

    def test_parse_spatial_shape_wrong_length(self):
        """parse_spatial_shape raises ValueError for non-3-element shapes."""
        with pytest.raises(ValueError, match='3 elements'):
            tio.utils.parse_spatial_shape((10, 20))

    def test_normalize_path(self):
        """normalize_path expands ~ and resolves to absolute path."""
        result = tio.utils.normalize_path('~/test')
        assert result.is_absolute()
        assert '~' not in str(result)

    def test_guess_external_viewer_env_var(self):
        """guess_external_viewer returns SITK_SHOW_COMMAND if set."""
        with patch.dict(os.environ, {'SITK_SHOW_COMMAND': '/usr/bin/viewer'}):
            result = tio.utils.guess_external_viewer()
            assert result == Path('/usr/bin/viewer')

    def test_guess_external_viewer_linux_itksnap(self):
        """guess_external_viewer finds itksnap on Linux."""
        with (
            patch('torchio.utils.sys') as mock_sys,
            patch('torchio.utils.shutil') as mock_shutil,
            patch.dict(os.environ, {}, clear=False),
        ):
            # Remove SITK_SHOW_COMMAND if present
            os.environ.pop('SITK_SHOW_COMMAND', None)
            mock_sys.platform = 'linux'
            mock_shutil.which = lambda x: '/usr/bin/itksnap' if x == 'itksnap' else None
            result = tio.utils.guess_external_viewer()
            assert result == Path('/usr/bin/itksnap')

    def test_guess_external_viewer_none(self):
        """guess_external_viewer returns None when no viewer found."""
        with (
            patch('torchio.utils.sys') as mock_sys,
            patch('torchio.utils.shutil') as mock_shutil,
            patch.dict(os.environ, {}, clear=False),
        ):
            os.environ.pop('SITK_SHOW_COMMAND', None)
            mock_sys.platform = 'unknown_platform'
            mock_shutil.which = lambda x: None
            result = tio.utils.guess_external_viewer()
            assert result is None

    def test_guess_external_viewer_linux_slicer(self):
        """guess_external_viewer finds Slicer on Linux when itksnap absent."""
        with (
            patch('torchio.utils.sys') as mock_sys,
            patch('torchio.utils.shutil') as mock_shutil,
            patch.dict(os.environ, {}, clear=False),
        ):
            os.environ.pop('SITK_SHOW_COMMAND', None)
            mock_sys.platform = 'linux'
            mock_shutil.which = lambda x: '/usr/bin/Slicer' if x == 'Slicer' else None
            result = tio.utils.guess_external_viewer()
            assert result == Path('/usr/bin/Slicer')

    def test_apply_transform_verbose_history(self):
        """apply_transform_to_file with verbose prints history."""
        transform = tio.RandomFlip()
        tio.utils.apply_transform_to_file(
            self.get_image_path('input_v'),
            transform,
            self.get_image_path('output_v'),
            verbose=True,
        )
