import copy
import sys
import tempfile
from typing import cast
from unittest.mock import patch

import numpy as np
import pytest
import torch

import torchio as tio

from ..utils import TorchioTestCase


class TestSubject(TorchioTestCase):
    """Tests for `Subject`."""

    def test_positional_args(self):
        with pytest.raises(ValueError):
            tio.Subject(cast(dict[str, object], 0))

    def test_input_dict(self):
        with tempfile.NamedTemporaryFile(delete=False) as f:
            input_dict = {'image': tio.ScalarImage(f.name)}
            tio.Subject(input_dict)
            tio.Subject(**input_dict)

    def test_no_sample(self):
        with tempfile.NamedTemporaryFile(delete=False) as f:
            input_dict = {'image': tio.ScalarImage(f.name)}
            subject = tio.Subject(input_dict)
            with pytest.raises(RuntimeError):
                with pytest.warns(UserWarning):
                    tio.RandomFlip()(subject)

    def test_history(self):
        transformed = tio.RandomGamma()(self.sample_subject)
        assert len(transformed.history) == 1

    def test_inconsistent_shape(self):
        subject = tio.Subject(
            a=tio.ScalarImage(tensor=torch.rand(1, 2, 3, 4)),
            b=tio.ScalarImage(tensor=torch.rand(2, 2, 3, 4)),
        )
        _ = subject.spatial_shape
        with pytest.raises(RuntimeError):
            _ = subject.shape

    def test_inconsistent_spatial_shape(self):
        subject = tio.Subject(
            a=tio.ScalarImage(tensor=torch.rand(1, 3, 3, 4)),
            b=tio.ScalarImage(tensor=torch.rand(2, 2, 3, 4)),
        )
        with pytest.raises(RuntimeError):
            _ = subject.spatial_shape

    @pytest.mark.slow
    @pytest.mark.skipif(sys.platform == 'win32', reason='Unstable on Windows')
    def test_plot(self):
        self.sample_subject.plot(
            show=False,
            output_path=self.dir / 'figure.png',
            cmap_dict={
                't2': 'viridis',
                'label': {0: 'yellow', 1: 'blue'},
            },
        )

    @pytest.mark.slow
    @pytest.mark.skipif(sys.platform == 'win32', reason='Unstable on Windows')
    def test_plot_one_image(self):
        path = self.get_image_path('t1_plot')
        subject = tio.Subject(t1=tio.ScalarImage(path))
        subject.plot(show=False)

    def test_same_space(self):
        # https://github.com/TorchIO-project/torchio/issues/381
        affine1 = np.array(
            [
                [
                    4.27109375e-14,
                    -8.71264808e-03,
                    9.99876633e-01,
                    -3.39850907e01,
                ],
                [
                    -5.54687500e-01,
                    -2.71630469e-12,
                    8.75148028e-17,
                    1.62282930e02,
                ],
                [
                    2.71575000e-12,
                    -5.54619070e-01,
                    -1.57073092e-02,
                    2.28515784e02,
                ],
                [0.00000000e00, 0.00000000e00, 0.00000000e00, 1.00000000e00],
            ]
        )
        affine2 = np.array(
            [
                [
                    3.67499773e-08,
                    -8.71257665e-03,
                    9.99876635e-01,
                    -3.39850922e01,
                ],
                [
                    -5.54687500e-01,
                    3.67499771e-08,
                    6.73024385e-08,
                    1.62282928e02,
                ],
                [
                    -3.73318194e-08,
                    -5.54619071e-01,
                    -1.57071802e-02,
                    2.28515778e02,
                ],
                [0.00000000e00, 0.00000000e00, 0.00000000e00, 1.00000000e00],
            ]
        )
        t = torch.rand(1, 2, 3, 4)
        subject = tio.Subject(
            im1=tio.ScalarImage(tensor=t, affine=affine1),
            im2=tio.ScalarImage(tensor=t, affine=affine2),
        )
        subject.check_consistent_space()

    def test_delete_image(self):
        subject = copy.deepcopy(self.sample_subject)
        subject.remove_image('t1')
        with pytest.raises(KeyError):
            subject['t1']
        with pytest.raises(AttributeError):
            _ = subject.t1

    def test_2d(self):
        subject = self.make_2d(self.sample_subject)
        assert subject.is_2d()

    def test_different_non_numeric(self):
        with pytest.raises(RuntimeError):
            self.sample_subject.check_consistent_attribute('path')

    def test_bad_arg(self):
        with pytest.raises(ValueError):
            tio.Subject(cast(dict[str, object], 0))

    def test_no_images(self):
        with pytest.raises(TypeError):
            tio.Subject(a=0)

    def test_copy_subject(self):
        sub_copy = copy.copy(self.sample_subject)
        assert isinstance(sub_copy, tio.data.Subject)
        sub_deep_copy = copy.deepcopy(self.sample_subject)
        assert isinstance(sub_deep_copy, tio.data.Subject)

    def test_copy_subclass(self):
        class DummySubjectSubClass(tio.data.Subject):
            def __init__(self, **kwargs):
                super().__init__(**kwargs)

        dummy_sub = DummySubjectSubClass(
            attr_1='abcd',
            attr_2=tio.ScalarImage(tensor=torch.zeros(1, 1, 1, 1)),
        )
        sub_copy = copy.copy(dummy_sub)
        assert isinstance(sub_copy, tio.data.Subject)
        assert isinstance(sub_copy, DummySubjectSubClass)
        sub_deep_copy = copy.deepcopy(dummy_sub)
        assert isinstance(sub_deep_copy, tio.data.Subject)
        assert isinstance(sub_deep_copy, DummySubjectSubClass)

    def test_load_unload(self):
        self.sample_subject.load()
        for image in self.sample_subject.get_images(intensity_only=False):
            assert image._loaded
        self.sample_subject.unload()
        for image in self.sample_subject.get_images(intensity_only=False):
            assert not image._loaded

    def test_subjects_batch(self):
        subjects = tio.SubjectsDataset(10 * [self.sample_subject])
        loader = tio.SubjectsLoader(subjects, batch_size=4)
        batch = next(iter(loader))
        assert batch.__class__ is dict

    def test_deep_copy_subject(self):
        sub_copy = copy.deepcopy(self.sample_subject)
        assert isinstance(sub_copy, tio.data.Subject)

        sub_copy_t1 = sub_copy.get_scalar_image('t1')
        sample_t1 = self.sample_subject.get_scalar_image('t1')
        new_tensor = torch.ones_like(sub_copy_t1.data)
        sub_copy_t1.set_data(new_tensor)
        # The data of the original subject should not be modified
        assert not torch.allclose(sub_copy_t1.data, sample_t1.data)

    def test_shallow_copy_subject(self):
        # We are creating a deep copy of the original subject first to not modify the original subject
        copy_original_subj = copy.deepcopy(self.sample_subject)
        sub_copy = copy.copy(copy_original_subj)
        assert isinstance(sub_copy, tio.data.Subject)

        sub_copy_t1 = sub_copy.get_scalar_image('t1')
        copy_original_t1 = copy_original_subj.get_scalar_image('t1')
        sample_t1 = self.sample_subject.get_scalar_image('t1')
        new_tensor = torch.ones_like(sub_copy_t1.data)
        sub_copy_t1.set_data(new_tensor)

        # The data of both copies needs to be the same as we are using a shallow copy
        assert torch.allclose(sub_copy_t1.data, copy_original_t1.data)
        # The data of the original subject should not be modified
        assert not torch.allclose(sub_copy_t1.data, sample_t1.data)
        assert not torch.allclose(copy_original_t1.data, sample_t1.data)

    def test_getitem_with_int(self):
        """Integer indexing returns cropped Subject along first spatial dim."""
        subject = self.sample_subject
        result = subject[0]
        assert isinstance(result, tio.Subject)

    def test_getitem_with_slice(self):
        """Slice indexing returns cropped Subject."""
        subject = self.sample_subject
        result = subject[0:5]
        assert isinstance(result, tio.Subject)

    def test_getitem_with_tuple(self):
        """Tuple indexing crops multiple spatial dimensions."""
        subject = self.sample_subject
        result = subject[0:5, 0:5, 0:5]
        assert isinstance(result, tio.Subject)

    def test_getitem_inconsistent_shapes_raises(self):
        """Indexing with inconsistent image shapes raises RuntimeError."""
        subject = self.get_inconsistent_shape_subject()
        with pytest.raises(RuntimeError, match='same spatial shape'):
            subject[0:5]

    def test_get_applied_transforms_ignore_intensity(self):
        """ignore_intensity=True filters out intensity transforms."""
        subject = copy.deepcopy(self.sample_subject)
        transform = tio.RandomNoise()
        transformed = transform(subject)
        transforms = transformed.get_applied_transforms(ignore_intensity=True)
        assert len(transforms) == 0

    def test_check_consistent_affine_raises(self):
        """Inconsistent affines raise RuntimeError."""
        t1 = tio.ScalarImage(tensor=torch.rand(1, 5, 5, 5))
        affine = np.eye(4)
        affine[0, 0] = 2.0
        t2 = tio.ScalarImage(tensor=torch.rand(1, 5, 5, 5), affine=affine)
        subject = tio.Subject(t1=t1, t2=t2)
        with pytest.raises(RuntimeError):
            subject.check_consistent_affine()

    def test_check_consistent_space_raises(self):
        """Inconsistent spacing raises RuntimeError with helpful message."""
        t1 = tio.ScalarImage(tensor=torch.rand(1, 5, 5, 5))
        affine = np.diag([2.0, 1.0, 1.0, 1.0])
        t2 = tio.ScalarImage(tensor=torch.rand(1, 5, 5, 5), affine=affine)
        subject = tio.Subject(t1=t1, t2=t2)
        with pytest.raises(RuntimeError, match='ToCanonical'):
            subject.check_consistent_space()

    def test_get_scalar_image_on_label_raises(self):
        """get_scalar_image on a LabelMap raises TypeError."""
        subject = self.sample_subject
        with pytest.raises(TypeError, match='not a scalar image'):
            subject.get_scalar_image('label')

    def test_get_label_map_on_scalar_raises(self):
        """get_label_map on a ScalarImage raises TypeError."""
        subject = self.sample_subject
        with pytest.raises(TypeError, match='not a label map'):
            subject.get_label_map('t1')

    def test_check_image_name_non_string(self):
        """_check_image_name raises ValueError for non-string input."""
        with pytest.raises(ValueError, match='must be a string'):
            tio.Subject._check_image_name(123)

    def test_add_image_non_image(self):
        """add_image raises ValueError for non-Image objects."""
        subject = self.sample_subject
        with pytest.raises(ValueError, match='instance of torchio.Image'):
            subject.add_image('not_an_image', 'bad')

    def test_repr_html_returns_html(self):
        """_repr_html_ returns HTML string when matplotlib is available."""
        subject = self.sample_subject
        html = subject._repr_html_()
        assert isinstance(html, str)
        assert '<' in html

    def test_repr_html_fallback_without_matplotlib(self):
        """_repr_html_ falls back to __repr__ when matplotlib is absent."""
        subject = self.sample_subject
        with patch.dict('sys.modules', {'matplotlib': None, 'matplotlib.figure': None}):
            result = subject._repr_html_()
        assert 'Subject' in result

    def test_plot_returns_none_by_default(self):
        """plot() returns None when return_fig is False."""
        result = self.sample_subject.plot(return_fig=False, show=False)
        assert result is None

    def test_plot_returns_figure(self):
        """plot(return_fig=True) returns a matplotlib Figure."""
        from matplotlib.figure import Figure

        result = self.sample_subject.plot(return_fig=True, show=False)
        assert isinstance(result, Figure)
