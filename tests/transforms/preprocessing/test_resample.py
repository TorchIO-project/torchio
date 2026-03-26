from typing import Any
from typing import cast

import numpy as np
import pytest
import torch

import torchio as tio

from ...utils import TorchioTestCase


def _apply_resample(target: object, subject: tio.Subject) -> tio.Subject:
    transformed = tio.Resample(cast(Any, target))(subject)
    assert isinstance(transformed, tio.Subject)
    return transformed


class TestResample(TorchioTestCase):
    """Tests for `Resample`."""

    def test_spacing(self):
        # Should this raise an error if sizes are different?
        spacing = 2
        transform = tio.Resample(spacing)
        transformed = transform(self.sample_subject)
        for image in transformed.get_images(intensity_only=False):
            assert image.spacing == 3 * (spacing,)

    def test_reference_name(self):
        subject = self.get_inconsistent_shape_subject()
        reference_name = 't1'
        transform = tio.Resample(reference_name)
        transformed = transform(subject)
        reference_image = subject[reference_name]
        for image in transformed.get_images(intensity_only=False):
            assert reference_image.shape == image.shape
            self.assert_tensor_almost_equal(
                reference_image.affine,
                image.affine,
            )

    def test_affine(self):
        spacing = 1
        affine_name = 'pre_affine'
        transform = tio.Resample(spacing, pre_affine_name=affine_name)
        transformed = transform(self.sample_subject)
        for image in transformed.values():
            if affine_name in image:
                target_affine = np.eye(4)
                target_affine[:3, 3] = 10, 0, -0.1
                self.assert_tensor_almost_equal(image.affine, target_affine)
            else:
                self.assert_tensor_equal(image.affine, np.eye(4))

    def test_missing_affine(self):
        transform = tio.Resample(1, pre_affine_name='missing')
        with pytest.raises(ValueError):
            transform(self.sample_subject)

    def test_reference_path(self):
        reference_image, reference_path = self.get_reference_image_and_path()
        transform = tio.Resample(reference_path)
        transformed = transform(self.sample_subject)
        for image in transformed.values():
            assert reference_image.shape == image.shape
            self.assert_tensor_almost_equal(
                reference_image.affine,
                image.affine,
            )

    def test_wrong_spacing_length(self):
        with pytest.raises(RuntimeError):
            _apply_resample((1, 2), self.sample_subject)

    def test_wrong_spacing_value(self):
        with pytest.raises(ValueError):
            tio.Resample(0)(self.sample_subject)

    def test_wrong_target_type(self):
        with pytest.raises(RuntimeError):
            tio.Resample(None)(self.sample_subject)

    def test_missing_reference(self):
        transform = tio.Resample('missing')
        with pytest.raises(ValueError):
            transform(self.sample_subject)

    def test_2d(self):
        """Check that image is still 2D after resampling."""
        image = tio.ScalarImage(tensor=torch.rand(1, 2, 3, 1))
        transform = tio.Resample(0.5)
        shape = transform(image).shape
        assert shape == (1, 4, 6, 1)

    def test_input_list(self):
        _apply_resample([1, 2, 3], self.sample_subject)

    def test_input_array(self):
        _apply_resample(np.asarray([1, 2, 3]), self.sample_subject)

    def test_image_target(self):
        tio.Resample(self.sample_subject.t1)(self.sample_subject)

    def test_bad_affine(self):
        shape = 1, 2, 3
        affine = np.eye(3)
        target = shape, affine
        transform = tio.Resample(target)
        with pytest.raises(RuntimeError):
            transform(self.sample_subject)

    def test_resample_flip_consistent(self):
        image = torch.rand(1, 10, 10, 10)
        resample = tio.Resample(1.35)
        flip = tio.Flip(0)
        flipped_and_resampled = resample(flip(image))
        resampled_and_flipped = flip(resample(image))
        self.assert_tensor_almost_equal(
            flipped_and_resampled.data,
            resampled_and_flipped.data,
        )

    def test_wrong_spacing_type_raises(self):
        """Passing a completely invalid type to _parse_spacing raises."""
        with pytest.raises(ValueError, match='Target must be'):
            tio.Resample._parse_spacing(cast(Any, object()))

    def test_negative_spacing_value_raises(self):
        """Negative spacing values should raise ValueError."""
        with pytest.raises(ValueError, match='strictly positive'):
            tio.Resample(-1)(self.sample_subject)

    def test_ndarray_wrong_length_raises(self):
        """NumPy array with != 3 elements as target raises ValueError."""
        with pytest.raises(ValueError, match='Target must be'):
            tio.Resample._parse_spacing(np.array([1.0, 2.0]))

    def test_sequence_wrong_length_raises(self):
        """Sequence with != 3 elements as target raises ValueError."""
        with pytest.raises(ValueError, match='Target must be'):
            tio.Resample._parse_spacing([1.0, 2.0])

    def test_sequence_non_numeric_element_raises(self):
        """Sequence containing a non-numeric element raises ValueError."""
        with pytest.raises(ValueError, match='Target must be'):
            tio.Resample(cast(Any, [1.0, 'bad', 3.0]))(self.sample_subject)

    def test_antialias_downsampling(self):
        """Anti-aliasing should be applied during downsampling."""
        image = tio.ScalarImage(tensor=torch.rand(1, 20, 20, 20))
        resample = tio.Resample(2, antialias=True)
        resampled = resample(image)
        assert resampled.spatial_shape == (10, 10, 10)

    def test_pre_affine_as_torch_tensor(self):
        """Pre-affine matrix as a torch.Tensor should be converted to numpy."""
        subject = tio.Subject(
            t1=tio.ScalarImage(
                self.get_image_path('t1_torch_aff'),
                pre_affine=torch.eye(4),
            ),
        )
        transform = tio.Resample(1, pre_affine_name='pre_affine')
        transform(subject)

    def test_check_affine_non_string_name_raises(self):
        """Non-string affine_name raises TypeError in check_affine."""
        image = tio.ScalarImage(tensor=torch.rand(1, 4, 4, 4))
        with pytest.raises(TypeError, match='must be a string'):
            tio.Resample.check_affine(cast(Any, 123), image)

    def test_check_affine_wrong_type_raises(self):
        """Affine matrix that is not ndarray or Tensor raises TypeError."""
        image = tio.ScalarImage(tensor=torch.rand(1, 4, 4, 4))
        image['my_affine'] = [[1, 0], [0, 1]]
        with pytest.raises(TypeError, match='must be a NumPy array'):
            tio.Resample.check_affine('my_affine', image)

    def test_check_affine_wrong_shape_raises(self):
        """Affine matrix with shape != (4, 4) raises ValueError."""
        image = tio.ScalarImage(tensor=torch.rand(1, 4, 4, 4))
        image['my_affine'] = np.eye(3)
        with pytest.raises(ValueError, match='must be .4, 4.'):
            tio.Resample.check_affine('my_affine', image)

    def test_target_is_same_image_skips_resampling(self):
        """When target is the same Image object, it is skipped."""
        image = tio.ScalarImage(tensor=torch.rand(1, 10, 10, 10))
        subject = tio.Subject(t1=image)
        transform = tio.Resample(target=image, copy=False)
        transformed = transform(subject)
        assert transformed['t1'].spatial_shape == image.spatial_shape

    def test_unrecognized_target_raises_runtime_error(self):
        """A target that doesn't match any dispatch branch raises RuntimeError."""
        with pytest.raises(RuntimeError, match='Target not understood'):
            tio.Resample(cast(Any, object()))(self.sample_subject)
