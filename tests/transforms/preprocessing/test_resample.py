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

    def test_shape_affine_target(self):
        shape = 6, 5, 4
        affine = np.diag((2.0, 3.0, 4.0, 1.0))
        transformed = tio.Resample((shape, affine))(self.sample_subject)
        for image in transformed.get_images(intensity_only=False):
            assert image.spatial_shape == shape
            self.assert_tensor_almost_equal(image.affine, affine)

    def test_scalars_only(self):
        transformed = tio.Resample(2, scalars_only=True)(self.sample_subject)
        assert transformed.t1.spacing == (2.0, 2.0, 2.0)
        assert transformed.t2.spacing == (2.0, 2.0, 2.0)
        self.assert_tensor_equal(transformed.label.data, self.sample_subject.label.data)
        self.assert_tensor_almost_equal(
            transformed.label.affine,
            self.sample_subject.label.affine,
        )

    def test_bad_affine(self):
        shape = 1, 2, 3
        affine = np.eye(3)
        target = shape, affine
        transform = tio.Resample(target)
        with pytest.raises(RuntimeError):
            transform(self.sample_subject)

    def test_resample_flip_consistent(self):
        """Flipping before or after resampling should yield the same data."""
        image = torch.rand(1, 10, 10, 10)
        resample = tio.Resample(1.35)
        flip = tio.Flip(0)
        flipped_and_resampled = resample(flip(image))
        resampled_and_flipped = flip(resample(image))
        self.assert_tensor_almost_equal(
            flipped_and_resampled.data,
            resampled_and_flipped.data,
        )

    def test_antialias(self):
        """Antialiasing should change a downsampled checkerboard pattern."""
        checkerboard = np.indices((10, 10, 10)).sum(axis=0) % 2
        subject = tio.Subject(
            image=tio.ScalarImage(
                tensor=torch.as_tensor(checkerboard[None], dtype=torch.float32),
                affine=np.eye(4),
            ),
        )

        without_antialias = tio.Resample(2, antialias=False)(subject)
        with_antialias = tio.Resample(2, antialias=True)(subject)

        self.assert_tensor_not_equal(
            without_antialias.image.data,
            with_antialias.image.data,
        )
