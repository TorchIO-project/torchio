import numpy as np
import pytest
import torch

import torchio as tio

from ...utils import TorchioTestCase


class TestResize(TorchioTestCase):
    """Tests for `Resize`."""

    def test_one_dim(self):
        target_shape = 5
        transform = tio.Resize(target_shape)
        transformed = transform(self.sample_subject)
        for image in transformed.get_images(intensity_only=False):
            assert image.spatial_shape == 3 * (target_shape,)

    def test_all_dims(self):
        target_shape = 11, 6, 7
        transform = tio.Resize(target_shape)
        transformed = transform(self.sample_subject)
        for image in transformed.get_images(intensity_only=False):
            assert image.spatial_shape == target_shape

    def test_negative_one_keeps_dimension(self):
        target_shape = -1, 5, 6
        transform = tio.Resize(target_shape)
        transformed = transform(self.sample_subject)
        for image in transformed.get_images(intensity_only=False):
            assert image.spatial_shape == (10, 5, 6)

    def test_crop_or_pad_fallback(self):
        """Resize should warn and fall back to CropOrPad for mismatched output shapes."""
        subject = tio.Subject(
            image=tio.ScalarImage(
                tensor=torch.rand(1, 10, 20, 30),
                affine=np.eye(4),
            ),
        )

        with pytest.warns(RuntimeWarning, match='Output shape'):
            transformed = tio.Resize((2, 2, 29))(subject)

        assert transformed.spatial_shape == (2, 2, 29)
