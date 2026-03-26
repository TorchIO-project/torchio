from typing import Any
from typing import cast

import pytest
import torch

import torchio as tio
from torchio import RandomNoise

from ...utils import TorchioTestCase


class TestRandomNoise(TorchioTestCase):
    """Tests for `RandomNoise`."""

    def test_no_noise(self):
        transform = RandomNoise(mean=0, std=0)
        transformed = transform(self.sample_subject)
        self.assert_tensor_almost_equal(
            self.sample_subject.t1.data,
            transformed.t1.data,
        )

    def test_with_noise(self):
        transform = RandomNoise()
        transformed = transform(self.sample_subject)
        self.assert_tensor_not_equal(
            self.sample_subject.t1.data,
            transformed.t1.data,
        )

    def test_constant_noise(self):
        transform = RandomNoise(mean=(5, 5), std=0)
        transformed = transform(self.sample_subject)
        self.assert_tensor_almost_equal(
            self.sample_subject.t1.data + 5,
            transformed.t1.data,
        )

    def test_negative_std(self):
        with pytest.raises(ValueError):
            RandomNoise(std=-2)

    def test_std_range_with_negative_min(self):
        with pytest.raises(ValueError):
            RandomNoise(std=(-0.5, 4))

    def test_wrong_std_type(self):
        with pytest.raises(ValueError):
            RandomNoise(std=cast(Any, 'wrong'))

    def test_wrong_mean_type(self):
        with pytest.raises(ValueError):
            RandomNoise(mean=cast(Any, 'wrong'))

    def test_no_images_returns_subject(self):
        """Applying to subject with no scalar images returns unchanged."""
        subject = tio.Subject(
            label=tio.LabelMap(tensor=torch.rand(1, 4, 4, 4)),
        )
        RandomNoise()(subject)
