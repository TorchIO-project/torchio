from typing import Any
from typing import cast

import pytest
import torch

import torchio as tio
from torchio import RandomGamma

from ...utils import TorchioTestCase


class TestRandomGamma(TorchioTestCase):
    """Tests for `RandomGamma`."""

    def get_random_tensor_zero_one(self):
        return torch.rand(4, 5, 6, 7)

    def test_with_zero_gamma(self):
        transform = RandomGamma(log_gamma=0)
        tensor = self.get_random_tensor_zero_one()
        transformed = transform(tensor)
        self.assert_tensor_almost_equal(tensor, transformed)

    def test_with_non_zero_gamma(self):
        transform = RandomGamma(log_gamma=(0.1, 0.3))
        tensor = self.get_random_tensor_zero_one()
        transformed = transform(tensor)
        self.assert_tensor_not_equal(tensor, transformed)

    def test_with_high_gamma(self):
        transform = RandomGamma(log_gamma=(100, 100))
        tensor = self.get_random_tensor_zero_one()
        transformed = transform(tensor)
        self.assert_tensor_almost_equal(
            tensor == 1,
            transformed,
        )

    def test_with_low_gamma(self):
        transform = RandomGamma(log_gamma=(-100, -100))
        tensor = self.get_random_tensor_zero_one()
        transformed = transform(tensor)
        self.assert_tensor_almost_equal(
            tensor > 0,
            transformed,
        )

    def test_wrong_gamma_type(self):
        with pytest.raises(ValueError):
            RandomGamma(log_gamma=cast(Any, 'wrong'))

    def test_no_images_returns_subject(self):
        """Applying to subject with no scalar images returns unchanged."""
        subject = tio.Subject(
            label=tio.LabelMap(tensor=torch.rand(1, 4, 4, 4)),
        )
        RandomGamma()(subject)
