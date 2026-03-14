from typing import Any
from typing import cast

import pytest

import torchio as tio

from ...utils import TorchioTestCase


class TestOneOf(TorchioTestCase):
    """Tests for `OneOf`."""

    def test_wrong_input_type(self):
        with pytest.raises(ValueError):
            tio.OneOf(cast(Any, 1))

    def test_negative_probabilities(self):
        transforms: dict[tio.Transform, float] = {
            tio.RandomAffine(): -1,
            tio.RandomElasticDeformation(): 1,
        }
        with pytest.raises(ValueError):
            tio.OneOf(transforms)

    def test_zero_probabilities(self):
        with pytest.raises(ValueError):
            transforms: dict[tio.Transform, float] = {
                tio.RandomAffine(): 0,
                tio.RandomElasticDeformation(): 0,
            }
            tio.OneOf(transforms)

    def test_not_transform(self):
        with pytest.raises(ValueError):
            tio.OneOf(cast(Any, {tio.RandomAffine: 1, tio.RandomElasticDeformation: 2}))

    def test_one_of(self):
        transforms: dict[tio.Transform, float] = {
            tio.RandomAffine(): 0.2,
            tio.RandomElasticDeformation(max_displacement=0.5): 0.8,
        }
        transform = tio.OneOf(transforms)
        transform(self.sample_subject)
