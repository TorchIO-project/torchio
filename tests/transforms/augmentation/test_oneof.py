from typing import Any
from typing import cast
from unittest.mock import patch

import pytest
import torch

import torchio as tio

from ...utils import TorchioTestCase


class ReturningTransform(tio.Transform):
    def __init__(self, output: str):
        super().__init__(parse_input=False)
        self.output = output
        self.args_names = ['output']

    def apply_transform(self, subject):
        return self.output


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

    def test_sequence_input(self):
        first = tio.RandomAffine()
        second = tio.RandomElasticDeformation()

        transform = tio.OneOf([first, second])

        assert list(transform.transforms_dict) == [first, second]
        assert list(transform.transforms_dict.values()) == [0.5, 0.5]

    def test_probabilities_are_normalized(self):
        first = tio.RandomAffine()
        second = tio.RandomElasticDeformation()
        transforms: dict[tio.Transform, float] = {
            first: 1,
            second: 3,
        }

        transform = tio.OneOf(transforms)

        assert list(transform.transforms_dict.values()) == pytest.approx([0.25, 0.75])

    def test_some_zero_probabilities(self):
        first = tio.RandomAffine()
        second = tio.RandomElasticDeformation()
        transforms: dict[tio.Transform, float] = {
            first: 0,
            second: 2,
        }

        transform = tio.OneOf(transforms)

        assert list(transform.transforms_dict.values()) == pytest.approx([0, 1])

    def test_selected_transform_is_applied(self):
        """Patch sampling to verify the transform chosen by `OneOf` is executed."""
        transform = tio.OneOf(
            [ReturningTransform('first'), ReturningTransform('second')]
        )

        with patch(
            'torchio.transforms.augmentation.composition.torch.multinomial',
            return_value=torch.tensor(1),
        ):
            transformed = transform.apply_transform(cast(Any, object()))

        assert transformed == 'second'

    def test_get_base_args(self):
        transform = tio.OneOf([tio.RandomNoise()])

        base_args = transform._get_base_args()

        assert 'parse_input' not in base_args
