from typing import Any
from typing import cast

import pytest
import torch

import torchio as tio

from ...utils import TorchioTestCase


class TestRandomBiasField(TorchioTestCase):
    def test_no_bias(self):
        transform = tio.RandomBiasField(coefficients=0)
        transformed = transform(self.sample_subject)
        self.assert_tensor_almost_equal(
            self.sample_subject.t1.data,
            transformed.t1.data,
        )

    def test_with_bias(self):
        transform = tio.RandomBiasField(coefficients=0.1)
        transformed = transform(self.sample_subject)
        self.assert_tensor_not_equal(
            self.sample_subject.t1.data,
            transformed.t1.data,
        )

    def test_wrong_coefficient_type(self):
        with pytest.raises(ValueError):
            tio.RandomBiasField(coefficients=cast(Any, 'wrong'))

    def test_negative_order(self):
        with pytest.raises(ValueError):
            tio.RandomBiasField(order=-1)

    def test_wrong_order_type(self):
        with pytest.raises(TypeError):
            tio.RandomBiasField(order=cast(Any, 'wrong'))

    def test_small_image(self):
        # https://github.com/TorchIO-project/torchio/issues/300
        tio.RandomBiasField()(torch.rand(1, 2, 3, 4))

    def test_no_images_returns_subject(self):
        """Applying to a subject with no scalar images returns it unchanged."""
        subject = tio.Subject(label=tio.LabelMap(tensor=torch.rand(1, 4, 4, 4)))
        transform = tio.RandomBiasField()
        result = transform(subject)
        assert result is not subject  # deepcopy still occurs

    def test_arguments_are_dict_mismatch_raises(self):
        """BiasField raises when only one of coefficients/order is a dict."""
        from torchio.transforms.augmentation.intensity.random_bias_field import (
            BiasField,
        )

        transform = BiasField(
            coefficients=cast(Any, {'t1': [0.1]}),
            order=3,
        )
        with pytest.raises(ValueError, match='all must be'):
            transform.arguments_are_dict()

    def test_arguments_are_dict_both_dicts(self):
        """BiasField.arguments_are_dict returns True when both are dicts."""
        from torchio.transforms.augmentation.intensity.random_bias_field import (
            BiasField,
        )

        transform = BiasField(
            coefficients=cast(Any, {'t1': [0.1]}),
            order=cast(Any, {'t1': 3}),
        )
        assert transform.arguments_are_dict() is True
