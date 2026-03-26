from typing import Any
from typing import cast

import pytest
import torch

import torchio as tio

from ...utils import TorchioTestCase


class TestRandomSwap(TorchioTestCase):
    def test_no_swap(self):
        transform = tio.RandomSwap(patch_size=5, num_iterations=0)
        transformed = transform(self.sample_subject)
        self.assert_tensor_equal(
            self.sample_subject.t1.data,
            transformed.t1.data,
        )

    def test_with_swap(self):
        transform = tio.RandomSwap(patch_size=5)
        transformed = transform(self.sample_subject)
        self.assert_tensor_not_equal(
            self.sample_subject.t1.data,
            transformed.t1.data,
        )

    def test_wrong_num_iterations_type(self):
        with pytest.raises(TypeError):
            tio.RandomSwap(num_iterations=cast(Any, 'wrong'))

    def test_negative_num_iterations(self):
        with pytest.raises(ValueError):
            tio.RandomSwap(num_iterations=-1)

    def test_no_images_returns_subject(self):
        """Applying to subject with no scalar images returns unchanged."""
        subject = tio.Subject(
            label=tio.LabelMap(tensor=torch.rand(1, 4, 4, 4)),
        )
        tio.RandomSwap()(subject)

    def test_patch_larger_than_image_raises(self):
        """Patch size larger than image spatial shape raises ValueError."""
        image = tio.ScalarImage(tensor=torch.rand(1, 5, 5, 5))
        subject = tio.Subject(t1=image)
        transform = tio.RandomSwap(patch_size=10)
        with pytest.raises(ValueError, match='cannot be.*larger'):
            transform(subject)
