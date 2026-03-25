from typing import Any
from typing import cast

import pytest
import torch

import torchio as tio
from torchio.transforms.data_parser import DataParser

from ..utils import TorchioTestCase


class TestDataParser(TorchioTestCase):
    """Tests for DataParser edge cases."""

    def test_unrecognized_input_type_raises(self):
        """Passing an unsupported type should raise ValueError."""
        parser = DataParser(data=cast(Any, 12345))
        with pytest.raises(ValueError, match='Input type not recognized'):
            parser.get_subject()

    def test_dict_with_non_tensor_image_value_raises(self):
        """Dict values selected as images must be tensors or arrays."""
        parser = DataParser(
            data={'image': 'not_a_tensor', 'other': 42},
            keys=['image'],
        )
        with pytest.raises(TypeError, match='tensors or arrays'):
            parser.get_subject()

    def test_dict_with_valid_tensor_value(self):
        """Dict with tensor values should parse correctly."""
        tensor = torch.randn(1, 10, 10, 10)
        parser = DataParser(
            data={'image': tensor, 'metadata': 'info'},
            keys=['image'],
        )
        subject = parser.get_subject()
        assert 'image' in subject
        assert 'metadata' in subject

    def test_dict_with_label_keys(self):
        """Dict with label_keys should create LabelMap images."""
        tensor = torch.randint(0, 3, (1, 10, 10, 10)).float()
        parser = DataParser(
            data={'seg': tensor},
            keys=['seg'],
            label_keys=['seg'],
        )
        subject = parser.get_subject()
        assert isinstance(subject['seg'], tio.LabelMap)
