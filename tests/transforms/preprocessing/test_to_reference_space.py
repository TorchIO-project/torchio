from typing import Any
from typing import cast

import pytest
import torch

import torchio as tio
from torchio.transforms.preprocessing.spatial.to_reference_space import ToReferenceSpace
from torchio.transforms.preprocessing.spatial.to_reference_space import (
    build_image_from_reference,
)

from ...utils import TorchioTestCase


class TestToReferenceSpace(TorchioTestCase):
    """Tests for `ToReferenceSpace`."""

    def test_non_image_reference_raises(self):
        """Passing a non-Image reference should raise TypeError."""
        with pytest.raises(TypeError, match='must be a TorchIO image'):
            ToReferenceSpace(reference=cast(Any, 'not_an_image'))

    def test_apply_transform(self):
        """apply_transform updates image data and affine from reference."""
        reference = tio.ScalarImage(tensor=torch.rand(1, 10, 20, 30))
        embedding = tio.ScalarImage(tensor=torch.rand(1, 5, 10, 15))
        subject = tio.Subject(emb=embedding)
        transform = ToReferenceSpace(reference=reference)
        transformed = transform(subject)
        assert transformed['emb'].spatial_shape == (5, 10, 15)

    def test_from_tensor(self):
        """from_tensor builds a TorchIO image from a tensor and reference."""
        reference = tio.ScalarImage(tensor=torch.rand(1, 10, 20, 30))
        tensor = torch.rand(8, 5, 10, 15)
        result = ToReferenceSpace.from_tensor(tensor, reference)
        assert isinstance(result, tio.Image)
        assert result.spatial_shape == (5, 10, 15)

    def test_build_image_from_reference(self):
        """build_image_from_reference preserves tensor shape and sets affine."""
        reference = tio.ScalarImage(tensor=torch.rand(1, 20, 20, 20))
        tensor = torch.rand(3, 10, 10, 10)
        result = build_image_from_reference(tensor, reference)
        assert result.data.shape == tensor.shape
