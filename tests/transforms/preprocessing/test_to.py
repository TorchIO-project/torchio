import torch

import torchio as tio

from ...utils import TorchioTestCase


class TestTo(TorchioTestCase):
    """Tests for :class:`tio.To` class."""

    def test_to(self):
        transform = tio.To(torch.int)
        tensor = 10 * torch.rand(2, 3, 4, 5)
        image = tio.ScalarImage(tensor=tensor)
        transformed = transform(image)
        assert image.data.dtype == torch.float32
        assert transformed.data.dtype == torch.int
