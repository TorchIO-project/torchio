from typing import Any
from typing import cast

import pytest
import torch

import torchio as tio
from torchio import RandomAnisotropy
from torchio import ScalarImage

from ...utils import TorchioTestCase


class TestRandomAnisotropy(TorchioTestCase):
    """Tests for `RandomAnisotropy`."""

    def test_downsample(self):
        transform = RandomAnisotropy(
            axes=1,
            downsampling=(2, 2),
        )
        transformed = transform(self.sample_subject)
        assert self.sample_subject.spacing[1] == transformed.spacing[1]

    def test_out_of_range_axis(self):
        with pytest.raises(ValueError):
            RandomAnisotropy(axes=3)

    def test_out_of_range_axis_in_tuple(self):
        with pytest.raises(ValueError):
            RandomAnisotropy(axes=(0, -1, 2))

    def test_wrong_axes_type(self):
        with pytest.raises(ValueError):
            RandomAnisotropy(axes=cast(Any, 'wrong'))

    def test_wrong_downsampling_type(self):
        with pytest.raises(ValueError):
            RandomAnisotropy(downsampling=cast(Any, 'wrong'))

    def test_below_one_downsampling(self):
        with pytest.raises(ValueError):
            RandomAnisotropy(downsampling=0.2)

    def test_2d_rgb(self):
        image = ScalarImage(tensor=torch.rand(3, 4, 5, 6))
        RandomAnisotropy()(image)

    def test_copy_false_preserves_shape(self):
        """Output shape and spacing must match input when copy=False (#1436)."""
        subject = tio.Subject(
            t1=tio.ScalarImage(tensor=torch.randn(1, 20, 22, 18)),
        )
        transform = RandomAnisotropy(
            axes=1,
            downsampling=(2, 2),
            copy=False,
        )
        original_shape = subject.shape
        original_spacing = subject.spacing
        result = transform(subject)
        assert result.shape == original_shape
        assert result.spacing == original_spacing

    def test_2d_with_axis_2_warns(self):
        """Applying to 2D image with axis 2 in axes warns and excludes it."""
        image = ScalarImage(tensor=torch.rand(1, 10, 10, 1))
        subject = tio.Subject(t1=image)
        transform = RandomAnisotropy(axes=(0, 1, 2))
        with pytest.warns(RuntimeWarning, match='2D'):
            transform(subject)
