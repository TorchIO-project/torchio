from typing import Any
from typing import cast

import pytest
import torch

import torchio as tio
from torchio import RandomGhosting
from torchio.transforms.augmentation.intensity.random_ghosting import Ghosting

from ...utils import TorchioTestCase


class TestRandomGhosting(TorchioTestCase):
    """Tests for `RandomGhosting`."""

    def test_with_zero_intensity(self):
        transform = RandomGhosting(intensity=0)
        transformed = transform(self.sample_subject)
        self.assert_tensor_almost_equal(
            self.sample_subject.t1.data,
            transformed.t1.data,
        )

    def test_with_zero_ghost(self):
        transform = RandomGhosting(num_ghosts=0)
        transformed = transform(self.sample_subject)
        self.assert_tensor_almost_equal(
            self.sample_subject.t1.data,
            transformed.t1.data,
        )

    def test_with_ghosting(self):
        transform = RandomGhosting()
        transformed = transform(self.sample_subject)
        self.assert_tensor_not_equal(
            self.sample_subject.t1.data,
            transformed.t1.data,
        )

    def test_intensity_range_with_negative_min(self):
        with pytest.raises(ValueError):
            RandomGhosting(intensity=(-0.5, 4))

    def test_wrong_intensity_type(self):
        with pytest.raises(ValueError):
            RandomGhosting(intensity=cast(Any, 'wrong'))

    def test_negative_num_ghosts(self):
        with pytest.raises(ValueError):
            RandomGhosting(num_ghosts=-1)

    def test_num_ghosts_range_with_negative_min(self):
        with pytest.raises(ValueError):
            RandomGhosting(num_ghosts=(-1, 4))

    def test_not_integer_num_ghosts(self):
        with pytest.raises(ValueError):
            RandomGhosting(num_ghosts=cast(Any, (0.7, 4)))

    def test_wrong_num_ghosts_type(self):
        with pytest.raises(ValueError):
            RandomGhosting(num_ghosts=cast(Any, 'wrong'))

    def test_out_of_range_axis(self):
        with pytest.raises(ValueError):
            RandomGhosting(axes=3)

    def test_out_of_range_axis_in_tuple(self):
        with pytest.raises(ValueError):
            RandomGhosting(axes=(0, -1, 2))

    def test_wrong_axes_type(self):
        with pytest.raises(ValueError):
            RandomGhosting(axes=cast(Any, None))

    def test_out_of_range_restore(self):
        with pytest.raises(ValueError):
            RandomGhosting(restore=-1)

    def test_wrong_restore_type(self):
        with pytest.raises(ValueError):
            RandomGhosting(restore=cast(Any, 'wrong'))

    def test_no_images_returns_subject(self):
        """Applying to subject with no scalar images returns unchanged."""
        subject = tio.Subject(
            label=tio.LabelMap(tensor=torch.rand(1, 4, 4, 4)),
        )
        transform = RandomGhosting()
        transform(subject)

    def test_string_axes_checks_orientation(self):
        """String axes trigger consistent orientation check."""
        transform = RandomGhosting(axes=(0, 'LR'))
        transform(self.sample_subject)

    def test_restore_range(self):
        """Non-None restore parameter samples a restore center value."""
        transform = RandomGhosting(restore=0.5)
        transform(self.sample_subject)

    def test_get_slices_to_restore_with_center(self):
        """_get_slices_to_restore computes a wider slice when given center."""
        spectrum = torch.randn(1, 32, 32, 32)
        _, slices = Ghosting._get_slices_to_restore(
            spectrum,
            axis=1,
            restore_center=0.5,
        )
        # With restore_center=0.5, the slice should span more than 1 voxel
        restored = spectrum[slices]
        assert restored.shape[1] > 1
