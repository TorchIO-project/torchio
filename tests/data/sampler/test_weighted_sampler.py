from unittest.mock import patch

import numpy as np
import pytest
import torch

import torchio as tio
from torchio.data import WeightedSampler

from ...utils import TorchioTestCase


class TestWeightedSampler(TorchioTestCase):
    """Tests for `WeightedSampler` class."""

    def test_weighted_sampler(self):
        subject = self.get_sample((1, 7, 7, 7))
        sampler = WeightedSampler(5, 'prob')
        patch = next(sampler(subject))
        location = patch[tio.LOCATION]
        assert isinstance(location, torch.Tensor)
        assert tuple(location[:3].tolist()) == (1, 1, 1)

    def get_sample(self, image_shape):
        t1 = torch.rand(*image_shape)
        prob = torch.zeros_like(t1)
        prob[0, 3, 3, 3] = 1
        subject = tio.Subject(
            t1=tio.ScalarImage(tensor=t1),
            prob=tio.ScalarImage(tensor=prob),
        )
        subject = tio.SubjectsDataset([subject])[0]
        return subject

    def test_inconsistent_shape(self):
        # https://github.com/TorchIO-project/torchio/issues/234#issuecomment-675029767
        subject = tio.Subject(
            im1=tio.ScalarImage(tensor=torch.rand(1, 4, 5, 6)),
            im2=tio.ScalarImage(tensor=torch.rand(2, 4, 5, 6)),
        )
        patch_size = 2
        sampler = tio.data.WeightedSampler(patch_size, 'im1')
        next(sampler(subject))

    def test_missing_probability_map(self):
        subject = self.get_sample((1, 7, 7, 7))
        sampler = WeightedSampler(5, 'missing')
        with pytest.raises(KeyError):
            sampler.get_probability_map_image(subject)

    def test_negative_probability_map(self):
        subject = self.get_sample((1, 7, 7, 7))
        subject.prob.set_data(subject.prob.data - 2)
        sampler = WeightedSampler(5, 'prob')
        with pytest.raises(ValueError, match='Negative values found'):
            sampler.get_probability_map(subject)

    def test_extract_patch_without_cdf(self):
        subject = self.get_sample((1, 7, 7, 7))
        sampler = WeightedSampler(5, 'prob')
        patch = sampler.extract_patch(subject, np.array((1, 1, 1)))
        assert patch.spatial_shape == (5, 5, 5)

    def test_extract_patch_tuple_without_cdf(self):
        subject = self.get_sample((1, 7, 7, 7))
        sampler = WeightedSampler(5, 'prob')
        patch = sampler.extract_patch(subject, (1, 1, 1))
        assert patch.spatial_shape == (5, 5, 5)

    def test_extract_patch_invalid_type_with_cdf(self):
        subject = self.get_sample((1, 7, 7, 7))
        sampler = WeightedSampler(5, 'prob')
        with pytest.raises(TypeError, match='Probability map must be a NumPy array'):
            sampler.extract_patch(subject, (1, 1, 1), cdf=np.array((1.0,)))

    def test_probability_zero_raises(self):
        probability_map = np.array(((0, 1), (1, 1)), dtype=np.float32)
        cdf = WeightedSampler.get_cumulative_distribution_function(probability_map)

        with (
            patch('torchio.data.sampler.weighted.np.searchsorted', return_value=0),
            pytest.raises(RuntimeError, match='Error retrieving probability'),
        ):
            WeightedSampler.sample_probability_map(probability_map, cdf)
