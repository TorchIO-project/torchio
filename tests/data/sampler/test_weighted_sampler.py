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

    def test_missing_probability_map_raises(self):
        """Missing probability map key raises KeyError."""
        subject = self.get_sample((1, 7, 7, 7))
        sampler = WeightedSampler(5, 'nonexistent')
        with pytest.raises(KeyError, match='nonexistent'):
            next(sampler(subject))

    def test_negative_probability_map_raises(self):
        """Negative values in probability map raise ValueError."""
        t1 = torch.rand(1, 7, 7, 7)
        prob = torch.zeros_like(t1)
        prob[0, 3, 3, 3] = -1.0
        subject = tio.Subject(
            t1=tio.ScalarImage(tensor=t1),
            prob=tio.ScalarImage(tensor=prob),
        )
        subject = tio.SubjectsDataset([subject])[0]
        sampler = WeightedSampler(5, 'prob')
        with pytest.raises(ValueError, match='Negative values'):
            next(sampler(subject))

    def test_extract_patch_without_cdf_numpy(self):
        """extract_patch with numpy array index and no CDF works."""
        subject = self.get_sample((1, 7, 7, 7))
        sampler = WeightedSampler(5, 'prob')
        index = np.array([1, 1, 1])
        patch = sampler.extract_patch(subject, index, cdf=None)
        assert isinstance(patch, tio.Subject)

    def test_extract_patch_with_cdf_non_array_raises(self):
        """extract_patch with CDF but tuple index raises TypeError."""
        subject = self.get_sample((1, 7, 7, 7))
        sampler = WeightedSampler(5, 'prob')
        cdf = np.array([0.5, 1.0])
        with pytest.raises(TypeError, match='NumPy array'):
            sampler.extract_patch(subject, (1, 1, 1), cdf=cdf)
