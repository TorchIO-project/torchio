import copy

import numpy as np
import torch

import torchio as tio

from ...utils import TorchioTestCase


class TestPCA(TorchioTestCase):
    """Tests for `PCA` transform."""

    def _make_subject(self, channels=16, size=5):
        """Create a subject with a multi-channel embedding image."""
        torch.manual_seed(42)
        tensor = torch.randn(channels, size, size, size)
        image = tio.ScalarImage(tensor=tensor)
        return tio.Subject(emb=image)

    def test_pca_default_output_clipped(self):
        """Default PCA produces 3 components with values clipped to [0, 1]."""
        subject = self._make_subject()
        transformed = tio.PCA()(subject)
        data = transformed['emb'].data
        assert data.shape[0] == 3
        assert data.min() >= 0.0
        assert data.max() <= 1.0

    def test_pca_preserves_spatial_shape_and_affine(self):
        """PCA preserves the spatial shape and affine of the input."""
        subject = self._make_subject()
        original_affine = subject['emb'].affine.copy()
        transformed = tio.PCA()(subject)
        assert transformed['emb'].spatial_shape == (5, 5, 5)
        np.testing.assert_array_equal(transformed['emb'].affine, original_affine)

    def test_pca_custom_components(self):
        """PCA with num_components=5 produces exactly 5 channels."""
        subject = self._make_subject(channels=32)
        transformed = tio.PCA(num_components=5)(subject)
        assert transformed['emb'].data.shape[0] == 5

    def test_pca_no_normalize_changes_scale(self):
        """Without normalize, component values are not divided by first std."""
        subject = self._make_subject()
        norm = tio.PCA(normalize=True, clip=False)(copy.deepcopy(subject))
        no_norm = tio.PCA(normalize=False, clip=False)(copy.deepcopy(subject))
        # Normalization divides by std of first component, so scales differ
        assert not torch.allclose(norm['emb'].data, no_norm['emb'].data)

    def test_pca_skewness_correction_flips_components(self):
        """With make_skewness_positive, each component has non-negative skewness."""
        subject = self._make_subject(channels=32, size=8)
        transformed = tio.PCA(
            make_skewness_positive=True, clip=False, values_range=None
        )(subject)
        data = transformed['emb'].numpy()
        for i in range(data.shape[0]):
            component = data[i].ravel()
            third_moment = np.mean(component**3)
            second_moment = np.mean(component**2)
            skewness = third_moment / second_moment ** (3 / 2)
            assert skewness >= -1e-6, f'Component {i} has negative skewness'

    def test_pca_no_clip_allows_out_of_range(self):
        """Without clipping, values can exceed [0, 1]."""
        subject = self._make_subject()
        transformed = tio.PCA(clip=False)(subject)
        data = transformed['emb'].data
        # With randn input and default values_range=(-2.3, 2.3),
        # some values are likely outside [0, 1]
        assert data.min() < 0.0 or data.max() > 1.0

    def test_pca_no_values_range_uses_data_extremes(self):
        """With values_range=None, the output spans [0, 1] from data min/max."""
        subject = self._make_subject()
        transformed = tio.PCA(values_range=None, clip=False)(subject)
        data = transformed['emb'].numpy()
        # Each component is normalized by its own min/max across all components
        assert data.min() >= -1e-6
        assert data.max() <= 1 + 1e-6

    def test_pca_reproducible_with_random_state(self):
        """Passing random_state via pca_kwargs produces deterministic output."""
        subject = self._make_subject()
        t1 = tio.PCA(pca_kwargs={'random_state': 0})(copy.deepcopy(subject))
        t2 = tio.PCA(pca_kwargs={'random_state': 0})(copy.deepcopy(subject))
        torch.testing.assert_close(t1['emb'].data, t2['emb'].data)
