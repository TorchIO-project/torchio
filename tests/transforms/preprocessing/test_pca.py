import torch

import torchio as tio

from ...utils import TorchioTestCase


class TestPCA(TorchioTestCase):
    """Tests for `PCA` transform."""

    def test_pca_default(self):
        """PCA with defaults reduces channels to 3 components."""
        tensor = torch.randn(16, 5, 5, 5)
        image = tio.ScalarImage(tensor=tensor)
        subject = tio.Subject(emb=image)
        transform = tio.PCA()
        transformed = transform(subject)
        assert transformed['emb'].data.shape[0] == 3
        assert transformed['emb'].spatial_shape == (5, 5, 5)

    def test_pca_custom_components(self):
        """PCA with custom num_components reduces to that many channels."""
        tensor = torch.randn(32, 4, 4, 4)
        image = tio.ScalarImage(tensor=tensor)
        subject = tio.Subject(emb=image)
        transform = tio.PCA(num_components=5)
        transformed = transform(subject)
        assert transformed['emb'].data.shape[0] == 5

    def test_pca_no_normalize(self):
        """PCA with normalize=False skips std-normalization."""
        tensor = torch.randn(16, 4, 4, 4)
        image = tio.ScalarImage(tensor=tensor)
        subject = tio.Subject(emb=image)
        transform = tio.PCA(normalize=False)
        transformed = transform(subject)
        assert transformed['emb'].data.shape[0] == 3

    def test_pca_no_skewness_correction(self):
        """PCA with make_skewness_positive=False skips sign flipping."""
        tensor = torch.randn(16, 4, 4, 4)
        image = tio.ScalarImage(tensor=tensor)
        subject = tio.Subject(emb=image)
        transform = tio.PCA(make_skewness_positive=False)
        transformed = transform(subject)
        assert transformed['emb'].data.shape[0] == 3

    def test_pca_no_values_range(self):
        """PCA with values_range=None uses data min/max for normalization."""
        tensor = torch.randn(16, 4, 4, 4)
        image = tio.ScalarImage(tensor=tensor)
        subject = tio.Subject(emb=image)
        transform = tio.PCA(values_range=None, clip=False)
        transformed = transform(subject)
        assert transformed['emb'].data.shape[0] == 3

    def test_pca_no_clip(self):
        """PCA with clip=False allows values outside [0, 1]."""
        tensor = torch.randn(16, 4, 4, 4)
        image = tio.ScalarImage(tensor=tensor)
        subject = tio.Subject(emb=image)
        transform = tio.PCA(clip=False)
        transformed = transform(subject)
        assert transformed['emb'].data.shape[0] == 3

    def test_pca_with_kwargs(self):
        """PCA with custom pca_kwargs forwards them to sklearn PCA."""
        tensor = torch.randn(16, 4, 4, 4)
        image = tio.ScalarImage(tensor=tensor)
        subject = tio.Subject(emb=image)
        transform = tio.PCA(pca_kwargs={'random_state': 42})
        transformed = transform(subject)
        assert transformed['emb'].data.shape[0] == 3
