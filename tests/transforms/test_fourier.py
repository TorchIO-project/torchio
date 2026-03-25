from unittest.mock import patch

import torch

from torchio.transforms.fourier import FourierTransform

from ..utils import TorchioTestCase


class TestFourierTransform(TorchioTestCase):
    """Tests for the NumPy FFT fallback in FourierTransform."""

    def test_fourier_transform_numpy_fallback(self):
        """Ensure forward FFT works via NumPy when torch.fft raises."""
        tensor = torch.randn(4, 4, 4)
        with patch('torch.fft.fftn', side_effect=ModuleNotFoundError):
            result = FourierTransform.fourier_transform(tensor)
        assert isinstance(result, torch.Tensor)
        assert result.shape == tensor.shape

    def test_inv_fourier_transform_numpy_fallback(self):
        """Ensure inverse FFT works via NumPy when torch.fft raises."""
        tensor = torch.randn(4, 4, 4).to(torch.complex64)
        with patch('torch.fft.ifftshift', side_effect=ModuleNotFoundError):
            result = FourierTransform.inv_fourier_transform(tensor)
        assert isinstance(result, torch.Tensor)
        assert result.shape == tensor.shape

    def test_fourier_roundtrip(self):
        """Verify that forward then inverse FFT approximately recovers input."""
        tensor = torch.randn(4, 4, 4)
        freq = FourierTransform.fourier_transform(tensor)
        recovered = FourierTransform.inv_fourier_transform(freq)
        torch.testing.assert_close(
            recovered.real.float(),
            tensor,
            atol=1e-5,
            rtol=1e-5,
        )
