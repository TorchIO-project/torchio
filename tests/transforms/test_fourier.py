import builtins
from unittest import mock

import numpy as np
import torch

import torchio as tio


def _import_without_torch_fft(original_import):
    def fake_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name == 'torch.fft':
            raise ModuleNotFoundError(name)
        return original_import(name, globals, locals, fromlist, level)

    return fake_import


def test_fourier_transform_moves_dc_component_to_center():
    tensor = torch.ones(1, 4, 4, 4)

    transformed = tio.FourierTransform.fourier_transform(tensor)
    reconstructed = tio.FourierTransform.inv_fourier_transform(transformed)

    assert torch.is_complex(transformed)
    center_index = np.ravel_multi_index((0, 2, 2, 2), transformed.shape)
    assert torch.argmax(transformed.abs()).item() == center_index
    torch.testing.assert_close(reconstructed.real, tensor, atol=1e-6, rtol=0)
    torch.testing.assert_close(
        reconstructed.imag,
        torch.zeros_like(tensor),
        atol=1e-6,
        rtol=0,
    )


def test_fourier_transform_falls_back_to_numpy_when_torch_fft_is_missing():
    tensor = torch.arange(16, dtype=torch.float32).reshape(1, 2, 2, 4)
    expected = torch.from_numpy(np.fft.fftshift(np.fft.fftn(tensor.numpy())))
    fake_import = _import_without_torch_fft(builtins.__import__)

    with mock.patch.object(builtins, '__import__', side_effect=fake_import):
        result = tio.FourierTransform.fourier_transform(tensor)

    torch.testing.assert_close(result, expected, check_dtype=False)


def test_inverse_fourier_transform_falls_back_to_numpy_when_torch_fft_is_missing():
    image = np.arange(16, dtype=np.float32).reshape(1, 2, 2, 4)
    spectrum = np.fft.fftshift(np.fft.fftn(image))
    tensor = torch.from_numpy(spectrum)
    expected = torch.from_numpy(np.fft.ifftn(np.fft.ifftshift(spectrum)))
    fake_import = _import_without_torch_fft(builtins.__import__)

    with mock.patch.object(builtins, '__import__', side_effect=fake_import):
        result = tio.FourierTransform.inv_fourier_transform(tensor)

    torch.testing.assert_close(result, expected, check_dtype=False)
