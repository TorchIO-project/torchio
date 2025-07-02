import pytest
import SimpleITK as sitk
import torch

import torchio as tio
from torchio.data.io import sitk_to_nib

from ...utils import TorchioTestCase


class TestPad(TorchioTestCase):
    """Tests for `Pad`."""

    def test_pad(self):
        image = self.sample_subject.t1
        padding = 1, 2, 3, 4, 5, 6
        sitk_image = image.as_sitk()
        low, high = padding[::2], padding[1::2]
        sitk_padded = sitk.ConstantPad(sitk_image, low, high, 0)
        tio_padded = tio.Pad(padding, padding_mode=0)(image)
        sitk_tensor, sitk_affine = sitk_to_nib(sitk_padded)
        tio_tensor, tio_affine = sitk_to_nib(tio_padded.as_sitk())
        self.assert_tensor_equal(sitk_tensor, tio_tensor)
        self.assert_tensor_equal(sitk_affine, tio_affine)

    def test_nans_history(self):
        padded = tio.Pad(1, padding_mode=2)(self.sample_subject)
        again = padded.history[0](self.sample_subject)
        assert not torch.isnan(again.t1.data).any()

    def test_padding_modes(self):
        def padding_func():
            return

        for padding_mode in [0, *tio.Pad.PADDING_MODES, padding_func]:
            tio.Pad(0, padding_mode=padding_mode)

        with self.assertRaises(KeyError):
            tio.Pad(0, padding_mode='abc')

    def test_padding_mean_label_map(self):
        with self.assertWarns(RuntimeWarning):
            tio.Pad(1, padding_mode='mean')(self.sample_subject.label)

    def test_padding_modes_global(self):
        x = torch.ones(1, 1, 2, 2, dtype=torch.int)
        x[..., 0, 0] = 0
        # The image should look like this:
        # 0 1
        # 1 1

        add_bottom_row = 0, 0, 0, 1, 0, 0
        with_zeros = tio.Pad(add_bottom_row)(x)
        assert with_zeros[0, 0, 2].tolist() == [0, 0]

        with_minimum = tio.Pad(add_bottom_row, padding_mode='minimum')(x)
        assert with_minimum[0, 0, 2].tolist() == [0, 0]

        with_maximum = tio.Pad(add_bottom_row, padding_mode='maximum')(x)
        assert with_maximum[0, 0, 2].tolist() == [1, 1]

        with_median = tio.Pad(add_bottom_row, padding_mode='median')(x)
        assert with_median[0, 0, 2].tolist() == [1, 1]

        # This is a special case: as we instantiated the tensor with integers,
        # the mean (3/4) will be trucated to 0.
        with_mean = tio.Pad(add_bottom_row, padding_mode='mean')(x)
        assert with_mean[0, 0, 2].tolist() == [0, 0]
        # So let's test with floats too
        x = x.float()
        with_mean = tio.Pad(add_bottom_row, padding_mode='mean')(x)
        assert with_mean[0, 0, 2].tolist() == [0.75, 0.75]

    def test_truncation_warning(self):
        x = torch.ones(1, 1, 2, 2, dtype=torch.int)
        pad = tio.Pad(1, padding_mode='mean')
        with pytest.warns(RuntimeWarning):
            pad(x)
