import SimpleITK as sitk

import torchio as tio

from ...utils import TorchioTestCase


class TestTranspose(TorchioTestCase):
    def test_transpose(self):
        transform = tio.Transpose()
        image = tio.ScalarImage(self.get_image_path('image'))
        transformed = transform(image)
        sitk_image = sitk.GetImageFromArray(image.numpy()[0])
        from_sitk = tio.ScalarImage.from_sitk(sitk_image)
        self.assert_tensor_equal(transformed.data, from_sitk.data)

    def test_orientation_reversed(self):
        transform = tio.Transpose()
        image = tio.ScalarImage(self.get_image_path('image'))
        transformed = transform(image)
        self.assertEqual(transformed.orientation_str, image.orientation_str[::-1])

    def test_is_invertible(self):
        """Transpose.is_invertible() returns True."""
        transform = tio.Transpose()
        assert transform.is_invertible()

    def test_inverse_is_self(self):
        """Transpose.inverse() returns self (it is its own inverse)."""
        transform = tio.Transpose()
        inverse = transform.inverse()
        assert inverse is transform
