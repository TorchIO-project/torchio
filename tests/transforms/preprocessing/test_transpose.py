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
