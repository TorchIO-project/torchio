import numpy as np
import SimpleITK as sitk
import torch

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

    def test_inverse(self):
        transform = tio.Transpose()
        image = tio.ScalarImage(
            tensor=torch.rand(1, 2, 3, 4),
            affine=np.diag((1.0, 2.0, 3.0, 1.0)),
        )

        transformed = transform(image)
        restored = transform.inverse()(transformed)

        assert transform.is_invertible()
        assert transform.inverse() is transform
        assert transformed.spacing == (3.0, 2.0, 1.0)
        self.assert_tensor_equal(restored.data, image.data)
        self.assert_tensor_almost_equal(restored.affine, image.affine)
