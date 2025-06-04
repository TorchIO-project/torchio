import torch

import torchio as tio

from ...utils import TorchioTestCase


class TestCrop(TorchioTestCase):
    def test_tensor_single_channel(self):
        crop = tio.Crop(1)
        assert crop(torch.rand(1, 10, 10, 10)).shape == (1, 8, 8, 8)

    def test_tensor_multi_channel(self):
        crop = tio.Crop(1)
        assert crop(torch.rand(3, 10, 10, 10)).shape == (3, 8, 8, 8)

    def test_subject_copy(self):
        crop = tio.Crop(1, copy=True)
        subject = tio.Subject(t1=tio.ScalarImage(tensor=torch.rand(1, 10, 10, 10)))
        cropped_subject = crop(subject)
        assert cropped_subject.t1.shape == (1, 8, 8, 8)
        assert subject.t1.shape == (1, 10, 10, 10)

    def test_subject_no_copy(self):
        crop = tio.Crop(1, copy=False)
        subject = tio.Subject(t1=tio.ScalarImage(tensor=torch.rand(1, 10, 10, 10)))
        cropped_subject = crop(subject)
        assert cropped_subject.t1.shape == (1, 8, 8, 8)
        assert subject.t1.shape == (1, 8, 8, 8)
