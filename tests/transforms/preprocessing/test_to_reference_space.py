import copy

import numpy as np
import pytest
import torch

import torchio as tio

from ...utils import TorchioTestCase


def get_reference_image(
    shape=(10, 20, 30),
    spacing=(1.0, 2.0, 3.0),
    image_class=tio.ScalarImage,
):
    tensor = torch.zeros(
        1,
        *shape,
        dtype=torch.uint8 if image_class is tio.LabelMap else torch.float32,
    )
    affine = np.diag([*spacing, 1.0])
    return image_class(tensor=tensor, affine=affine)


class TestToReferenceSpace(TorchioTestCase):
    def test_from_tensor(self):
        reference = get_reference_image()
        tensor = torch.arange(2 * 5 * 10 * 15, dtype=torch.float32).reshape(
            2,
            5,
            10,
            15,
        )

        image = tio.ToReferenceSpace.from_tensor(tensor, reference)

        assert isinstance(image, tio.ScalarImage)
        self.assert_tensor_equal(image.data, tensor)
        assert image.spatial_shape == (5, 10, 15)
        assert image.spacing == (2.0, 4.0, 6.0)
        self.assert_tensor_almost_equal(
            image.affine,
            np.array(
                [
                    [2.0, 0.0, 0.0, 0.5],
                    [0.0, 4.0, 0.0, 1.0],
                    [0.0, 0.0, 6.0, 1.5],
                    [0.0, 0.0, 0.0, 1.0],
                ]
            ),
        )

    def test_apply_transform(self):
        reference = get_reference_image()
        tensor = torch.arange(2 * 5 * 10 * 15, dtype=torch.float32).reshape(
            2,
            5,
            10,
            15,
        )
        subject = tio.Subject(
            embedding=tio.ScalarImage(tensor=tensor, affine=np.eye(4)),
        )

        transformed = tio.ToReferenceSpace(reference)(subject)
        expected = tio.ToReferenceSpace.from_tensor(tensor, reference)

        self.assert_tensor_equal(transformed.embedding.data, expected.data)
        self.assert_tensor_almost_equal(
            transformed.embedding.affine,
            expected.affine,
        )

    def test_include(self):
        reference = get_reference_image()
        tensor = torch.arange(2 * 5 * 10 * 15, dtype=torch.float32).reshape(
            2,
            5,
            10,
            15,
        )
        label = torch.ones(1, 5, 10, 15, dtype=torch.uint8)
        subject = tio.Subject(
            t1=tio.ScalarImage(tensor=tensor, affine=np.eye(4)),
            label=tio.LabelMap(tensor=label, affine=np.eye(4)),
        )
        original = copy.deepcopy(subject)

        transformed = tio.ToReferenceSpace(reference, include=['label'])(subject)

        self.assert_tensor_equal(transformed.t1.data, original.t1.data)
        self.assert_tensor_almost_equal(transformed.t1.affine, original.t1.affine)
        self.assert_tensor_equal(transformed.label.data, original.label.data)
        assert transformed.label.spacing == (2.0, 4.0, 6.0)

    def test_reference_class_is_preserved(self):
        """from_tensor should preserve the reference image class when rebuilding data."""
        reference = get_reference_image(image_class=tio.LabelMap)
        tensor = torch.zeros(2, 5, 10, 15, dtype=torch.uint8)

        image = tio.ToReferenceSpace.from_tensor(tensor, reference)

        assert isinstance(image, tio.LabelMap)
        assert image.spacing == (2.0, 4.0, 6.0)

    def test_wrong_reference_type(self):
        with pytest.raises(TypeError, match='TorchIO image'):
            tio.ToReferenceSpace(torch.zeros(1, 2, 3, 4))
