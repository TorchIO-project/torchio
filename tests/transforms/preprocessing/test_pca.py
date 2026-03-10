import copy

import numpy as np
import pytest
import torch

import torchio as tio
from torchio.transforms.preprocessing.intensity.pca import _compute_pca

from ...utils import TorchioTestCase

pytest.importorskip('sklearn')


def get_embedding_image():
    shape = (2, 2, 2)
    flat = np.array([10, 4, 3, 2, 1, 0, 0, 0], dtype=np.float32)
    values = flat.reshape(shape)
    coords = np.indices(shape).astype(np.float32)
    channels = np.stack(
        [
            -values,
            values + coords[0],
            flat[::-1].reshape(shape) + 2 * coords[1],
            0.5 * values + coords[2],
        ]
    )
    affine = np.diag([1.5, 2.5, 3.5, 1.0])
    return tio.ScalarImage(tensor=torch.from_numpy(channels), affine=affine)


def get_skewness(component: np.ndarray) -> np.float32:
    third_cumulant = np.mean(component**3)
    second_cumulant = np.mean(component**2)
    return third_cumulant / second_cumulant ** (3 / 2)


class TestPCA(TorchioTestCase):
    def test_basic_pca(self):
        image = get_embedding_image()
        transformed = tio.PCA(num_components=2)(image)

        assert transformed.shape == (2, 2, 2, 2)
        self.assert_tensor_almost_equal(transformed.affine, image.affine)
        assert transformed.data.min().item() >= 0
        assert transformed.data.max().item() <= 1

    def test_include(self):
        image = get_embedding_image()
        subject = tio.Subject(
            t1=image,
            t2=tio.ScalarImage(tensor=image.data + 3, affine=image.affine),
        )
        original = copy.deepcopy(subject)

        transformed = tio.PCA(num_components=2, include=['t1'])(subject)

        assert transformed.t1.shape == (2, 2, 2, 2)
        self.assert_tensor_equal(transformed.t2.data, original.t2.data)
        self.assert_tensor_almost_equal(transformed.t2.affine, original.t2.affine)

    def test_values_range_none_uses_data_range(self):
        image = get_embedding_image()
        transformed = tio.PCA(
            num_components=2,
            whiten=False,
            normalize=False,
            make_skewness_positive=False,
            values_range=None,
            clip=False,
        )(image)

        assert transformed.data.min().item() == pytest.approx(0)
        assert transformed.data.max().item() == pytest.approx(1)

    def test_clip(self):
        image = get_embedding_image()
        kwargs = dict(
            num_components=2,
            whiten=False,
            normalize=False,
            make_skewness_positive=False,
            values_range=(-5, 5),
        )

        unclipped = tio.PCA(clip=False, **kwargs)(image)
        clipped = tio.PCA(clip=True, **kwargs)(image)

        assert unclipped.data.min().item() < 0
        assert unclipped.data.max().item() > 1
        assert clipped.data.min().item() >= 0
        assert clipped.data.max().item() <= 1

    def test_make_skewness_positive(self):
        """make_skewness_positive should flip the leading component to positive skew."""
        image = get_embedding_image()
        kwargs = dict(
            embeddings=image,
            num_components=2,
            whiten=False,
            normalize=False,
            values_range=(-100, 100),
            clip=False,
        )

        not_corrected = _compute_pca(
            make_skewness_positive=False,
            **kwargs,
        )
        corrected = _compute_pca(
            make_skewness_positive=True,
            **kwargs,
        )

        not_corrected_first = not_corrected.data[0].numpy() * 200 - 100
        corrected_first = corrected.data[0].numpy() * 200 - 100

        assert get_skewness(not_corrected_first) < 0
        assert get_skewness(corrected_first) > 0

    def test_num_components_too_large(self):
        image = get_embedding_image()
        with pytest.raises(ValueError):
            tio.PCA(num_components=5)(image)

    def test_pca_kwargs(self):
        image = get_embedding_image()
        transformed = tio.PCA(
            num_components=2,
            whiten=False,
            pca_kwargs={'svd_solver': 'full'},
        )(image)
        assert transformed.shape == (2, 2, 2, 2)
