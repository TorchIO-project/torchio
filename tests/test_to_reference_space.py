"""Tests for ToReferenceSpace transform."""

from __future__ import annotations

import numpy as np
import pytest
import torch

import torchio as tio


def _reference(shape=(64, 64, 64), spacing=2.0, origin=(10, 20, 30)) -> tio.ScalarImage:
    affine = np.diag([spacing, spacing, spacing, 1.0])
    affine[:3, 3] = origin
    return tio.ScalarImage(
        torch.rand(1, *shape),
        affine=tio.AffineMatrix(affine),
    )


def _fov_center(image: tio.Image) -> np.ndarray:
    shape = np.array(image.spatial_shape)
    matrix = image.affine.data.cpu().numpy()
    corner0 = matrix @ np.array([0, 0, 0, 1.0])
    corner1 = matrix @ np.array([*(shape - 1), 1.0])
    return (corner0[:3] + corner1[:3]) / 2


class TestFromTensor:
    def test_shape_preserved(self) -> None:
        ref = _reference()
        embedding = torch.rand(8, 16, 16, 16)
        image = tio.ToReferenceSpace.from_tensor(embedding, ref)
        assert image.spatial_shape == (16, 16, 16)
        assert image.data.shape[0] == 8

    def test_spacing_scaled(self) -> None:
        ref = _reference(shape=(64, 64, 64), spacing=2.0)
        embedding = torch.rand(1, 16, 16, 16)  # downsample x4
        image = tio.ToReferenceSpace.from_tensor(embedding, ref)
        np.testing.assert_allclose(image.spacing, (8.0, 8.0, 8.0), atol=1e-5)

    def test_center_preserved(self) -> None:
        ref = _reference()
        embedding = torch.rand(1, 16, 16, 16)
        image = tio.ToReferenceSpace.from_tensor(embedding, ref)
        np.testing.assert_allclose(_fov_center(image), _fov_center(ref), atol=1e-4)

    def test_class_preserved(self) -> None:
        ref = tio.LabelMap(torch.zeros(1, 32, 32, 32))
        embedding = torch.rand(1, 8, 8, 8)
        image = tio.ToReferenceSpace.from_tensor(embedding, ref)
        assert isinstance(image, tio.LabelMap)

    def test_same_shape_keeps_affine(self) -> None:
        ref = _reference(shape=(32, 32, 32), spacing=1.5)
        embedding = torch.rand(1, 32, 32, 32)
        image = tio.ToReferenceSpace.from_tensor(embedding, ref)
        np.testing.assert_allclose(
            image.affine.data.cpu().numpy(),
            ref.affine.data.cpu().numpy(),
            atol=1e-5,
        )

    def test_anisotropic_shape(self) -> None:
        ref = _reference(shape=(64, 32, 16), spacing=1.0)
        embedding = torch.rand(1, 16, 16, 16)
        image = tio.ToReferenceSpace.from_tensor(embedding, ref)
        np.testing.assert_allclose(image.spacing, (4.0, 2.0, 1.0), atol=1e-5)


class TestTransform:
    def test_data_unchanged(self) -> None:
        ref = _reference()
        subject = tio.Subject(emb=tio.ScalarImage(torch.rand(8, 16, 16, 16)))
        original = subject.emb.data.clone()
        result = tio.ToReferenceSpace(ref)(subject)
        torch.testing.assert_close(result.emb.data, original)

    def test_affine_updated(self) -> None:
        ref = _reference()
        subject = tio.Subject(emb=tio.ScalarImage(torch.rand(1, 16, 16, 16)))
        result = tio.ToReferenceSpace(ref)(subject)
        np.testing.assert_allclose(result.emb.spacing, (8.0, 8.0, 8.0), atol=1e-5)

    def test_applies_to_all_images(self) -> None:
        ref = _reference()
        subject = tio.Subject(
            a=tio.ScalarImage(torch.rand(1, 16, 16, 16)),
            b=tio.ScalarImage(torch.rand(1, 8, 8, 8)),
        )
        result = tio.ToReferenceSpace(ref)(subject)
        np.testing.assert_allclose(result.a.spacing, (8.0, 8.0, 8.0), atol=1e-5)
        np.testing.assert_allclose(result.b.spacing, (16.0, 16.0, 16.0), atol=1e-5)

    def test_invalid_reference_raises(self) -> None:
        with pytest.raises(TypeError, match="reference must be a TorchIO Image"):
            tio.ToReferenceSpace("not an image")  # type: ignore[arg-type]


class TestExport:
    def test_top_level(self) -> None:
        assert hasattr(tio, "ToReferenceSpace")
