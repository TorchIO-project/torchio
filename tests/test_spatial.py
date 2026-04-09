"""Tests for spatial transforms."""

from __future__ import annotations

import numpy as np
import torch

import torchio as tio
from torchio import AffineMatrix
from torchio.transforms import Affine as AffineTransform


def _make_subject(
    shape: tuple[int, int, int] = (11, 11, 11),
    spacing: tuple[float, float, float] = (1.0, 1.0, 1.0),
) -> tio.Subject:
    data = torch.arange(
        shape[0] * shape[1] * shape[2],
        dtype=torch.float32,
    ).reshape(1, *shape)
    label = torch.zeros(1, *shape, dtype=torch.float32)
    label[
        0,
        shape[0] // 4 : shape[0] // 4 + 2,
        shape[1] // 4 : shape[1] // 4 + 2,
        shape[2] // 4 : shape[2] // 4 + 2,
    ] = 1
    affine = np.diag([spacing[0], spacing[1], spacing[2], 1.0])
    return tio.Subject(
        t1=tio.ScalarImage(data, affine=affine),
        seg=tio.LabelMap(label, affine=affine),
    )


class TestSpatial:
    def test_identity(self) -> None:
        subject = _make_subject()
        transformed = tio.Spatial()(subject)

        torch.testing.assert_close(transformed.t1.data, subject.t1.data)
        torch.testing.assert_close(transformed.seg.data, subject.seg.data)
        np.testing.assert_allclose(
            transformed.t1.affine.numpy(),
            subject.t1.affine.numpy(),
        )

    def test_affine_first_changes_result(self) -> None:
        subject = _make_subject()
        control_points = torch.zeros(5, 5, 5, 3)
        control_points[2, 2, 2, 0] = 2.0

        kwargs = {
            "scales": 1.0,
            "degrees": (0.0, 0.0, 45.0),
            "translation": 0.0,
            "control_points": control_points,
            "default_pad_value": 0.0,
            "default_pad_label": 0.0,
        }

        first = tio.Spatial(affine_first=True, **kwargs)(subject)
        second = tio.Spatial(affine_first=False, **kwargs)(subject)

        assert not torch.allclose(first.t1.data, second.t1.data)


class TestResample:
    def test_spacing_target_changes_shape_and_affine(self) -> None:
        subject = _make_subject(shape=(12, 12, 12), spacing=(1.0, 1.0, 1.0))
        transformed = tio.Resample(2)(subject)

        assert transformed.t1.spatial_shape == (6, 6, 6)
        assert transformed.seg.spatial_shape == (6, 6, 6)
        np.testing.assert_allclose(transformed.t1.spacing, (2.0, 2.0, 2.0))
        np.testing.assert_allclose(
            transformed.t1.affine.numpy(),
            transformed.seg.affine.numpy(),
        )

    def test_named_image_target_uses_reference_space(self) -> None:
        reference = tio.ScalarImage(
            torch.ones(1, 6, 6, 6),
            affine=np.diag([2.0, 2.0, 2.0, 1.0]),
        )
        moving = tio.ScalarImage(
            torch.ones(1, 12, 12, 12),
            affine=np.diag([1.0, 1.0, 1.0, 1.0]),
        )
        subject = tio.Subject(t1=reference, t2=moving)

        transformed = tio.Resample("t1")(subject)

        assert transformed.t2.spatial_shape == transformed.t1.spatial_shape
        np.testing.assert_allclose(
            transformed.t2.affine.numpy(),
            transformed.t1.affine.numpy(),
        )

    def test_inverse_restores_geometry(self) -> None:
        subject = _make_subject(shape=(12, 12, 12))
        transformed = tio.Resample(2)(subject)

        restored = transformed.apply_inverse_transform()

        assert restored.t1.spatial_shape == subject.t1.spatial_shape
        np.testing.assert_allclose(
            restored.t1.affine.numpy(),
            subject.t1.affine.numpy(),
        )


class TestAffine:
    def test_transform_changes_data(self) -> None:
        subject = _make_subject()
        transform = AffineTransform(
            scales=1.0,
            degrees=(0.0, 0.0, 90.0),
            translation=0.0,
            center="image",
            default_pad_value=0.0,
            default_pad_label=0.0,
        )

        transformed = transform(subject)

        assert not torch.allclose(transformed.t1.data, subject.t1.data)
        np.testing.assert_allclose(
            transformed.t1.affine.numpy(),
            subject.t1.affine.numpy(),
        )

    def test_inverse_restores_geometry(self) -> None:
        subject = _make_subject()
        transform = AffineTransform(
            scales=(1.1, 0.9, 1.0),
            degrees=(0.0, 0.0, 20.0),
            translation=(1.0, -2.0, 0.5),
            center="image",
            default_pad_value=0.0,
            default_pad_label=0.0,
        )

        restored = transform(subject).apply_inverse_transform()

        assert restored.t1.spatial_shape == subject.t1.spatial_shape
        np.testing.assert_allclose(
            restored.t1.affine.numpy(),
            subject.t1.affine.numpy(),
        )


class TestElasticDeformation:
    def test_accepts_tensor_control_points(self) -> None:
        subject = _make_subject()
        control_points = torch.zeros(5, 5, 5, 3)
        control_points[2, 2, 2, 0] = 2.0

        transformed = tio.ElasticDeformation(control_points=control_points)(subject)

        assert not torch.allclose(transformed.t1.data, subject.t1.data)

    def test_label_interpolation_preserves_label_values(self) -> None:
        subject = _make_subject()
        transformed = AffineTransform(
            scales=(1.1, 1.0, 1.0),
            degrees=(0.0, 0.0, 15.0),
            translation=0.0,
            center="image",
            default_pad_value=0.0,
            default_pad_label=0.0,
        )(subject)

        unique = set(transformed.seg.data.unique().tolist())
        assert unique <= {0.0, 1.0}


class TestExports:
    def test_root_exports_expose_transform_and_matrix(self) -> None:
        assert hasattr(tio, "Spatial")
        assert hasattr(tio, "Resample")
        assert hasattr(tio, "ElasticDeformation")
        assert tio.Affine is AffineTransform
        assert tio.AffineMatrix is AffineMatrix
