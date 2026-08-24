"""Tests for Motion transform."""

from __future__ import annotations

import pytest
import torch

import torchio as tio
from torchio.transforms.intensity.motion import _apply_rigid_transform


def _make_subject(with_label: bool = True) -> tio.Subject:
    data = torch.rand(1, 10, 10, 10) * 100
    kwargs: dict = {"t1": tio.ScalarImage(data)}
    if with_label:
        seg = torch.zeros(1, 10, 10, 10, dtype=torch.float32)
        seg[0, 2:5, 2:5, 2:5] = 1
        seg[0, 6:9, 6:9, 6:9] = 2
        kwargs["seg"] = tio.LabelMap(seg)
    return tio.Subject(**kwargs)


class TestMotion:
    def test_changes_data(self) -> None:
        subject = _make_subject(with_label=False)
        original = subject.t1.data.clone()
        result = tio.Motion(degrees=15, translation=10)(subject)
        assert not torch.allclose(result.t1.data, original)

    def test_num_transforms_validation(self) -> None:
        with pytest.raises(ValueError, match="num_transforms"):
            tio.Motion(num_transforms=0)

    def test_leaves_labels_unchanged(self) -> None:
        subject = _make_subject()
        original_seg = subject.seg.data.clone()
        result = tio.Motion()(subject)
        torch.testing.assert_close(result.seg.data, original_seg)

    def test_preserves_shape(self) -> None:
        subject = _make_subject(with_label=False)
        result = tio.Motion()(subject)
        assert result.t1.data.shape == subject.t1.data.shape

    def test_single_transform(self) -> None:
        subject = _make_subject(with_label=False)
        result = tio.Motion(num_transforms=1)(subject)
        assert result.t1.data.shape == subject.t1.data.shape


class TestMotionPerInstance:
    def _batch(self, batch_size: int = 5) -> tio.SubjectsBatch:
        data = torch.rand(1, 12, 12, 12)
        subjects = [
            tio.Subject(t1=tio.ScalarImage(data.clone())) for _ in range(batch_size)
        ]
        return tio.SubjectsBatch.from_subjects(subjects)

    def test_per_instance_differs_across_batch(self) -> None:
        torch.manual_seed(0)
        batch = self._batch()
        result = tio.Motion(degrees=(5, 15), translation=(5, 15), num_transforms=2)(
            batch
        )
        params = result.applied_transforms[-1].params
        assert "_batched_keys" in params
        assert len(params["transforms"]) == batch.batch_size
        assert not torch.allclose(result.t1.data[0], result.t1.data[1])

    def test_per_instance_false_is_shared(self) -> None:
        torch.manual_seed(0)
        batch = self._batch()
        transform = tio.Motion(
            degrees=(5, 15),
            translation=(5, 15),
            num_transforms=2,
            per_instance=False,
        )
        result = transform(batch)
        torch.testing.assert_close(result.t1.data[0], result.t1.data[1])

    def test_single_subject_keeps_scalar_params(self) -> None:
        subject = tio.Subject(t1=tio.ScalarImage(torch.rand(1, 12, 12, 12)))
        result = tio.Motion(degrees=15, translation=10)(subject)
        assert "_batched_keys" not in result.applied_transforms[-1].params


class TestMotionDegenerateSegments:
    def test_too_many_transforms_for_first_axis_raises(self) -> None:
        # num_transforms + 1 segments cannot exceed the first spatial axis
        # size; the transform must raise a clear error rather than silently
        # replacing the whole spectrum.
        subject = tio.Subject(t1=tio.ScalarImage(torch.rand(1, 2, 8, 8)))
        with pytest.raises(ValueError, match="motion segments"):
            tio.Motion(degrees=5, translation=5, num_transforms=4)(subject)


class TestMotionAffineParity:
    """Motion's rigid step shares parameter conventions with Affine.

    See #812: the same Euler angles and translation must describe the
    same geometric transform in `Motion` and in `Affine`. For an image
    whose affine is the identity, the voxel-space rigid step used by
    `Motion` must match `Affine(..., center="image")` exactly.
    """

    @staticmethod
    def _volume() -> torch.Tensor:
        # Distinct axis sizes expose axis-order, sign, and normalization
        # mistakes that cubic volumes hide.
        torch.manual_seed(42)
        data = torch.zeros(1, 12, 16, 20)
        data[0, 3:7, 5:10, 8:14] = torch.rand(4, 5, 6) + 1
        return data

    @pytest.mark.parametrize(
        ("degrees", "translation"),
        [
            ((0.0, 0.0, 0.0), (2.0, -3.0, 4.0)),
            ((10.0, 0.0, 0.0), (0.0, 0.0, 0.0)),
            ((0.0, 10.0, 0.0), (0.0, 0.0, 0.0)),
            ((0.0, 0.0, 10.0), (0.0, 0.0, 0.0)),
            ((8.0, -5.0, 12.0), (3.0, 1.5, -2.0)),
        ],
    )
    def test_rigid_step_matches_affine(
        self,
        degrees: tuple[float, float, float],
        translation: tuple[float, float, float],
    ) -> None:
        data = self._volume()
        moved = _apply_rigid_transform(
            data.unsqueeze(0),
            torch.tensor([degrees]),
            torch.tensor([translation]),
        )[0]
        transform = tio.Affine(
            degrees=degrees,
            translation=translation,
            center="image",
            default_pad_value=0,
        )
        expected = transform(tio.Subject(t1=tio.ScalarImage(data.clone()))).t1.data
        torch.testing.assert_close(moved, expected, atol=1e-4, rtol=1e-4)

    def test_integer_translation_matches_roll(self) -> None:
        # A pure integer-voxel translation must shift the volume by
        # exactly that many voxels along the tensor axes (i, j, k),
        # in the positive index direction.
        torch.manual_seed(0)
        data = torch.rand(1, 1, 12, 16, 20)
        shifts = (2, -3, 4)
        moved = _apply_rigid_transform(
            data,
            torch.zeros(1, 3),
            torch.tensor([shifts], dtype=torch.float32),
        )
        expected = torch.roll(data, shifts=shifts, dims=(2, 3, 4))
        # Trim the wrapped-around (rolled) and zero-padded (moved) borders.
        interior = (
            slice(None),
            slice(None),
            slice(2, None),
            slice(None, -3),
            slice(4, None),
        )
        torch.testing.assert_close(
            moved[interior],
            expected[interior],
            atol=1e-5,
            rtol=1e-5,
        )

    @pytest.mark.filterwarnings(
        "ignore:.*affine_grid behavior has changed.*:UserWarning",
    )
    def test_singleton_axis_does_not_crash(self) -> None:
        # A 2D (single-slice) image has a zero half-extent along the
        # last axis; the transform must stay finite.
        subject = tio.Subject(t1=tio.ScalarImage(torch.rand(1, 8, 8, 1)))
        result = tio.Motion(degrees=5, translation=2, num_transforms=1)(subject)
        assert torch.isfinite(result.t1.data).all()
