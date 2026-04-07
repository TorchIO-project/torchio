"""Tests for ImagesBatch and SubjectsBatch."""

from __future__ import annotations

import torch

import torchio as tio
from torchio.data.batch import ImagesBatch
from torchio.data.batch import SubjectsBatch


class TestImagesBatch:
    def test_from_images(self) -> None:
        images = [tio.ScalarImage(torch.rand(1, 8, 8, 8)) for _ in range(4)]
        batch = ImagesBatch.from_images(images)
        assert batch.data.shape == (4, 1, 8, 8, 8)

    def test_batch_size(self) -> None:
        batch = ImagesBatch(
            data=torch.rand(4, 1, 8, 8, 8),
            affines=[tio.Affine() for _ in range(4)],
        )
        assert batch.batch_size == 4

    def test_to_device(self) -> None:
        batch = ImagesBatch(
            data=torch.rand(2, 1, 4, 4, 4),
            affines=[tio.Affine() for _ in range(2)],
        )
        result = batch.to(torch.float64)
        assert result.data.dtype == torch.float64

    def test_unbatch(self) -> None:
        images = [tio.ScalarImage(torch.rand(1, 8, 8, 8)) for _ in range(3)]
        batch = ImagesBatch.from_images(images)
        restored = batch.unbatch()
        assert len(restored) == 3
        for img in restored:
            assert isinstance(img, tio.ScalarImage)
            assert img.shape == (1, 8, 8, 8)

    def test_getitem_int(self) -> None:
        batch = ImagesBatch(
            data=torch.rand(4, 1, 8, 8, 8),
            affines=[tio.Affine() for _ in range(4)],
        )
        img = batch[0]
        assert isinstance(img, tio.ScalarImage)
        assert img.shape == (1, 8, 8, 8)

    def test_per_sample_affines(self) -> None:
        affine_a = tio.Affine.from_spacing((1.0, 1.0, 1.0))
        affine_b = tio.Affine.from_spacing((2.0, 2.0, 2.0))
        batch = ImagesBatch(
            data=torch.rand(2, 1, 8, 8, 8),
            affines=[affine_a, affine_b],
        )
        assert batch[0].affine.spacing == (1.0, 1.0, 1.0)
        assert batch[1].affine.spacing == (2.0, 2.0, 2.0)

    def test_repr(self) -> None:
        batch = ImagesBatch(
            data=torch.rand(4, 1, 8, 8, 8),
            affines=[tio.Affine() for _ in range(4)],
        )
        assert "4" in repr(batch)
        assert "8" in repr(batch)


class TestSubjectsBatch:
    def test_from_subjects(self) -> None:
        subjects = [
            tio.Subject(
                t1=tio.ScalarImage(torch.rand(1, 8, 8, 8)),
                seg=tio.LabelMap(torch.randint(0, 3, (1, 8, 8, 8))),
                age=42 + i,
            )
            for i in range(4)
        ]
        batch = SubjectsBatch.from_subjects(subjects)
        assert batch["t1"].data.shape == (4, 1, 8, 8, 8)
        assert batch["seg"].data.shape == (4, 1, 8, 8, 8)

    def test_attribute_access(self) -> None:
        subjects = [
            tio.Subject(
                t1=tio.ScalarImage(torch.rand(1, 8, 8, 8)),
            )
            for _ in range(2)
        ]
        batch = SubjectsBatch.from_subjects(subjects)
        assert batch.t1.data.shape == (2, 1, 8, 8, 8)

    def test_batch_size(self) -> None:
        subjects = [
            tio.Subject(
                t1=tio.ScalarImage(torch.rand(1, 8, 8, 8)),
            )
            for _ in range(3)
        ]
        batch = SubjectsBatch.from_subjects(subjects)
        assert batch.batch_size == 3

    def test_unbatch(self) -> None:
        subjects = [
            tio.Subject(
                t1=tio.ScalarImage(torch.rand(1, 8, 8, 8)),
                age=42 + i,
            )
            for i in range(3)
        ]
        batch = SubjectsBatch.from_subjects(subjects)
        restored = batch.unbatch()
        assert len(restored) == 3
        for i, sub in enumerate(restored):
            assert isinstance(sub, tio.Subject)
            assert sub.t1.shape == (1, 8, 8, 8)
            assert sub.age == 42 + i

    def test_to_device(self) -> None:
        subjects = [
            tio.Subject(
                t1=tio.ScalarImage(torch.rand(1, 8, 8, 8)),
            )
            for _ in range(2)
        ]
        batch = SubjectsBatch.from_subjects(subjects)
        result = batch.to(torch.float64)
        assert result.t1.data.dtype == torch.float64

    def test_metadata_preserved(self) -> None:
        subjects = [
            tio.Subject(
                t1=tio.ScalarImage(torch.rand(1, 8, 8, 8)),
                age=42 + i,
                name=f"sub_{i}",
            )
            for i in range(3)
        ]
        batch = SubjectsBatch.from_subjects(subjects)
        assert batch.metadata["age"] == [42, 43, 44]
        assert batch.metadata["name"] == ["sub_0", "sub_1", "sub_2"]


class TestBatchTransforms:
    def test_flip_images_batch(self) -> None:
        images = [
            tio.ScalarImage(torch.arange(8).reshape(1, 2, 2, 2).float())
            for _ in range(3)
        ]
        batch = ImagesBatch.from_images(images)
        original = batch.data.clone()
        result = tio.Flip(axes=(0,))(batch)
        assert isinstance(result, ImagesBatch)
        assert result.data.shape == (3, 1, 2, 2, 2)
        # Check data was actually flipped
        assert not torch.equal(result.data, original)

    def test_flip_subjects_batch(self) -> None:
        subjects = [
            tio.Subject(
                t1=tio.ScalarImage(torch.rand(1, 8, 8, 8)),
            )
            for _ in range(4)
        ]
        batch = SubjectsBatch.from_subjects(subjects)
        result = tio.Flip(axes=(0,))(batch)
        assert isinstance(result, SubjectsBatch)
        assert result.t1.data.shape == (4, 1, 8, 8, 8)

    def test_noise_images_batch(self) -> None:
        images = [tio.ScalarImage(torch.zeros(1, 4, 4, 4)) for _ in range(3)]
        batch = ImagesBatch.from_images(images)
        result = tio.Noise(std=1.0)(batch)
        # Noise should have been added
        assert result.data.abs().sum() > 0

    def test_batch_preserves_affines(self) -> None:
        affine_a = tio.Affine.from_spacing((1.0, 1.0, 1.0))
        affine_b = tio.Affine.from_spacing((2.0, 2.0, 2.0))
        images = [
            tio.ScalarImage(torch.rand(1, 8, 8, 8), affine=affine_a),
            tio.ScalarImage(torch.rand(1, 8, 8, 8), affine=affine_b),
        ]
        batch = ImagesBatch.from_images(images)
        result = tio.Flip(axes=(0,))(batch)
        assert result.affines[0].spacing == (1.0, 1.0, 1.0)
        assert result.affines[1].spacing == (2.0, 2.0, 2.0)

    def test_batch_copy_preserves_original(self) -> None:
        images = [tio.ScalarImage(torch.zeros(1, 4, 4, 4)) for _ in range(2)]
        batch = ImagesBatch.from_images(images)
        original = batch.data.clone()
        tio.Noise(std=1.0)(batch)
        # Original should be unchanged (copy=True default)
        torch.testing.assert_close(batch.data, original)
