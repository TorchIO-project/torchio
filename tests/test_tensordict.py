"""Tests for tensordict integration and data loaders."""

from __future__ import annotations

import torch
from torch.utils.data import Dataset

import torchio as tio
from torchio.data.bboxes import BoundingBoxes
from torchio.data.bboxes import BoundingBoxFormat
from torchio.data.points import Points


def _make_subject(idx: int = 0) -> tio.Subject:
    return tio.Subject(
        t1=tio.ScalarImage.from_tensor(torch.rand(1, 16, 16, 16)),
        seg=tio.LabelMap.from_tensor(torch.randint(0, 3, (1, 16, 16, 16))),
        landmarks=Points(torch.rand(5, 3)),
        tumors=BoundingBoxes(
            torch.rand(2, 6),
            format=BoundingBoxFormat.IJKIJK,
        ),
        age=42 + idx,
        name=f"subject_{idx}",
    )


# ── Image.to_tensordict / from_tensordict ─────────────────────────────


class TestImageTensorDict:
    def test_round_trip(self) -> None:
        image = tio.ScalarImage.from_tensor(torch.rand(1, 16, 16, 16))
        td = image.to_tensordict()
        restored = tio.Image.from_tensordict(td)
        torch.testing.assert_close(restored.data, image.data)

    def test_affine_preserved(self) -> None:
        image = tio.ScalarImage.from_tensor(torch.rand(1, 16, 16, 16))
        td = image.to_tensordict()
        restored = tio.Image.from_tensordict(td)
        torch.testing.assert_close(
            torch.as_tensor(restored.affine.numpy()),
            torch.as_tensor(image.affine.numpy()),
        )

    def test_class_preserved(self) -> None:
        for cls in (tio.ScalarImage, tio.LabelMap):
            image = cls.from_tensor(torch.rand(1, 8, 8, 8))
            td = image.to_tensordict()
            restored = tio.Image.from_tensordict(td)
            assert type(restored) is cls

    def test_metadata_preserved(self) -> None:
        image = tio.ScalarImage.from_tensor(
            torch.rand(1, 8, 8, 8),
            metadata={"protocol": "MPRAGE"},
        )
        td = image.to_tensordict()
        restored = tio.Image.from_tensordict(td)
        assert restored.metadata["protocol"] == "MPRAGE"

    def test_points_preserved(self) -> None:
        pts = Points(torch.rand(3, 3))
        image = tio.ScalarImage.from_tensor(
            torch.rand(1, 8, 8, 8),
            points={"lm": pts},
        )
        td = image.to_tensordict()
        restored = tio.Image.from_tensordict(td)
        assert "lm" in restored.points
        torch.testing.assert_close(restored.points["lm"].data, pts.data)

    def test_bboxes_preserved(self) -> None:
        boxes = BoundingBoxes(
            torch.rand(2, 6),
            format=BoundingBoxFormat.IJKIJK,
        )
        image = tio.ScalarImage.from_tensor(
            torch.rand(1, 8, 8, 8),
            bounding_boxes={"roi": boxes},
        )
        td = image.to_tensordict()
        restored = tio.Image.from_tensordict(td)
        assert "roi" in restored.bounding_boxes
        torch.testing.assert_close(restored.bounding_boxes["roi"].data, boxes.data)

    def test_stacking(self) -> None:
        images = [tio.ScalarImage.from_tensor(torch.rand(1, 8, 8, 8)) for _ in range(4)]
        tds = [img.to_tensordict() for img in images]
        batch = torch.stack(tds)
        assert batch.batch_size == torch.Size([4])
        assert batch["data"].shape == (4, 1, 8, 8, 8)


# ── Subject.to_tensordict / from_tensordict ────────────────────────────


class TestSubjectTensorDict:
    def test_round_trip(self) -> None:
        subject = _make_subject()
        td = subject.to_tensordict()
        restored = tio.Subject.from_tensordict(td)

        assert set(restored.images.keys()) == {"t1", "seg"}
        torch.testing.assert_close(restored.t1.data, subject.t1.data)
        torch.testing.assert_close(restored.seg.data, subject.seg.data)

    def test_affine_preserved(self) -> None:
        subject = _make_subject()
        td = subject.to_tensordict()
        restored = tio.Subject.from_tensordict(td)
        torch.testing.assert_close(
            torch.as_tensor(restored.t1.affine.numpy()),
            torch.as_tensor(subject.t1.affine.numpy()),
        )

    def test_metadata_preserved(self) -> None:
        subject = _make_subject(7)
        td = subject.to_tensordict()
        restored = tio.Subject.from_tensordict(td)
        assert restored.metadata["age"] == 49
        assert restored.metadata["name"] == "subject_7"

    def test_points_preserved(self) -> None:
        subject = _make_subject()
        td = subject.to_tensordict()
        restored = tio.Subject.from_tensordict(td)
        assert "landmarks" in restored.points
        torch.testing.assert_close(
            restored.landmarks.data,
            subject.landmarks.data,
        )

    def test_bboxes_preserved(self) -> None:
        subject = _make_subject()
        td = subject.to_tensordict()
        restored = tio.Subject.from_tensordict(td)
        assert "tumors" in restored.bounding_boxes
        torch.testing.assert_close(
            restored.tumors.data,
            subject.tumors.data,
        )

    def test_image_types_preserved(self) -> None:
        subject = _make_subject()
        td = subject.to_tensordict()
        restored = tio.Subject.from_tensordict(td)
        assert isinstance(restored.t1, tio.ScalarImage)
        assert isinstance(restored.seg, tio.LabelMap)

    def test_stacking(self) -> None:
        subjects = [_make_subject(i) for i in range(4)]
        tds = [s.to_tensordict() for s in subjects]
        batch = torch.stack(tds)

        assert batch.batch_size == torch.Size([4])
        assert batch["t1", "data"].shape == (4, 1, 16, 16, 16)

    def test_metadata_only_subject(self) -> None:
        subject = tio.Subject(age=42, name="test")
        td = subject.to_tensordict()
        restored = tio.Subject.from_tensordict(td)
        assert restored.metadata["age"] == 42
        assert restored.metadata["name"] == "test"

    def test_applied_transforms_preserved(self) -> None:
        subject = _make_subject()
        subject.applied_transforms = [{"name": "RandomFlip", "axis": 0}]
        td = subject.to_tensordict()
        restored = tio.Subject.from_tensordict(td)
        assert len(restored.applied_transforms) == 1
        assert restored.applied_transforms[0]["name"] == "RandomFlip"


# ── collate_subjects / collate_images ──────────────────────────────────


class TestCollate:
    def test_collate_subjects(self) -> None:
        subjects = [_make_subject(i) for i in range(4)]
        tds = [s.to_tensordict() for s in subjects]
        batch = tio.collate_subjects(tds)
        assert batch.batch_size == torch.Size([4])
        assert batch["t1", "data"].shape == (4, 1, 16, 16, 16)

    def test_collate_images(self) -> None:
        images = [tio.ScalarImage.from_tensor(torch.rand(1, 8, 8, 8)) for _ in range(4)]
        tds = [img.to_tensordict() for img in images]
        batch = tio.collate_images(tds)
        assert batch.batch_size == torch.Size([4])
        assert batch["data"].shape == (4, 1, 8, 8, 8)


# ── SubjectsLoader ────────────────────────────────────────────────────


class _SimpleSubjectsDataset(Dataset):
    def __init__(self, n: int = 8) -> None:
        self.n = n

    def __len__(self) -> int:
        return self.n

    def __getitem__(self, idx: int) -> tio.Subject:
        return _make_subject(idx)


class TestSubjectsLoader:
    def test_basic_iteration(self) -> None:
        dataset = _SimpleSubjectsDataset(8)
        loader = tio.SubjectsLoader(dataset, batch_size=4)
        batch = next(iter(loader))

        assert batch.batch_size == torch.Size([4])
        assert batch["t1", "data"].shape == (4, 1, 16, 16, 16)

    def test_all_batches(self) -> None:
        dataset = _SimpleSubjectsDataset(8)
        loader = tio.SubjectsLoader(dataset, batch_size=4)
        batches = list(loader)
        assert len(batches) == 2

    def test_metadata_in_batch(self) -> None:
        dataset = _SimpleSubjectsDataset(4)
        loader = tio.SubjectsLoader(dataset, batch_size=4)
        batch = next(iter(loader))
        ages = [batch[i].get_non_tensor("_meta_age") for i in range(4)]
        assert ages == [42, 43, 44, 45]

    def test_variable_size_points(self) -> None:
        """Subjects with different numbers of points should batch fine."""

        class VarPointsDataset(Dataset):
            def __len__(self) -> int:
                return 4

            def __getitem__(self, idx: int) -> tio.Subject:
                n_points = 3 + idx
                return tio.Subject(
                    t1=tio.ScalarImage.from_tensor(torch.rand(1, 8, 8, 8)),
                    pts=Points(torch.rand(n_points, 3)),
                )

        loader = tio.SubjectsLoader(VarPointsDataset(), batch_size=4)
        batch = next(iter(loader))
        assert batch["t1", "data"].shape == (4, 1, 8, 8, 8)

    def test_passes_dataloader_kwargs(self) -> None:
        dataset = _SimpleSubjectsDataset(8)
        loader = tio.SubjectsLoader(
            dataset,
            batch_size=2,
            shuffle=False,
            num_workers=0,
        )
        batches = list(loader)
        assert len(batches) == 4


# ── ImagesLoader ──────────────────────────────────────────────────────


class _SimpleImagesDataset(Dataset):
    def __init__(self, n: int = 8) -> None:
        self.n = n

    def __len__(self) -> int:
        return self.n

    def __getitem__(self, idx: int) -> tio.ScalarImage:
        return tio.ScalarImage.from_tensor(torch.rand(1, 8, 8, 8))


class TestImagesLoader:
    def test_basic_iteration(self) -> None:
        dataset = _SimpleImagesDataset(8)
        loader = tio.ImagesLoader(dataset, batch_size=4)
        batch = next(iter(loader))

        assert batch.batch_size == torch.Size([4])
        assert batch["data"].shape == (4, 1, 8, 8, 8)

    def test_all_batches(self) -> None:
        dataset = _SimpleImagesDataset(8)
        loader = tio.ImagesLoader(dataset, batch_size=4)
        batches = list(loader)
        assert len(batches) == 2

    def test_affine_in_batch(self) -> None:
        dataset = _SimpleImagesDataset(4)
        loader = tio.ImagesLoader(dataset, batch_size=4)
        batch = next(iter(loader))
        assert batch["affine"].shape == (4, 4, 4)
