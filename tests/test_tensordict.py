"""Tests for data loaders and batch collation."""

from __future__ import annotations

import torch
from torch.utils.data import Dataset

import torchio as tio


def _make_subject(idx: int = 0) -> tio.Subject:
    return tio.Subject(
        t1=tio.ScalarImage.from_tensor(torch.rand(1, 16, 16, 16)),
        seg=tio.LabelMap.from_tensor(torch.randint(0, 3, (1, 16, 16, 16))),
        age=42 + idx,
        name=f"subject_{idx}",
    )


# ── collate_subjects / collate_images ──────────────────────────────────


class TestCollate:
    def test_collate_subjects(self) -> None:
        subjects = [_make_subject(i) for i in range(4)]
        batch = tio.collate_subjects(subjects)
        assert batch.batch_size == 4
        assert batch.t1.data.shape == (4, 1, 16, 16, 16)

    def test_collate_images(self) -> None:
        images = [tio.ScalarImage.from_tensor(torch.rand(1, 8, 8, 8)) for _ in range(4)]
        batch = tio.collate_images(images)
        assert batch.batch_size == 4
        assert batch.data.shape == (4, 1, 8, 8, 8)


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

        assert batch.batch_size == 4
        assert batch.t1.data.shape == (4, 1, 16, 16, 16)

    def test_all_batches(self) -> None:
        dataset = _SimpleSubjectsDataset(8)
        loader = tio.SubjectsLoader(dataset, batch_size=4)
        batches = list(loader)
        assert len(batches) == 2

    def test_metadata_in_batch(self) -> None:
        dataset = _SimpleSubjectsDataset(4)
        loader = tio.SubjectsLoader(dataset, batch_size=4)
        batch = next(iter(loader))
        assert batch.metadata["age"] == [42, 43, 44, 45]

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

        assert batch.batch_size == 4
        assert batch.data.shape == (4, 1, 8, 8, 8)

    def test_all_batches(self) -> None:
        dataset = _SimpleImagesDataset(8)
        loader = tio.ImagesLoader(dataset, batch_size=4)
        batches = list(loader)
        assert len(batches) == 2

    def test_affines_in_batch(self) -> None:
        dataset = _SimpleImagesDataset(4)
        loader = tio.ImagesLoader(dataset, batch_size=4)
        batch = next(iter(loader))
        assert len(batch.affines) == 4
        assert isinstance(batch.affines[0], tio.Affine)
