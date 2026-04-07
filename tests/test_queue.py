"""Tests for Queue."""

from __future__ import annotations

import torch

import torchio as tio


def _make_subjects(
    n: int = 4,
    shape: tuple[int, int, int] = (20, 20, 20),
) -> list[tio.Subject]:
    return [
        tio.Subject(
            t1=tio.ScalarImage(
                torch.rand(1, *shape) + i,
            ),
        )
        for i in range(n)
    ]


class TestQueueBasic:
    def test_yields_correct_total(self) -> None:
        subjects = _make_subjects(4)
        sampler = tio.UniformSampler(subjects[0], patch_size=8)
        queue = tio.Queue(
            subjects,
            patch_sampler=sampler,
            patches_per_volume=5,
            max_length=50,
        )
        patches = list(queue)
        assert len(patches) == 4 * 5

    def test_shuffle_patches(self) -> None:
        subjects = _make_subjects(4)
        sampler = tio.UniformSampler(subjects[0], patch_size=8)
        queue = tio.Queue(
            subjects,
            patch_sampler=sampler,
            patches_per_volume=5,
            shuffle_patches=True,
        )
        patches = list(queue)
        assert len(patches) == 20

    def test_no_shuffle(self) -> None:
        subjects = _make_subjects(2)
        sampler = tio.UniformSampler(subjects[0], patch_size=8)
        queue = tio.Queue(
            subjects,
            patch_sampler=sampler,
            patches_per_volume=3,
            shuffle_subjects=False,
            shuffle_patches=False,
        )
        patches = list(queue)
        assert len(patches) == 6


class TestQueueTransform:
    def test_transform_applied(self) -> None:
        subjects = _make_subjects(2)
        sampler = tio.UniformSampler(subjects[0], patch_size=8)
        transform = tio.Flip(axes=0, p=1)
        queue = tio.Queue(
            subjects,
            patch_sampler=sampler,
            patches_per_volume=2,
            transform=transform,
        )
        patches = list(queue)
        assert len(patches) == 4


class TestQueueThreaded:
    def test_num_workers(self) -> None:
        subjects = _make_subjects(4)
        sampler = tio.UniformSampler(subjects[0], patch_size=8)
        queue = tio.Queue(
            subjects,
            patch_sampler=sampler,
            patches_per_volume=3,
            num_workers=2,
        )
        patches = list(queue)
        assert len(patches) == 12


class TestQueueDistributed:
    def test_disjoint_subsets(self) -> None:
        """Two simulated ranks receive disjoint subject subsets."""
        from torch.utils.data import SubsetRandomSampler

        subjects = _make_subjects(6)
        sampler = tio.UniformSampler(subjects[0], patch_size=8)

        rank0_indices = [0, 2, 4]
        rank1_indices = [1, 3, 5]
        sampler0 = SubsetRandomSampler(rank0_indices)
        sampler1 = SubsetRandomSampler(rank1_indices)

        queue0 = tio.Queue(
            subjects,
            patch_sampler=sampler,
            patches_per_volume=2,
            shuffle_subjects=False,
            subject_sampler=sampler0,
        )
        queue1 = tio.Queue(
            subjects,
            patch_sampler=sampler,
            patches_per_volume=2,
            shuffle_subjects=False,
            subject_sampler=sampler1,
        )

        patches0 = list(queue0)
        patches1 = list(queue1)
        assert len(patches0) == 3 * 2
        assert len(patches1) == 3 * 2

    def test_shuffle_with_sampler_raises(self) -> None:
        import pytest
        from torch.utils.data import SubsetRandomSampler

        subjects = _make_subjects(2)
        sampler = tio.UniformSampler(subjects[0], patch_size=8)
        with pytest.raises(ValueError, match="shuffle_subjects"):
            tio.Queue(
                subjects,
                patch_sampler=sampler,
                shuffle_subjects=True,
                subject_sampler=SubsetRandomSampler([0, 1]),
            )


class TestQueueMemory:
    def test_max_memory(self) -> None:
        subjects = _make_subjects(2, shape=(10, 10, 10))
        sampler = tio.UniformSampler(subjects[0], patch_size=8)
        queue = tio.Queue(
            subjects,
            patch_sampler=sampler,
            max_length=100,
        )
        # 1 channel * 8^3 voxels * 4 bytes * 100 patches
        assert queue.max_memory == 1 * 512 * 4 * 100

    def test_max_memory_pretty(self) -> None:
        subjects = _make_subjects(2, shape=(10, 10, 10))
        sampler = tio.UniformSampler(subjects[0], patch_size=8)
        queue = tio.Queue(
            subjects,
            patch_sampler=sampler,
            max_length=100,
        )
        assert "KiB" in queue.max_memory_pretty


class TestQueueProperties:
    def test_num_subjects(self) -> None:
        subjects = _make_subjects(5)
        sampler = tio.UniformSampler(subjects[0], patch_size=8)
        queue = tio.Queue(subjects, patch_sampler=sampler)
        assert queue.num_subjects == 5

    def test_patches_per_epoch(self) -> None:
        subjects = _make_subjects(4)
        sampler = tio.UniformSampler(subjects[0], patch_size=8)
        queue = tio.Queue(
            subjects,
            patch_sampler=sampler,
            patches_per_volume=10,
        )
        assert queue.patches_per_epoch == 40
