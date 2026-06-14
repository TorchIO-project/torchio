"""Tests for SomeOf transform."""

from __future__ import annotations

import torch

import torchio as tio


def _make_subject() -> tio.Subject:
    return tio.Subject(
        t1=tio.ScalarImage(torch.rand(1, 10, 10, 10)),
    )


class TestSomeOf:
    def test_applies_subset(self) -> None:
        subject = _make_subject()
        transform = tio.SomeOf(
            [tio.Flip(axes=(0,)), tio.Gamma(log_gamma=0.0)],
            num_transforms=1,
        )
        result = transform(subject)
        assert result.t1.data.shape == subject.t1.data.shape

    def test_all_transforms(self) -> None:
        subject = _make_subject()
        transform = tio.SomeOf(
            [tio.Flip(axes=(0,)), tio.Gamma(log_gamma=0.0)],
            num_transforms=2,
        )
        result = transform(subject)
        assert result.t1.data.shape == subject.t1.data.shape

    def test_range_num_transforms(self) -> None:
        subject = _make_subject()
        transform = tio.SomeOf(
            [
                tio.Flip(axes=(0,)),
                tio.Gamma(log_gamma=0.0),
                tio.Blur(std=0.0),
            ],
            num_transforms=(1, 3),
        )
        result = transform(subject)
        assert result.t1.data.shape == subject.t1.data.shape

    def test_history_recorded(self) -> None:
        subject = _make_subject()
        transform = tio.SomeOf(
            [tio.Flip(axes=(0,))],
            num_transforms=1,
        )
        result = transform(subject)
        assert len(result.applied_transforms) > 0


class TestSomeOfPerInstance:
    def _batch(self, batch_size: int = 16) -> tio.SubjectsBatch:
        data = torch.rand(1, 8, 8, 8)
        subjects = [
            tio.Subject(t1=tio.ScalarImage(data.clone())) for _ in range(batch_size)
        ]
        return tio.SubjectsBatch.from_subjects(subjects)

    def test_per_element_subsets_differ(self) -> None:
        torch.manual_seed(0)
        batch = self._batch()
        transform = tio.SomeOf(
            [tio.Gamma(log_gamma=0.5), tio.Flip(axes=(0,)), tio.Noise(std=0.3)],
            num_transforms=2,
        )
        result = transform(batch)
        counts = {len(subject.applied_transforms) for subject in result.unbatch()}
        # Each element applies exactly two transforms.
        assert counts == {2}
        name_sets = {
            tuple(sorted(t.name for t in subject.applied_transforms))
            for subject in result.unbatch()
        }
        assert len(name_sets) > 1

    def test_per_instance_false_is_batch_wide(self) -> None:
        torch.manual_seed(0)
        batch = self._batch(batch_size=8)
        transform = tio.SomeOf(
            [tio.Gamma(log_gamma=0.5), tio.Flip(axes=(0,)), tio.Noise(std=0.3)],
            num_transforms=2,
            per_instance=False,
        )
        result = transform(batch)
        name_sets = {
            tuple(sorted(t.name for t in subject.applied_transforms))
            for subject in result.unbatch()
        }
        assert len(name_sets) == 1

    def test_single_subject_unaffected(self) -> None:
        subject = _make_subject()
        transform = tio.SomeOf(
            [tio.Gamma(log_gamma=0.5), tio.Flip(axes=(0,))],
            num_transforms=1,
        )
        result = transform(subject)
        assert len(result.applied_transforms) == 1


class TestSomeOfCopy:
    def test_does_not_mutate_input(self) -> None:
        subject = _make_subject()
        snapshot = subject.t1.data.clone()
        tio.SomeOf([tio.Gamma(log_gamma=0.5)], num_transforms=1)(subject)
        torch.testing.assert_close(subject.t1.data, snapshot)

    def test_restores_child_copy_flag(self) -> None:
        child = tio.Gamma(log_gamma=0.5)
        assert child.copy is True
        tio.SomeOf([child], num_transforms=1)(_make_subject())
        assert child.copy is True

    def test_children_applied_without_copy(self) -> None:
        seen: list[bool] = []

        class _Spy(tio.IntensityTransform):
            def apply_transform(self, batch, params):
                seen.append(self.copy)
                return batch

        tio.SomeOf([_Spy()], num_transforms=1)(_make_subject())
        assert seen == [False]
