"""Tests for patch samplers and aggregator."""

from __future__ import annotations

import torch

import torchio as tio
from torchio.data.patch import PatchLocation


def _make_subject(
    shape: tuple[int, int, int] = (20, 20, 20),
) -> tio.Subject:
    data = torch.arange(
        shape[0] * shape[1] * shape[2],
        dtype=torch.float32,
    ).reshape(1, *shape)
    return tio.Subject(t1=tio.ScalarImage(data))


def _make_subject_with_label(
    shape: tuple[int, int, int] = (20, 20, 20),
) -> tio.Subject:
    data = torch.rand(1, *shape)
    label = torch.zeros(1, *shape, dtype=torch.float32)
    label[0, 8:12, 8:12, 8:12] = 1
    return tio.Subject(
        t1=tio.ScalarImage(data),
        seg=tio.LabelMap(label),
    )


# ---------------------------------------------------------------------------
# PatchLocation
# ---------------------------------------------------------------------------


class TestPatchLocation:
    def test_index_fin(self) -> None:
        loc = PatchLocation(index=(0, 0, 0), size=(10, 10, 10))
        assert loc.index_fin == (10, 10, 10)

    def test_to_slices(self) -> None:
        loc = PatchLocation(index=(5, 10, 15), size=(3, 4, 5))
        si, sj, sk = loc.to_slices()
        assert si == slice(5, 8)
        assert sj == slice(10, 14)
        assert sk == slice(15, 20)

    def test_scaled(self) -> None:
        loc = PatchLocation(index=(10, 20, 30), size=(8, 8, 8))
        scaled = loc.scaled((0.5, 0.5, 0.5))
        assert scaled.index == (5, 10, 15)
        assert scaled.size == (4, 4, 4)


# ---------------------------------------------------------------------------
# GridSampler
# ---------------------------------------------------------------------------


class TestGridSampler:
    def test_no_overlap(self) -> None:
        subject = _make_subject((20, 20, 20))
        sampler = tio.GridSampler(subject, patch_size=10)
        assert len(sampler) == 8  # 2x2x2 grid
        patch = sampler[0]
        assert patch.t1.spatial_shape == (10, 10, 10)

    def test_with_overlap(self) -> None:
        subject = _make_subject((20, 20, 20))
        sampler = tio.GridSampler(subject, patch_size=12, patch_overlap=4)
        assert len(sampler) > 1
        for i in range(len(sampler)):
            assert sampler[i].t1.spatial_shape == (12, 12, 12)

    def test_with_padding(self) -> None:
        subject = _make_subject((20, 20, 20))
        sampler = tio.GridSampler(
            subject,
            patch_size=12,
            patch_overlap=4,
            padding_mode="constant",
        )
        assert len(sampler) > 1

    def test_patch_has_location(self) -> None:
        subject = _make_subject((20, 20, 20))
        sampler = tio.GridSampler(subject, patch_size=10)
        patch = sampler[0]
        loc = patch.patch_location
        assert isinstance(loc, PatchLocation)

    def test_covers_volume(self) -> None:
        """All voxels should be covered by at least one patch."""
        subject = _make_subject((15, 15, 15))
        sampler = tio.GridSampler(subject, patch_size=10)
        covered = torch.zeros(15, 15, 15)
        for i in range(len(sampler)):
            loc = sampler[i].patch_location
            si, sj, sk = loc.to_slices()
            covered[si, sj, sk] = 1
        assert covered.all()

    def test_works_with_dataloader(self) -> None:
        from torchio.loader import SubjectsLoader

        subject = _make_subject((20, 20, 20))
        sampler = tio.GridSampler(subject, patch_size=10)
        loader = SubjectsLoader(sampler, batch_size=4)
        total = 0
        for batch in loader:
            total += batch.batch_size
        assert total == 8


# ---------------------------------------------------------------------------
# UniformSampler
# ---------------------------------------------------------------------------


class TestUniformSampler:
    def test_yields_correct_count(self) -> None:
        subject = _make_subject((20, 20, 20))
        sampler = tio.UniformSampler(subject, patch_size=8, num_patches=5)
        patches = list(sampler)
        assert len(patches) == 5

    def test_correct_shape(self) -> None:
        subject = _make_subject((20, 20, 20))
        sampler = tio.UniformSampler(subject, patch_size=8, num_patches=1)
        patch = next(iter(sampler))
        assert patch.t1.spatial_shape == (8, 8, 8)

    def test_patches_vary(self) -> None:
        torch.manual_seed(42)
        subject = _make_subject((50, 50, 50))
        sampler = tio.UniformSampler(subject, patch_size=8, num_patches=3)
        locations = [p.patch_location.index for p in sampler]
        assert len(set(locations)) > 1

    def test_works_with_dataloader(self) -> None:
        from torchio.loader import SubjectsLoader

        subject = _make_subject((20, 20, 20))
        sampler = tio.UniformSampler(subject, patch_size=8, num_patches=10)
        loader = SubjectsLoader(sampler, batch_size=4)
        total = sum(batch.batch_size for batch in loader)
        assert total == 10


# ---------------------------------------------------------------------------
# WeightedSampler
# ---------------------------------------------------------------------------


class TestWeightedSampler:
    def test_samples_from_high_probability(self) -> None:
        subject = _make_subject_with_label()
        sampler = tio.WeightedSampler(
            subject,
            patch_size=4,
            probability_map="seg",
            num_patches=10,
        )
        patches = list(sampler)
        assert len(patches) == 10
        for p in patches:
            assert p.t1.spatial_shape == (4, 4, 4)


# ---------------------------------------------------------------------------
# LabelSampler
# ---------------------------------------------------------------------------


class TestLabelSampler:
    def test_samples_near_labels(self) -> None:
        subject = _make_subject_with_label()
        sampler = tio.LabelSampler(
            subject,
            patch_size=4,
            label_name="seg",
            num_patches=10,
        )
        patches = list(sampler)
        assert len(patches) == 10

    def test_custom_probabilities(self) -> None:
        subject = _make_subject_with_label()
        sampler = tio.LabelSampler(
            subject,
            patch_size=4,
            label_name="seg",
            label_probabilities={0: 0.0, 1: 1.0},
            num_patches=5,
        )
        patches = list(sampler)
        assert len(patches) == 5


# ---------------------------------------------------------------------------
# PatchAggregator
# ---------------------------------------------------------------------------


class TestAggregatorCrop:
    def test_reconstruct_identity(self) -> None:
        """Crop mode with no overlap should perfectly reconstruct."""
        subject = _make_subject((20, 20, 20))
        sampler = tio.GridSampler(subject, patch_size=10)
        aggregator = tio.PatchAggregator(
            spatial_shape=(20, 20, 20),
            overlap_mode="crop",
        )
        for i in range(len(sampler)):
            patch = sampler[i]
            loc = patch.patch_location
            aggregator.add_batch(
                patch.t1.data.unsqueeze(0),
                [loc],
            )
        output = aggregator.get_output()
        torch.testing.assert_close(output, subject.t1.data)

    def test_with_overlap(self) -> None:
        subject = _make_subject((20, 20, 20))
        sampler = tio.GridSampler(subject, patch_size=12, patch_overlap=4)
        aggregator = tio.PatchAggregator(
            spatial_shape=(20, 20, 20),
            overlap_mode="crop",
            patch_overlap=4,
        )
        for i in range(len(sampler)):
            patch = sampler[i]
            loc = patch.patch_location
            aggregator.add_batch(
                patch.t1.data.unsqueeze(0),
                [loc],
            )
        output = aggregator.get_output()
        assert output.shape == (1, 20, 20, 20)


class TestAggregatorAverage:
    def test_average_mode(self) -> None:
        subject = _make_subject((20, 20, 20))
        sampler = tio.GridSampler(subject, patch_size=12, patch_overlap=4)
        aggregator = tio.PatchAggregator(
            spatial_shape=(20, 20, 20),
            overlap_mode="average",
        )
        for i in range(len(sampler)):
            patch = sampler[i]
            loc = patch.patch_location
            aggregator.add_batch(
                patch.t1.data.unsqueeze(0),
                [loc],
            )
        output = aggregator.get_output()
        assert output.shape == (1, 20, 20, 20)


class TestAggregatorHann:
    def test_hann_mode(self) -> None:
        subject = _make_subject((20, 20, 20))
        sampler = tio.GridSampler(subject, patch_size=12, patch_overlap=4)
        aggregator = tio.PatchAggregator(
            spatial_shape=(20, 20, 20),
            overlap_mode="hann",
        )
        for i in range(len(sampler)):
            patch = sampler[i]
            loc = patch.patch_location
            aggregator.add_batch(
                patch.t1.data.unsqueeze(0),
                [loc],
            )
        output = aggregator.get_output()
        assert output.shape == (1, 20, 20, 20)


class TestAggregatorOutputShape:
    def test_downsampled_output(self) -> None:
        """Aggregator with output smaller than input."""
        aggregator = tio.PatchAggregator(
            spatial_shape=(20, 20, 20),
            overlap_mode="average",
            output_shape=(10, 10, 10),
        )
        loc = PatchLocation(index=(0, 0, 0), size=(20, 20, 20))
        patch = torch.rand(1, 1, 10, 10, 10)
        aggregator.add_batch(patch, [loc])
        output = aggregator.get_output()
        assert output.shape == (1, 10, 10, 10)


class TestAggregatorMultiKey:
    def test_dict_output(self) -> None:
        aggregator = tio.PatchAggregator(
            spatial_shape=(10, 10, 10),
            overlap_mode="average",
        )
        loc = PatchLocation(index=(0, 0, 0), size=(10, 10, 10))
        seg = torch.rand(1, 2, 10, 10, 10)
        emb = torch.rand(1, 64, 10, 10, 10)
        aggregator.add_batch({"seg": seg, "emb": emb}, [loc])
        assert aggregator.get_output("seg").shape == (2, 10, 10, 10)
        assert aggregator.get_output("emb").shape == (64, 10, 10, 10)


class TestAggregatorValidation:
    def test_invalid_mode(self) -> None:
        import pytest

        with pytest.raises(ValueError, match="overlap_mode"):
            tio.PatchAggregator(
                spatial_shape=(10, 10, 10),
                overlap_mode="invalid",
            )
