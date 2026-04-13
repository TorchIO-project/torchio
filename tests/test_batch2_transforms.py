"""Tests for batch 2 transforms (label, artifact, spatial, other)."""

from __future__ import annotations

import pytest
import torch

import torchio as tio


def _make_subject(with_label: bool = True) -> tio.Subject:
    data = torch.rand(1, 10, 10, 10) * 100
    kwargs: dict = {"t1": tio.ScalarImage(data)}
    if with_label:
        seg = torch.zeros(1, 10, 10, 10, dtype=torch.float32)
        seg[0, 2:5, 2:5, 2:5] = 1
        seg[0, 6:9, 6:9, 6:9] = 2
        kwargs["seg"] = tio.LabelMap(seg)
    return tio.Subject(**kwargs)


# ── Contour ──────────────────────────────────────────────────────────


class TestContour:
    def test_basic_contour(self) -> None:
        subject = _make_subject()
        result = tio.Contour()(subject)
        # Contour should be binary (0 or 1)
        unique = result.seg.data.unique().tolist()
        assert set(unique) <= {0.0, 1.0}

    def test_solid_block_has_boundary(self) -> None:
        seg = torch.zeros(1, 10, 10, 10, dtype=torch.float32)
        seg[0, 3:7, 3:7, 3:7] = 1
        subject = tio.Subject(seg=tio.LabelMap(seg))
        result = tio.Contour()(subject)
        # Interior voxels (not touching boundary) should be 0
        assert result.seg.data[0, 4, 5, 5] == 0
        # Edge voxels should be 1
        assert result.seg.data[0, 3, 5, 5] == 1

    def test_uniform_label_no_contour(self) -> None:
        seg = torch.ones(1, 10, 10, 10, dtype=torch.float32)
        subject = tio.Subject(seg=tio.LabelMap(seg))
        result = tio.Contour()(subject)
        # Only the border voxels (touching padded -1) are contour.
        # Interior voxels away from edges should be 0.
        assert result.seg.data[0, 4, 4, 4] == 0

    def test_leaves_scalar_unchanged(self) -> None:
        subject = _make_subject()
        original = subject.t1.data.clone()
        result = tio.Contour()(subject)
        torch.testing.assert_close(result.t1.data, original)


# ── RemapLabels ──────────────────────────────────────────────────────


class TestRemapLabels:
    def test_basic_remap(self) -> None:
        subject = _make_subject()
        result = tio.RemapLabels({1: 10, 2: 20})(subject)
        assert 10 in result.seg.data.unique().tolist()
        assert 20 in result.seg.data.unique().tolist()
        assert 1 not in result.seg.data.unique().tolist()
        assert 2 not in result.seg.data.unique().tolist()

    def test_merge_labels(self) -> None:
        subject = _make_subject()
        result = tio.RemapLabels({2: 1})(subject)
        assert 2 not in result.seg.data.unique().tolist()
        assert 1 in result.seg.data.unique().tolist()

    def test_swap_labels(self) -> None:
        subject = _make_subject()
        original_1_count = (subject.seg.data == 1).sum().item()
        original_2_count = (subject.seg.data == 2).sum().item()
        result = tio.RemapLabels({1: 2, 2: 1})(subject)
        assert (result.seg.data == 1).sum().item() == original_2_count
        assert (result.seg.data == 2).sum().item() == original_1_count

    def test_leaves_unlisted_labels(self) -> None:
        subject = _make_subject()
        original_0_count = (subject.seg.data == 0).sum().item()
        result = tio.RemapLabels({1: 10})(subject)
        assert (result.seg.data == 0).sum().item() == original_0_count

    def test_inverse(self) -> None:
        subject = _make_subject()
        original = subject.seg.data.clone()
        transformed = tio.RemapLabels({1: 10, 2: 20})(subject)
        restored = transformed.apply_inverse_transform()
        torch.testing.assert_close(restored.seg.data, original)

    def test_leaves_scalar_unchanged(self) -> None:
        subject = _make_subject()
        original = subject.t1.data.clone()
        result = tio.RemapLabels({1: 10})(subject)
        torch.testing.assert_close(result.t1.data, original)


# ── SequentialLabels ─────────────────────────────────────────────────


class TestSequentialLabels:
    def test_basic_sequential(self) -> None:
        seg = torch.zeros(1, 5, 5, 5, dtype=torch.float32)
        seg[0, 0:2, :, :] = 5
        seg[0, 3:5, :, :] = 10
        subject = tio.Subject(seg=tio.LabelMap(seg))
        result = tio.SequentialLabels()(subject)
        unique = sorted(result.seg.data.unique().tolist())
        assert unique == [0.0, 1.0, 2.0]

    def test_already_sequential(self) -> None:
        subject = _make_subject()
        result = tio.SequentialLabels()(subject)
        # {0, 1, 2} is already sequential, should be unchanged
        torch.testing.assert_close(result.seg.data, subject.seg.data)

    def test_inverse(self) -> None:
        seg = torch.zeros(1, 5, 5, 5, dtype=torch.float32)
        seg[0, 0:2, :, :] = 5
        seg[0, 3:5, :, :] = 10
        subject = tio.Subject(
            t1=tio.ScalarImage(torch.rand(1, 5, 5, 5)),
            seg=tio.LabelMap(seg),
        )
        original = subject.seg.data.clone()
        transformed = tio.SequentialLabels()(subject)
        restored = transformed.apply_inverse_transform()
        torch.testing.assert_close(restored.seg.data, original)

    def test_leaves_scalar_unchanged(self) -> None:
        subject = _make_subject()
        original = subject.t1.data.clone()
        result = tio.SequentialLabels()(subject)
        torch.testing.assert_close(result.t1.data, original)


# ── Lambda ───────────────────────────────────────────────────────────


class TestLambda:
    def test_double(self) -> None:
        subject = _make_subject(with_label=False)
        original = subject.t1.data.clone()
        result = tio.Lambda(lambda x: 2 * x)(subject)
        torch.testing.assert_close(result.t1.data, 2 * original)

    def test_scalar_only(self) -> None:
        subject = _make_subject()
        original_seg = subject.seg.data.clone()
        result = tio.Lambda(lambda x: x * 0, types_to_apply="scalar")(subject)
        # Scalar should be zeroed
        assert result.t1.data.sum() == 0
        # Label should be untouched
        torch.testing.assert_close(result.seg.data, original_seg)

    def test_label_only(self) -> None:
        subject = _make_subject()
        original_t1 = subject.t1.data.clone()
        result = tio.Lambda(lambda x: x * 0, types_to_apply="label")(subject)
        # Label should be zeroed
        assert result.seg.data.sum() == 0
        # Scalar should be untouched
        torch.testing.assert_close(result.t1.data, original_t1)

    def test_not_callable_raises(self) -> None:
        with pytest.raises(TypeError, match="callable"):
            tio.Lambda(42)  # type: ignore[arg-type]

    def test_unknown_types_to_apply_applies_all(self) -> None:
        """Unknown types_to_apply falls through to apply to all images."""
        subject = _make_subject()
        result = tio.Lambda(lambda x: x * 0, types_to_apply="unknown")(subject)
        assert result.t1.data.sum() == 0
        assert result.seg.data.sum() == 0


# ── Resize ───────────────────────────────────────────────────────────


class TestResize:
    def test_resize_to_target(self) -> None:
        subject = _make_subject(with_label=False)
        result = tio.Resize(5)(subject)
        assert result.t1.data.shape[1:] == (5, 5, 5)

    def test_resize_anisotropic(self) -> None:
        subject = _make_subject(with_label=False)
        result = tio.Resize((8, 6, 4))(subject)
        assert result.t1.data.shape[1:] == (8, 6, 4)

    def test_resize_preserves_dtype(self) -> None:
        subject = _make_subject()
        result = tio.Resize(5)(subject)
        assert result.t1.data.dtype == subject.t1.data.dtype

    def test_resize_labels_nearest(self) -> None:
        subject = _make_subject()
        result = tio.Resize(5)(subject)
        # Label values should still be integer-like
        unique = result.seg.data.unique().tolist()
        for v in unique:
            assert v == int(v)

    def test_resize_with_labels(self) -> None:
        subject = _make_subject()
        result = tio.Resize(5)(subject)
        assert result.seg.data.shape[1:] == (5, 5, 5)


# ── Spike ────────────────────────────────────────────────────────────


class TestSpike:
    def test_changes_data(self) -> None:
        subject = _make_subject(with_label=False)
        original = subject.t1.data.clone()
        result = tio.Spike(num_spikes=3, intensity=2.0)(subject)
        assert not torch.allclose(result.t1.data, original)

    def test_zero_intensity_is_identity(self) -> None:
        subject = _make_subject(with_label=False)
        original = subject.t1.data.clone()
        result = tio.Spike(intensity=0.0)(subject)
        torch.testing.assert_close(result.t1.data, original)

    def test_leaves_labels_unchanged(self) -> None:
        subject = _make_subject()
        original_seg = subject.seg.data.clone()
        result = tio.Spike()(subject)
        torch.testing.assert_close(result.seg.data, original_seg)

    def test_single_spike(self) -> None:
        subject = _make_subject(with_label=False)
        original = subject.t1.data.clone()
        result = tio.Spike(num_spikes=1, intensity=1.0)(subject)
        assert not torch.allclose(result.t1.data, original)


# ── Ghosting ─────────────────────────────────────────────────────────


class TestGhosting:
    def test_changes_data(self) -> None:
        subject = _make_subject(with_label=False)
        original = subject.t1.data.clone()
        result = tio.Ghosting(num_ghosts=5, intensity=0.8)(subject)
        assert not torch.allclose(result.t1.data, original)

    def test_zero_intensity_is_identity(self) -> None:
        subject = _make_subject(with_label=False)
        original = subject.t1.data.clone()
        result = tio.Ghosting(intensity=0.0)(subject)
        torch.testing.assert_close(result.t1.data, original)

    def test_leaves_labels_unchanged(self) -> None:
        subject = _make_subject()
        original_seg = subject.seg.data.clone()
        result = tio.Ghosting()(subject)
        torch.testing.assert_close(result.seg.data, original_seg)

    def test_specific_axis(self) -> None:
        subject = _make_subject(with_label=False)
        original = subject.t1.data.clone()
        result = tio.Ghosting(axes=(1,), intensity=0.8)(subject)
        assert not torch.allclose(result.t1.data, original)

    def test_restore_fraction(self) -> None:
        subject = _make_subject(with_label=False)
        result = tio.Ghosting(restore=0.2, intensity=0.8)(subject)
        assert result.t1.data.shape == subject.t1.data.shape


# ── Motion ───────────────────────────────────────────────────────────


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


# ── Exports ──────────────────────────────────────────────────────────


class TestBatch2Exports:
    @pytest.mark.parametrize(
        "name",
        [
            "Contour",
            "Ghosting",
            "Lambda",
            "Motion",
            "RemapLabels",
            "Resize",
            "SequentialLabels",
            "Spike",
        ],
    )
    def test_top_level_export(self, name: str) -> None:
        assert hasattr(tio, name)
