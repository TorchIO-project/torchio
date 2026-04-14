"""Tests for CornucopiaAdapter transform."""

from __future__ import annotations

import cornucopia as cc
import pytest
import torch

import torchio as tio


def _make_subject() -> tio.Subject:
    return tio.Subject(
        t1=tio.ScalarImage(torch.rand(1, 8, 8, 8) * 100),
        seg=tio.LabelMap(torch.zeros(1, 8, 8, 8)),
    )


# ── Adapter logic ────────────────────────────────────────────────────


class TestCornucopiaAdapterLogic:
    def test_not_callable_raises(self) -> None:
        with pytest.raises(TypeError, match="callable"):
            tio.CornucopiaAdapter(42)  # type: ignore[arg-type]

    def test_p_zero_is_identity(self) -> None:
        subject = _make_subject()
        original = subject.t1.data.clone()
        result = tio.CornucopiaAdapter(cc.GaussianNoiseTransform(), p=0.0)(subject)
        torch.testing.assert_close(result.t1.data, original)

    def test_include_filter(self) -> None:
        subject = _make_subject()
        original_seg = subject.seg.data.clone()
        result = tio.CornucopiaAdapter(cc.GaussianNoiseTransform(), include=["t1"])(
            subject
        )
        torch.testing.assert_close(result.seg.data, original_seg)

    def test_exclude_filter(self) -> None:
        subject = _make_subject()
        original_seg = subject.seg.data.clone()
        result = tio.CornucopiaAdapter(cc.GaussianNoiseTransform(), exclude=["seg"])(
            subject
        )
        torch.testing.assert_close(result.seg.data, original_seg)

    def test_scalar_images_come_first(self) -> None:
        """Scalars are passed before labels to the callable."""
        received: list[str] = []

        def spy(*tensors: torch.Tensor) -> tuple[torch.Tensor, ...]:
            for t in tensors:
                received.append("scalar" if t.sum() > 0 else "label")
            return tensors

        subject = _make_subject()
        tio.CornucopiaAdapter(spy)(subject)
        assert received[0] == "scalar"
        assert received[1] == "label"

    def test_not_invertible(self) -> None:
        adapter = tio.CornucopiaAdapter(cc.FlipTransform())
        assert adapter.invertible is False

    def test_no_history_recorded(self) -> None:
        subject = _make_subject()
        result = tio.CornucopiaAdapter(cc.FlipTransform())(subject)
        for at in result.applied_transforms:
            assert not isinstance(at, tio.CornucopiaAdapter)

    def test_in_compose(self) -> None:
        subject = _make_subject()
        pipeline = tio.Compose(
            [
                tio.CornucopiaAdapter(cc.FlipTransform()),
                tio.Gamma(log_gamma=0.0),
            ]
        )
        result = pipeline(subject)
        assert result.t1.data.shape == subject.t1.data.shape


# ── Real Cornucopia transforms ───────────────────────────────────────


class TestCornucopiaAdapterTransforms:
    def test_gaussian_noise(self) -> None:
        subject = tio.Subject(
            t1=tio.ScalarImage(torch.rand(1, 16, 16, 16)),
        )
        original = subject.t1.data.clone()
        result = tio.CornucopiaAdapter(cc.GaussianNoiseTransform())(subject)
        assert not torch.allclose(result.t1.data, original)

    def test_flip(self) -> None:
        subject = _make_subject()
        result = tio.CornucopiaAdapter(cc.FlipTransform())(subject)
        assert result.t1.data.shape == subject.t1.data.shape

    def test_gamma(self) -> None:
        subject = tio.Subject(
            t1=tio.ScalarImage(torch.rand(1, 8, 8, 8).clamp(0.01, 1)),
        )
        original = subject.t1.data.clone()
        result = tio.CornucopiaAdapter(cc.GammaTransform())(subject)
        assert result.t1.data.shape == original.shape

    def test_elastic_shared(self) -> None:
        """Elastic deformation is shared across image and label."""
        subject = _make_subject()
        result = tio.CornucopiaAdapter(cc.ElasticTransform())(subject)
        assert result.t1.data.shape == subject.t1.data.shape
        assert result.seg.data.shape == subject.seg.data.shape

    def test_affine(self) -> None:
        subject = _make_subject()
        result = tio.CornucopiaAdapter(cc.AffineTransform())(subject)
        assert result.t1.data.shape == subject.t1.data.shape

    def test_sequential(self) -> None:
        """Cornucopia SequentialTransform (compose via +)."""
        seq = cc.GaussianNoiseTransform() + cc.FlipTransform()
        subject = _make_subject()
        result = tio.CornucopiaAdapter(seq)(subject)
        assert result.t1.data.shape == subject.t1.data.shape


# ── Export ───────────────────────────────────────────────────────────


class TestCornucopiaExport:
    def test_top_level(self) -> None:
        assert hasattr(tio, "CornucopiaAdapter")
