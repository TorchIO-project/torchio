"""Tests for Anisotropy transform."""

from __future__ import annotations

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


class TestAnisotropy:
    def test_changes_data(self) -> None:
        subject = _make_subject(with_label=False)
        original = subject.t1.data.clone()
        result = tio.Anisotropy(downsampling=3.0)(subject)
        assert not torch.allclose(result.t1.data, original)

    def test_preserves_shape(self) -> None:
        subject = _make_subject(with_label=False)
        result = tio.Anisotropy(downsampling=2.0)(subject)
        assert result.t1.data.shape == subject.t1.data.shape

    def test_specific_axis(self) -> None:
        subject = _make_subject(with_label=False)
        original = subject.t1.data.clone()
        result = tio.Anisotropy(axes=(0,), downsampling=3.0)(subject)
        assert not torch.allclose(result.t1.data, original)

    def test_labels_use_nearest(self) -> None:
        subject = _make_subject()
        result = tio.Anisotropy(downsampling=2.0)(subject)
        unique = result.seg.data.unique().tolist()
        for v in unique:
            assert v == int(v)

    def test_factor_one_is_identity(self) -> None:
        subject = _make_subject(with_label=False)
        original = subject.t1.data.clone()
        result = tio.Anisotropy(downsampling=1.0)(subject)
        torch.testing.assert_close(result.t1.data, original)
