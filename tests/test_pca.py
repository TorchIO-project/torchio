"""Tests for PCA transform."""

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


class TestPCA:
    def test_reduces_channels(self) -> None:
        data = torch.rand(8, 10, 10, 10)
        subject = tio.Subject(emb=tio.ScalarImage(data))
        result = tio.PCA(num_components=3)(subject)
        assert result.emb.data.shape[0] == 3

    def test_output_range(self) -> None:
        data = torch.randn(16, 10, 10, 10)
        subject = tio.Subject(emb=tio.ScalarImage(data))
        result = tio.PCA(num_components=3, clip=True)(subject)
        assert result.emb.data.min() >= 0.0
        assert result.emb.data.max() <= 1.0

    def test_too_few_channels_raises(self) -> None:
        data = torch.rand(2, 10, 10, 10)
        subject = tio.Subject(emb=tio.ScalarImage(data))
        with pytest.raises(ValueError, match="channels"):
            tio.PCA(num_components=5)(subject)

    def test_invalid_num_components_raises(self) -> None:
        with pytest.raises(ValueError, match="num_components"):
            tio.PCA(num_components=0)

    def test_no_whitening(self) -> None:
        data = torch.randn(8, 10, 10, 10)
        subject = tio.Subject(emb=tio.ScalarImage(data))
        result = tio.PCA(num_components=3, whiten=False, normalize=False)(subject)
        assert result.emb.data.shape[0] == 3
