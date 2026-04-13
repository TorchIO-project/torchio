"""Tests for Clamp transform."""

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


class TestClamp:
    def test_clamps_values(self) -> None:
        data = torch.tensor([-10.0, 0.0, 50.0, 200.0]).reshape(1, 1, 1, 4)
        subject = tio.Subject(t1=tio.ScalarImage(data))
        result = tio.Clamp(out_min=0, out_max=100)(subject)
        assert result.t1.data.min() >= 0
        assert result.t1.data.max() <= 100

    def test_clamp_min_only(self) -> None:
        data = torch.tensor([-5.0, 0.0, 5.0]).reshape(1, 1, 1, 3)
        subject = tio.Subject(t1=tio.ScalarImage(data))
        result = tio.Clamp(out_min=0)(subject)
        assert result.t1.data.min() >= 0

    def test_invalid_range_raises(self) -> None:
        with pytest.raises(ValueError, match="out_min"):
            tio.Clamp(out_min=100, out_max=0)
