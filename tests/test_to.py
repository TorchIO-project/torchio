"""Tests for To transform."""

from __future__ import annotations

import torch

import torchio as tio


def _make_subject() -> tio.Subject:
    return tio.Subject(
        t1=tio.ScalarImage(torch.rand(1, 5, 5, 5)),
    )


class TestTo:
    def test_cast_dtype(self) -> None:
        subject = _make_subject()
        result = tio.To(torch.float64)(subject)
        assert result.t1.data.dtype == torch.float64

    def test_cast_to_half(self) -> None:
        subject = _make_subject()
        result = tio.To(torch.float16)(subject)
        assert result.t1.data.dtype == torch.float16

    def test_device_cpu(self) -> None:
        subject = _make_subject()
        result = tio.To("cpu")(subject)
        assert result.t1.data.device.type == "cpu"

    def test_in_compose(self) -> None:
        subject = _make_subject()
        pipeline = tio.Compose(
            [
                tio.To(torch.float64),
                tio.Gamma(log_gamma=0.0),
            ]
        )
        result = pipeline(subject)
        assert result.t1.data.shape == subject.t1.data.shape
