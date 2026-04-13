"""Tests for LabelsToImage transform."""

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


class TestLabelsToImage:
    def test_generates_image(self) -> None:
        subject = _make_subject()
        result = tio.LabelsToImage(label_key="seg")(subject)
        assert "image_from_labels" in result
        assert result.image_from_labels.data.shape[1:] == (10, 10, 10)

    def test_custom_key(self) -> None:
        subject = _make_subject()
        result = tio.LabelsToImage(label_key="seg", image_key="synth")(subject)
        assert "synth" in result

    def test_auto_detect_label(self) -> None:
        subject = _make_subject()
        result = tio.LabelsToImage()(subject)
        assert "image_from_labels" in result

    def test_ignore_background(self) -> None:
        subject = _make_subject()
        result = tio.LabelsToImage(
            label_key="seg",
            ignore_background=True,
        )(subject)
        bg_mask = subject.seg.data == 0
        bg_values = result.image_from_labels.data[0, bg_mask[0]]
        assert bg_values.abs().max() < 1e-5

    def test_no_label_raises(self) -> None:
        subject = _make_subject(with_label=False)
        with pytest.raises(KeyError, match="No LabelMap"):
            tio.LabelsToImage()(subject)

    def test_missing_key_raises(self) -> None:
        subject = _make_subject()
        with pytest.raises(KeyError, match="nope"):
            tio.LabelsToImage(label_key="nope")(subject)
