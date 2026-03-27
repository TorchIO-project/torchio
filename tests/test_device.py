"""Tests for .to() device methods on Image and Subject, and To transform."""

from __future__ import annotations

import pytest
import torch

import torchio as tio
from torchio.data.bboxes import BoundingBoxFormat
from torchio.data.bboxes import BoundingBoxes
from torchio.data.points import Points

HAS_CUDA = torch.cuda.is_available()
HAS_MPS = torch.backends.mps.is_available()


class TestImageTo:
    def test_to_returns_self(self) -> None:
        image = tio.ScalarImage.from_tensor(torch.rand(1, 4, 4, 4))
        result = image.to("cpu")
        assert result is image

    def test_device_property(self) -> None:
        image = tio.ScalarImage.from_tensor(torch.rand(1, 4, 4, 4))
        assert image.device == torch.device("cpu")

    def test_to_dtype(self) -> None:
        image = tio.ScalarImage.from_tensor(torch.rand(1, 4, 4, 4))
        result = image.to(torch.float16)
        assert result.data.dtype == torch.float16

    @pytest.mark.skipif(not HAS_CUDA, reason="No CUDA")
    def test_to_cuda(self) -> None:
        image = tio.ScalarImage.from_tensor(torch.rand(1, 4, 4, 4))
        result = image.to("cuda")
        assert result.device.type == "cuda"
        assert result.data.is_cuda

    @pytest.mark.skipif(not HAS_MPS, reason="No MPS")
    def test_to_mps(self) -> None:
        image = tio.ScalarImage.from_tensor(torch.rand(1, 4, 4, 4))
        result = image.to("mps")
        assert result.device.type == "mps"

    @pytest.mark.skipif(not HAS_MPS, reason="No MPS")
    def test_mps_round_trip(self) -> None:
        image = tio.ScalarImage.from_tensor(torch.rand(1, 4, 4, 4))
        original = image.data.clone()
        image.to("mps").to("cpu")
        torch.testing.assert_close(image.data, original)


class TestSubjectTo:
    def test_to_returns_self(self) -> None:
        subject = tio.Subject(
            t1=tio.ScalarImage.from_tensor(torch.rand(1, 4, 4, 4)),
        )
        result = subject.to("cpu")
        assert result is subject

    def test_moves_all_images(self) -> None:
        subject = tio.Subject(
            t1=tio.ScalarImage.from_tensor(torch.rand(1, 4, 4, 4)),
            seg=tio.LabelMap.from_tensor(torch.randint(0, 3, (1, 4, 4, 4))),
        )
        result = subject.to(torch.float64)
        assert result.t1.data.dtype == torch.float64
        assert result.seg.data.dtype == torch.float64

    def test_moves_points(self) -> None:
        subject = tio.Subject(
            t1=tio.ScalarImage.from_tensor(torch.rand(1, 4, 4, 4)),
            pts=Points(torch.rand(3, 3)),
        )
        result = subject.to(torch.float64)
        assert result.pts.data.dtype == torch.float64

    def test_moves_bboxes(self) -> None:
        subject = tio.Subject(
            t1=tio.ScalarImage.from_tensor(torch.rand(1, 4, 4, 4)),
            boxes=BoundingBoxes(
                torch.rand(2, 6),
                format=BoundingBoxFormat.IJKIJK,
            ),
        )
        result = subject.to(torch.float64)
        assert result.boxes.data.dtype == torch.float64

    @pytest.mark.skipif(not HAS_CUDA, reason="No CUDA")
    def test_to_cuda(self) -> None:
        subject = tio.Subject(
            t1=tio.ScalarImage.from_tensor(torch.rand(1, 4, 4, 4)),
        )
        result = subject.to("cuda")
        assert result.t1.data.is_cuda

    @pytest.mark.skipif(not HAS_MPS, reason="No MPS")
    def test_to_mps(self) -> None:
        subject = tio.Subject(
            t1=tio.ScalarImage.from_tensor(torch.rand(1, 4, 4, 4)),
            pts=Points(torch.rand(3, 3)),
        )
        result = subject.to("mps")
        assert result.t1.device.type == "mps"
        assert result.pts.device.type == "mps"
        assert result.device.type == "mps"


class TestToTransform:
    def test_to_dtype(self) -> None:
        subject = tio.Subject(
            t1=tio.ScalarImage.from_tensor(torch.rand(1, 4, 4, 4)),
        )
        transform = tio.To(torch.float64)
        result = transform(subject)
        assert result.t1.data.dtype == torch.float64

    def test_to_device_str(self) -> None:
        subject = tio.Subject(
            t1=tio.ScalarImage.from_tensor(torch.rand(1, 4, 4, 4)),
        )
        transform = tio.To("cpu")
        result = transform(subject)
        assert result.t1.device == torch.device("cpu")

    def test_history_recorded(self) -> None:
        subject = tio.Subject(
            t1=tio.ScalarImage.from_tensor(torch.rand(1, 4, 4, 4)),
        )
        transform = tio.To(torch.float64)
        result = transform(subject)
        assert len(result.applied_transforms) == 1
        assert result.applied_transforms[0].name == "To"

    def test_accepts_image(self) -> None:
        image = tio.ScalarImage.from_tensor(torch.rand(1, 4, 4, 4))
        result = tio.To(torch.float64)(image)
        assert isinstance(result, tio.Image)
        assert result.data.dtype == torch.float64

    def test_accepts_tensor(self) -> None:
        tensor = torch.rand(1, 4, 4, 4)
        result = tio.To(torch.float64)(tensor)
        assert isinstance(result, torch.Tensor)
        assert result.dtype == torch.float64

    @pytest.mark.skipif(not HAS_MPS, reason="No MPS")
    def test_to_mps_via_transform(self) -> None:
        subject = tio.Subject(
            t1=tio.ScalarImage.from_tensor(torch.rand(1, 4, 4, 4)),
        )
        result = tio.To("mps")(subject)
        assert result.t1.device.type == "mps"
