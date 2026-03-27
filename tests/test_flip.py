"""Tests for the Flip spatial transform."""

from __future__ import annotations

import pytest
import torch

import torchio as tio
from torchio.data.bboxes import BoundingBoxFormat
from torchio.data.bboxes import BoundingBoxes
from torchio.data.points import Points

HAS_MPS = torch.backends.mps.is_available()


class TestFlip:
    def test_flip_axis_0(self) -> None:
        tensor = torch.arange(8).reshape(1, 2, 2, 2).float()
        image = tio.ScalarImage.from_tensor(tensor)
        subject = tio.Subject(t1=image)
        result = tio.Flip(axes=(0,))(subject)
        expected = torch.flip(tensor, [1])
        torch.testing.assert_close(result.t1.data, expected)

    def test_flip_multiple_axes(self) -> None:
        tensor = torch.arange(8).reshape(1, 2, 2, 2).float()
        image = tio.ScalarImage.from_tensor(tensor)
        subject = tio.Subject(t1=image)
        result = tio.Flip(axes=(0, 1))(subject)
        expected = torch.flip(tensor, [1, 2])
        torch.testing.assert_close(result.t1.data, expected)

    def test_flip_all_images(self) -> None:
        subject = tio.Subject(
            t1=tio.ScalarImage.from_tensor(torch.rand(1, 4, 4, 4)),
            seg=tio.LabelMap.from_tensor(torch.randint(0, 3, (1, 4, 4, 4))),
        )
        original_t1 = subject.t1.data.clone()
        original_seg = subject.seg.data.clone()
        result = tio.Flip(axes=(2,))(subject)
        # Both should be flipped
        assert not torch.equal(result.t1.data, original_t1)
        assert not torch.equal(result.seg.data, original_seg)

    def test_flip_is_self_inverse(self) -> None:
        tensor = torch.rand(1, 4, 5, 6)
        image = tio.ScalarImage.from_tensor(tensor.clone())
        subject = tio.Subject(t1=image)
        flip = tio.Flip(axes=(0, 1, 2))
        result = flip(flip(subject))
        torch.testing.assert_close(result.t1.data, tensor)

    def test_flip_with_probability(self) -> None:
        subject = tio.Subject(
            t1=tio.ScalarImage.from_tensor(torch.rand(1, 4, 4, 4)),
        )
        original = subject.t1.data.clone()
        result = tio.Flip(axes=(0,), p=0.0)(subject)
        torch.testing.assert_close(result.t1.data, original)

    def test_flip_history_recorded(self) -> None:
        subject = tio.Subject(
            t1=tio.ScalarImage.from_tensor(torch.rand(1, 4, 4, 4)),
        )
        result = tio.Flip(axes=(1,))(subject)
        assert len(result.applied_transforms) == 1
        trace = result.applied_transforms[0]
        assert trace.name == "Flip"
        assert trace.params["axes"] == (1,)

    def test_flip_accepts_image(self) -> None:
        image = tio.ScalarImage.from_tensor(torch.rand(1, 4, 4, 4))
        result = tio.Flip(axes=(0,))(image)
        assert isinstance(result, tio.Image)

    def test_flip_accepts_tensor(self) -> None:
        tensor = torch.rand(1, 4, 4, 4)
        result = tio.Flip(axes=(0,))(tensor)
        assert isinstance(result, torch.Tensor)

    def test_flip_in_compose(self) -> None:
        subject = tio.Subject(
            t1=tio.ScalarImage.from_tensor(torch.rand(1, 4, 4, 4)),
        )
        pipeline = tio.Compose([tio.Flip(axes=(0,)), tio.Flip(axes=(1,))])
        result = pipeline(subject)
        assert len(result.applied_transforms) == 2

    def test_flip_differentiable(self) -> None:
        tensor = torch.rand(1, 4, 4, 4, requires_grad=True)
        result = tio.Flip(axes=(0,))(tensor)
        loss = result.sum()
        loss.backward()
        assert tensor.grad is not None

    @pytest.mark.skipif(not HAS_MPS, reason="No MPS")
    def test_flip_on_mps(self) -> None:
        subject = tio.Subject(
            t1=tio.ScalarImage.from_tensor(torch.rand(1, 4, 4, 4)),
        )
        subject.to("mps")
        result = tio.Flip(axes=(0,))(subject)
        assert result.t1.device.type == "mps"

    def test_invalid_axis(self) -> None:
        subject = tio.Subject(
            t1=tio.ScalarImage.from_tensor(torch.rand(1, 4, 4, 4)),
        )
        with pytest.raises(ValueError, match="0, 1, 2"):
            tio.Flip(axes=(3,))(subject)
