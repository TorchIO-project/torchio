"""Tests for the Noise intensity transform."""

from __future__ import annotations

import pytest
import torch

import torchio as tio

HAS_MPS = torch.backends.mps.is_available()


class TestNoise:
    def test_adds_noise(self) -> None:
        tensor = torch.zeros(1, 8, 8, 8)
        subject = tio.Subject(t1=tio.ScalarImage.from_tensor(tensor))
        result = tio.Noise(std=1.0)(subject)
        # Should no longer be all zeros
        assert result.t1.data.abs().sum() > 0

    def test_mean_param(self) -> None:
        tensor = torch.zeros(1, 8, 8, 8)
        subject = tio.Subject(t1=tio.ScalarImage.from_tensor(tensor))
        result = tio.Noise(mean=10.0, std=0.0)(subject)
        torch.testing.assert_close(
            result.t1.data.mean(),
            torch.tensor(10.0),
            atol=0.01,
            rtol=0,
        )

    def test_zero_std_no_change(self) -> None:
        tensor = torch.rand(1, 8, 8, 8)
        subject = tio.Subject(t1=tio.ScalarImage.from_tensor(tensor.clone()))
        result = tio.Noise(std=0.0)(subject)
        torch.testing.assert_close(result.t1.data, tensor)

    def test_only_scalar_images(self) -> None:
        subject = tio.Subject(
            t1=tio.ScalarImage.from_tensor(torch.zeros(1, 8, 8, 8)),
            seg=tio.LabelMap.from_tensor(torch.zeros(1, 8, 8, 8, dtype=torch.long)),
        )
        original_seg = subject.seg.data.clone()
        result = tio.Noise(std=1.0)(subject)
        torch.testing.assert_close(result.seg.data, original_seg)

    def test_history_recorded(self) -> None:
        subject = tio.Subject(
            t1=tio.ScalarImage.from_tensor(torch.rand(1, 4, 4, 4)),
        )
        result = tio.Noise(std=0.5)(subject)
        assert len(result.applied_transforms) == 1
        trace = result.applied_transforms[0]
        assert trace.name == "Noise"
        assert "mean" in trace.params
        assert "std" in trace.params
        assert "seed" in trace.params

    def test_probability(self) -> None:
        tensor = torch.rand(1, 4, 4, 4)
        subject = tio.Subject(t1=tio.ScalarImage.from_tensor(tensor.clone()))
        result = tio.Noise(std=1.0, p=0.0)(subject)
        torch.testing.assert_close(result.t1.data, tensor)

    def test_accepts_image(self) -> None:
        image = tio.ScalarImage.from_tensor(torch.rand(1, 4, 4, 4))
        result = tio.Noise(std=0.1)(image)
        assert isinstance(result, tio.Image)

    def test_accepts_tensor(self) -> None:
        tensor = torch.rand(1, 4, 4, 4)
        result = tio.Noise(std=0.1)(tensor)
        assert isinstance(result, torch.Tensor)

    def test_differentiable(self) -> None:
        tensor = torch.rand(1, 4, 4, 4, requires_grad=True)
        result = tio.Noise(std=0.1)(tensor)
        loss = result.sum()
        loss.backward()
        assert tensor.grad is not None

    def test_in_compose(self) -> None:
        subject = tio.Subject(
            t1=tio.ScalarImage.from_tensor(torch.rand(1, 4, 4, 4)),
        )
        pipeline = tio.Compose([tio.Noise(std=0.1), tio.Noise(std=0.2)])
        result = pipeline(subject)
        assert len(result.applied_transforms) == 2

    def test_include_exclude(self) -> None:
        subject = tio.Subject(
            t1=tio.ScalarImage.from_tensor(torch.zeros(1, 4, 4, 4)),
            t2=tio.ScalarImage.from_tensor(torch.zeros(1, 4, 4, 4)),
        )
        original_t2 = subject.t2.data.clone()
        result = tio.Noise(std=1.0, include=["t1"])(subject)
        torch.testing.assert_close(result.t2.data, original_t2)

    @pytest.mark.skipif(not HAS_MPS, reason="No MPS")
    def test_noise_on_mps(self) -> None:
        subject = tio.Subject(
            t1=tio.ScalarImage.from_tensor(torch.zeros(1, 4, 4, 4)),
        )
        subject.to("mps")
        result = tio.Noise(std=1.0)(subject)
        assert result.t1.device.type == "mps"
        assert result.t1.data.abs().sum() > 0

    def test_seed_reproducibility(self) -> None:
        subject = tio.Subject(
            t1=tio.ScalarImage.from_tensor(torch.zeros(1, 8, 8, 8)),
        )
        result = tio.Noise(std=1.0)(subject)
        seed = result.applied_transforms[0].params["seed"]
        # Replay with same seed
        subject2 = tio.Subject(
            t1=tio.ScalarImage.from_tensor(torch.zeros(1, 8, 8, 8)),
        )
        noise = tio.Noise(std=1.0)
        params = {"mean": 0.0, "std": 1.0, "seed": seed}
        result2 = noise.apply(subject2, params)
        torch.testing.assert_close(result2.t1.data, result.t1.data)

    def test_negative_std_raises(self) -> None:
        with pytest.raises(ValueError):
            tio.Noise(std=-1.0)
