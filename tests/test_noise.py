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
        result = tio.Noise(std=0.1, copy=False)(tensor)
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
        """Replaying with saved params reproduces the same noise."""
        from torchio.data.batch import SubjectsBatch

        subject1 = tio.Subject(
            t1=tio.ScalarImage.from_tensor(torch.zeros(1, 8, 8, 8)),
        )
        subject2 = tio.Subject(
            t1=tio.ScalarImage.from_tensor(torch.zeros(1, 8, 8, 8)),
        )
        noise = tio.Noise(std=1.0)
        params = {
            "mean": 0.0,
            "std": 1.0,
            "seed": 42,
        }
        batch1 = SubjectsBatch.from_subjects([subject1])
        batch2 = SubjectsBatch.from_subjects([subject2])
        result1 = noise.apply_transform(batch1, params)
        result2 = noise.apply_transform(batch2, params)
        r1 = result1.unbatch()[0]
        r2 = result2.unbatch()[0]
        torch.testing.assert_close(r1.t1.data, r2.t1.data)

    def test_negative_std_raises(self) -> None:
        with pytest.raises(ValueError, match="non-negative"):
            tio.Noise(std=-1.0)

    def test_random_std_range(self) -> None:
        """std=(lo, hi) samples uniformly each call."""
        subject = tio.Subject(
            t1=tio.ScalarImage.from_tensor(torch.zeros(1, 8, 8, 8)),
        )
        noise = tio.Noise(std=(0.5, 1.5))
        stds = set()
        for _ in range(10):
            result = noise(subject)
            sampled_std = result.applied_transforms[-1].params["std"]
            assert 0.5 <= sampled_std <= 1.5
            stds.add(round(sampled_std, 4))
        # Should have sampled different values
        assert len(stds) > 1

    def test_random_mean_range(self) -> None:
        noise = tio.Noise(mean=(-1.0, 1.0), std=0.0)
        subject = tio.Subject(
            t1=tio.ScalarImage.from_tensor(torch.zeros(1, 8, 8, 8)),
        )
        means = set()
        for _ in range(10):
            result = noise(subject)
            sampled_mean = result.applied_transforms[-1].params["mean"]
            assert -1.0 <= sampled_mean <= 1.0
            means.add(round(sampled_mean, 4))
        assert len(means) > 1

    def test_deterministic_scalar(self) -> None:
        """Scalar std is always the same."""
        noise = tio.Noise(std=0.5)
        subject = tio.Subject(
            t1=tio.ScalarImage.from_tensor(torch.zeros(1, 4, 4, 4)),
        )
        result = noise(subject)
        assert result.applied_transforms[0].params["std"] == 0.5

    def test_rician_noise(self) -> None:
        tensor = torch.ones(1, 8, 8, 8)
        subject = tio.Subject(t1=tio.ScalarImage.from_tensor(tensor))
        result = tio.Noise(std=0.5, rician=True)(subject)
        # Rician noise is always non-negative
        assert (result.t1.data >= 0).all()
        # Should differ from the original
        assert not torch.equal(result.t1.data, tensor)

    def test_rician_recorded_in_params(self) -> None:
        subject = tio.Subject(
            t1=tio.ScalarImage.from_tensor(torch.rand(1, 4, 4, 4)),
        )
        result = tio.Noise(std=0.1, rician=True)(subject)
        assert result.applied_transforms[0].params["rician"] is True

    def test_gaussian_vs_rician_differ(self) -> None:
        torch.manual_seed(42)
        tensor = torch.ones(1, 8, 8, 8)
        subject_g = tio.Subject(t1=tio.ScalarImage.from_tensor(tensor.clone()))
        subject_r = tio.Subject(t1=tio.ScalarImage.from_tensor(tensor.clone()))
        gaussian = tio.Noise(std=0.5, rician=False)(subject_g)
        rician = tio.Noise(std=0.5, rician=True)(subject_r)
        # They should produce different results
        assert not torch.equal(gaussian.t1.data, rician.t1.data)

    def test_distribution_for_std(self) -> None:
        from torch.distributions import Uniform

        subject = tio.Subject(
            t1=tio.ScalarImage.from_tensor(torch.zeros(1, 8, 8, 8)),
        )
        noise = tio.Noise(std=Uniform(0.1, 0.5))
        result = noise(subject)
        sampled_std = result.applied_transforms[0].params["std"]
        assert 0.1 <= sampled_std <= 0.5

    def test_distribution_for_mean(self) -> None:
        from torch.distributions import Normal

        subject = tio.Subject(
            t1=tio.ScalarImage.from_tensor(torch.zeros(1, 8, 8, 8)),
        )
        noise = tio.Noise(mean=Normal(0.0, 0.1), std=0.0)
        result = noise(subject)
        # Mean was sampled from N(0, 0.1) — should be near 0
        sampled_mean = result.applied_transforms[0].params["mean"]
        assert isinstance(sampled_mean, float)

    def test_lognormal_distribution(self) -> None:
        from torch.distributions import LogNormal

        subject = tio.Subject(
            t1=tio.ScalarImage.from_tensor(torch.zeros(1, 8, 8, 8)),
        )
        noise = tio.Noise(std=LogNormal(loc=-2.0, scale=0.5))
        result = noise(subject)
        sampled_std = result.applied_transforms[0].params["std"]
        assert sampled_std > 0  # LogNormal always positive
