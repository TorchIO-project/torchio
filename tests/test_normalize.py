"""Tests for Normalize."""

from __future__ import annotations

import numpy as np
import pytest
import torch

import torchio as tio


def _make_subject(
    values: torch.Tensor | None = None,
    with_label: bool = False,
) -> tio.Subject:
    if values is None:
        values = torch.arange(1000, dtype=torch.float32).reshape(1, 10, 10, 10)
    kwargs: dict = {"t1": tio.ScalarImage(values)}
    if with_label:
        mask = torch.zeros(1, 10, 10, 10)
        mask[0, 2:8, 2:8, 2:8] = 1
        kwargs["brain"] = tio.LabelMap(mask)
    return tio.Subject(**kwargs)


class TestBasic:
    def test_default_rescales_to_minus1_1(self) -> None:
        subject = _make_subject()
        result = tio.Normalize()(subject)
        data = result.t1.data
        assert abs(data.min().item() - (-1.0)) < 1e-5
        assert abs(data.max().item() - 1.0) < 1e-5

    def test_rescale_to_0_1(self) -> None:
        subject = _make_subject()
        result = tio.Normalize(out_min=0.0, out_max=1.0)(subject)
        data = result.t1.data
        assert abs(data.min().item()) < 1e-5
        assert abs(data.max().item() - 1.0) < 1e-5

    def test_rescale_to_0_255(self) -> None:
        subject = _make_subject()
        result = tio.Normalize(out_min=0.0, out_max=255.0)(subject)
        data = result.t1.data
        assert abs(data.min().item()) < 1e-3
        assert abs(data.max().item() - 255.0) < 1e-3

    def test_ct_windowing(self) -> None:
        data = torch.tensor([-1500, -1000, 0, 500, 1000, 2000], dtype=torch.float32)
        data = data.reshape(1, 1, 1, 6)
        subject = tio.Subject(ct=tio.ScalarImage(data))
        result = tio.Normalize(
            out_min=0.0,
            out_max=1.0,
            in_min=-1000.0,
            in_max=1000.0,
        )(subject)
        out = result.ct.data.flatten()
        # -1500 gets clipped to -1000 -> maps to 0
        assert abs(out[0].item()) < 1e-5
        # 0 maps to 0.5
        assert abs(out[2].item() - 0.5) < 1e-5
        # 2000 gets clipped to 1000 -> maps to 1
        assert abs(out[5].item() - 1.0) < 1e-5


class TestPercentiles:
    def test_percentile_clipping(self) -> None:
        data = torch.cat(
            [
                torch.zeros(1, 5, 10, 10),
                torch.ones(1, 5, 10, 10) * 100,
            ],
            dim=1,
        )
        # First 50% is 0, second 50% is 100
        subject = tio.Subject(t1=tio.ScalarImage(data))
        result = tio.Normalize(
            out_min=0.0,
            out_max=1.0,
            percentile_low=1.0,
            percentile_high=99.0,
        )(subject)
        assert result.t1.data.min() >= -0.01
        assert result.t1.data.max() <= 1.01

    def test_nnunet_percentiles(self) -> None:
        torch.manual_seed(42)
        data = torch.randn(1, 20, 20, 20) * 100
        subject = tio.Subject(t1=tio.ScalarImage(data))
        result = tio.Normalize(
            out_min=0.0,
            out_max=1.0,
            percentile_low=0.5,
            percentile_high=99.5,
        )(subject)
        # Most values should be in [0, 1], outliers clipped
        in_range = (result.t1.data >= -0.01) & (result.t1.data <= 1.01)
        assert in_range.float().mean() > 0.98


class TestMasking:
    def test_masking_with_label_key(self) -> None:
        subject = _make_subject(with_label=True)
        result = tio.Normalize(
            out_min=0.0,
            out_max=1.0,
            masking_method="brain",
        )(subject)
        assert result.t1.data.shape == subject.t1.data.shape

    def test_masking_with_callable(self) -> None:
        subject = _make_subject()
        result = tio.Normalize(
            out_min=0.0,
            out_max=1.0,
            masking_method=lambda x: x > 500,
        )(subject)
        assert result.t1.data.shape == subject.t1.data.shape

    def test_masking_key_not_found_raises(self) -> None:
        subject = _make_subject()
        with pytest.raises(KeyError, match="nonexistent"):
            tio.Normalize(
                masking_method="nonexistent",
            )(subject)

    def test_masking_key_not_labelmap_raises(self) -> None:
        subject = tio.Subject(
            t1=tio.ScalarImage(torch.rand(1, 4, 4, 4)),
            t2=tio.ScalarImage(torch.rand(1, 4, 4, 4)),
        )
        with pytest.raises(TypeError, match="LabelMap"):
            tio.Normalize(masking_method="t2")(subject)


class TestRandom:
    def test_random_out_range(self) -> None:
        subject = _make_subject()
        transform = tio.Normalize(
            out_min=(-2.0, -0.5),
            out_max=(0.5, 2.0),
        )
        results = [transform(subject).t1.data.min().item() for _ in range(5)]
        # With random sampling, not all results should be identical
        assert len({f"{v:.2f}" for v in results}) > 1

    def test_random_percentiles(self) -> None:
        torch.manual_seed(0)
        subject = _make_subject()
        transform = tio.Normalize(
            out_min=0.0,
            out_max=1.0,
            percentile_low=(0.0, 5.0),
            percentile_high=(95.0, 100.0),
        )
        # Random percentiles change the clipping bounds, so
        # values above the high percentile get clamped to 1.0 and
        # the interior distribution shifts.
        results = [transform(subject).t1.data.mean().item() for _ in range(10)]
        assert len({f"{v:.4f}" for v in results}) > 1


class TestEdgeCases:
    def test_constant_value_warns(self) -> None:
        data = torch.ones(1, 4, 4, 4) * 42.0
        subject = tio.Subject(t1=tio.ScalarImage(data))
        with pytest.warns(RuntimeWarning, match="zero"):
            result = tio.Normalize()(subject)
        # Data unchanged
        torch.testing.assert_close(result.t1.data, data)

    def test_empty_mask_warns(self) -> None:
        subject = _make_subject()
        with pytest.warns(RuntimeWarning, match="empty"):
            tio.Normalize(
                out_min=0.0,
                out_max=1.0,
                masking_method=lambda x: torch.zeros_like(x, dtype=torch.bool),
            )(subject)

    def test_leaves_label_maps_unchanged(self) -> None:
        subject = _make_subject(with_label=True)
        original_label = subject.brain.data.clone()
        result = tio.Normalize()(subject)
        torch.testing.assert_close(result.brain.data, original_label)


class TestInverse:
    def test_inverse_restores_values(self) -> None:
        subject = _make_subject()
        original = subject.t1.data.clone()
        transformed = tio.Normalize(
            out_min=0.0,
            out_max=1.0,
        )(subject)
        restored = transformed.apply_inverse_transform()
        np.testing.assert_allclose(
            restored.t1.data.numpy(),
            original.numpy(),
            atol=1e-4,
        )

    def test_inverse_with_ct_windowing(self) -> None:
        data = torch.linspace(-500, 500, 1000).reshape(1, 10, 10, 10)
        subject = tio.Subject(ct=tio.ScalarImage(data))
        transformed = tio.Normalize(
            out_min=0.0,
            out_max=1.0,
            in_min=-1000.0,
            in_max=1000.0,
        )(subject)
        restored = transformed.apply_inverse_transform()
        # Clipped values can't be restored, but the linear map is reversed
        np.testing.assert_allclose(
            restored.ct.data.numpy(),
            data.numpy(),
            atol=1e-4,
        )


class TestExports:
    def test_available_at_top_level(self) -> None:
        assert hasattr(tio, "RescaleIntensity")


class TestAlias:
    def test_rescale_intensity_alias(self) -> None:
        assert tio.RescaleIntensity is tio.Normalize


class TestNormalizeLargeImage:
    """torch.quantile rejects tensors with more than 2**24 elements."""

    LIMIT = 2**24

    def test_percentile_range_min_max_endpoints(self) -> None:
        # The 0/100 endpoints use min/max, which has no size limit.
        from torchio.transforms.intensity.normalize import _percentile_range

        values = torch.linspace(-5.0, 10.0, 10_000).reshape(1, -1, 1, 1)
        low, high = _percentile_range(values, None, 0.0, 100.0, "t1")
        assert low == pytest.approx(-5.0, abs=1e-3)
        assert high == pytest.approx(10.0, abs=1e-3)

    def test_percentile_range_interior_subsamples(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        # Exercise the subsample branch quickly via a small cap.
        from torchio.transforms.intensity import normalize as norm

        monkeypatch.setattr(norm, "_QUANTILE_SUBSAMPLE_SIZE", 1000)
        values = torch.linspace(0.0, 100.0, 5000).reshape(1, -1, 1, 1)
        low, high = norm._percentile_range(values, None, 25.0, 75.0, "t1")
        assert 24.0 < low < 26.0
        assert 74.0 < high < 76.0

    def test_rescale_intensity_large_image(self) -> None:
        # Full integration on a tensor exceeding torch.quantile's 2**24 limit.
        # The default 0/100 percentiles use min/max, so this stays fast.
        data = torch.linspace(0.0, 1000.0, self.LIMIT + 1000).reshape(1, -1, 1, 1)
        result = tio.RescaleIntensity(out_min=0, out_max=1)(tio.ScalarImage(data))
        assert float(result.data.min()) == pytest.approx(0.0, abs=1e-4)
        assert float(result.data.max()) == pytest.approx(1.0, abs=1e-4)


class TestNormalizePerInstance:
    def _batch(self, batch_size: int = 6) -> tio.SubjectsBatch:
        subjects = [
            tio.Subject(t1=tio.ScalarImage(torch.rand(1, 8, 8, 8) * 100))
            for _ in range(batch_size)
        ]
        return tio.SubjectsBatch.from_subjects(subjects)

    def test_per_instance_out_range_differs(self) -> None:
        torch.manual_seed(0)
        batch = self._batch()
        transform = tio.RescaleIntensity(out_min=(-1.0, 0.0), out_max=(0.5, 1.0))
        result = transform(batch)
        params = result.applied_transforms[-1].params
        assert "_batched_keys" in params
        assert len(params["out_min"]) == batch.batch_size
        assert len(set(params["out_min"])) > 1
        # Each element rescaled to its own output range.
        for i in range(batch.batch_size):
            data = result.t1.data[i]
            assert data.min() >= params["out_min"][i] - 1e-4
            assert data.max() <= params["out_max"][i] + 1e-4

    def test_per_instance_false_shares_params(self) -> None:
        torch.manual_seed(0)
        batch = self._batch()
        transform = tio.RescaleIntensity(
            out_min=(-1.0, 0.0),
            out_max=(0.5, 1.0),
            per_instance=False,
        )
        result = transform(batch)
        params = result.applied_transforms[-1].params
        assert isinstance(params["out_min"], float)

    def test_single_subject_keeps_scalar_params(self) -> None:
        subject = tio.Subject(t1=tio.ScalarImage(torch.rand(1, 8, 8, 8) * 100))
        result = tio.RescaleIntensity(out_min=(-1.0, 0.0), out_max=(0.5, 1.0))(subject)
        assert isinstance(result.applied_transforms[-1].params["out_min"], float)

    def test_per_instance_inverse_zero_range_no_nan(self) -> None:
        # A degenerate out_min == out_max (zero output range) must not
        # produce NaNs on the per-element inverse.
        torch.manual_seed(0)
        data = torch.rand(1, 8, 8, 8) * 100
        subjects = [tio.Subject(t1=tio.ScalarImage(data.clone())) for _ in range(4)]
        batch = tio.SubjectsBatch.from_subjects(subjects)
        transform = tio.RescaleIntensity(out_min=0.0, out_max=0.0)
        result = transform(batch)
        assert "_batched_keys" in result.applied_transforms[-1].params
        restored = result.apply_inverse_transform()
        assert not torch.isnan(restored.t1.data).any()
        # Identical inputs: the batch-shared input range covers every
        # element, so only the per-element output range varies and the
        # round-trip is exact.
        torch.manual_seed(0)
        data = torch.rand(1, 8, 8, 8) * 100
        subjects = [tio.Subject(t1=tio.ScalarImage(data.clone())) for _ in range(6)]
        batch = tio.SubjectsBatch.from_subjects(subjects)
        original = batch.t1.data.clone()
        transform = tio.RescaleIntensity(out_min=(-1.0, 0.0), out_max=(0.5, 1.0))
        result = transform(batch)
        restored = result.apply_inverse_transform()
        torch.testing.assert_close(restored.t1.data, original, atol=1e-3, rtol=0)
