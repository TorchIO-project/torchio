"""Tests for Normalize."""

from __future__ import annotations

import numpy as np
import pytest
import torch

import torchio as tio
from torchio.transforms._statistics import compute_quantile


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
    @pytest.mark.parametrize(
        ("bounds", "expected"),
        [
            ({"in_min": 50.0}, [0.0, 0.0, 1.0]),
            ({"in_max": 50.0}, [0.0, 1.0, 1.0]),
            ({"in_min": tio.Choice([50.0])}, [0.0, 0.0, 1.0]),
            ({"in_max": tio.Choice([50.0])}, [0.0, 1.0, 1.0]),
        ],
    )
    def test_single_explicit_bound(self, bounds: dict, expected: list[float]) -> None:
        image = tio.ScalarImage(torch.tensor([0.0, 50.0, 100.0]).reshape(1, 1, 1, 3))
        result = tio.Normalize(out_min=0.0, out_max=1.0, **bounds)(image)
        torch.testing.assert_close(result.data.flatten(), torch.tensor(expected))

    @pytest.mark.parametrize("bound", ["in_min", "in_max"])
    def test_single_bound_with_masked_percentiles(self, bound: str) -> None:
        data = torch.tensor([0.0, 20.0, 40.0, 60.0, 80.0]).reshape(1, 1, 1, 5)
        subject = tio.Subject(
            a=tio.ScalarImage(data),
            b=tio.ScalarImage(data * 2),
            mask=tio.LabelMap(torch.tensor([0, 1, 1, 1, 0]).reshape(1, 1, 1, 5)),
        )
        bounds = {bound: 0.0 if bound == "in_min" else 160.0}
        transform = tio.Normalize(
            out_min=0.0,
            out_max=1.0,
            percentile_low=25.0,
            percentile_high=75.0,
            masking_method="mask",
            **bounds,
        )
        result = transform(subject)
        ranges = {"a": (0.0, 50.0), "b": (0.0, 100.0)}
        if bound == "in_max":
            ranges = {"a": (30.0, 160.0), "b": (60.0, 160.0)}
        assert result.applied_transforms[-1].params["in_ranges"] == ranges
        restored = result.apply_inverse_transform()
        for name, (low, high) in ranges.items():
            expected = subject[name].data.clamp(low, high)
            torch.testing.assert_close(
                result[name].data, (expected - low) / (high - low)
            )
            torch.testing.assert_close(restored[name].data, expected)
        torch.testing.assert_close(result.mask.data, subject.mask.data)

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


class TestInputRangeContract:
    @pytest.mark.parametrize("mode", ["auto", "minimum", "maximum", "both"])
    @pytest.mark.parametrize("dtype", [torch.float32, torch.float64, torch.int16])
    @pytest.mark.parametrize("shape", [(2, 7, 9, 1), (2, 4, 5, 6)])
    @pytest.mark.parametrize("masked", [False, True])
    @pytest.mark.parametrize("batched", [False, True])
    def test_matches_numpy_reference(
        self,
        mode: str,
        dtype: torch.dtype,
        shape: tuple[int, int, int, int],
        masked: bool,
        batched: bool,
    ) -> None:
        rng = np.random.default_rng(1509)
        data = torch.from_numpy(rng.normal(70, 50, size=shape)).to(dtype)
        mask = torch.zeros((1, *shape[1:]), dtype=torch.int16)
        mask[:, 1:-1, 1:-1, :] = 1
        subjects = [
            tio.Subject(
                a=tio.ScalarImage((data + 30 * index).to(dtype)),
                b=tio.ScalarImage((data * 2 + 10 - 30 * index).to(dtype)),
                mask=tio.LabelMap(mask.clone()),
            )
            for index in range(2 if batched else 1)
        ]
        source = tio.SubjectsBatch.from_subjects(subjects) if batched else subjects[0]
        bounds = {}
        if mode in ("minimum", "both"):
            bounds["in_min"] = -20.0
        if mode in ("maximum", "both"):
            bounds["in_max"] = 180.0
        result = tio.Normalize(
            out_min=-2.0,
            out_max=3.0,
            percentile_low=10.0,
            percentile_high=90.0,
            masking_method="mask" if masked else None,
            **bounds,
        )(source)
        for name in ("a", "b"):
            # Keep the existing first-element, batch-shared range convention.
            reference_data = subjects[0][name].data.float().numpy()
            values = reference_data
            if masked:
                values = values[np.broadcast_to(mask.numpy().astype(bool), shape)]
            # Independent percentile implementation and documented affine map.
            low, high = np.percentile(values, [10, 90])
            low = bounds.get("in_min", low)
            high = bounds.get("in_max", high)
            original = source[name].data.float().numpy()
            expected = (np.clip(original, low, high) - low) / (high - low) * 5 - 2
            actual = result[name].data.numpy()
            np.testing.assert_allclose(actual, expected, atol=2e-5, rtol=2e-5)
            assert actual.shape == original.shape
            assert np.isfinite(actual).all()
            assert actual.min() >= -2.0 - 1e-5
            assert actual.max() <= 3.0 + 1e-5
        torch.testing.assert_close(result.mask.data, source.mask.data)

    @pytest.mark.parametrize("bound", ["in_min", "in_max"])
    @pytest.mark.parametrize("kind", ["tuple", "choice", "distribution"])
    @pytest.mark.parametrize("seed", [0, 7, 1509])
    def test_random_bound_is_shared_and_reproducible(
        self, bound: str, kind: str, seed: int
    ) -> None:
        lower, upper = (10.0, 30.0) if bound == "in_min" else (60.0, 80.0)
        specifications = {
            "tuple": (lower, upper),
            "choice": tio.Choice([lower, upper]),
            "distribution": torch.distributions.Uniform(lower, upper),
        }
        data = torch.linspace(0, 100, 24).reshape(1, 4, 6, 1)
        subject = tio.Subject(a=tio.ScalarImage(data), b=tio.ScalarImage(data * 2))
        bounds: dict = {bound: specifications[kind]}
        transform = tio.Normalize(out_min=0.0, out_max=1.0, **bounds)
        torch.manual_seed(seed)
        result = transform(subject)
        index = 0 if bound == "in_min" else 1
        ranges = result.applied_transforms[-1].params["in_ranges"]
        sampled = ranges["a"][index]
        assert lower <= sampled <= upper
        assert ranges["b"][index] == sampled
        if kind == "choice":
            assert sampled in (lower, upper)
        torch.manual_seed(seed)
        replay = transform(subject)
        for name in ("a", "b"):
            torch.testing.assert_close(replay[name].data, result[name].data)
            low, high = ranges[name]
            expected = (subject[name].data.clamp(low, high) - low) / (high - low)
            torch.testing.assert_close(result[name].data, expected)

    @pytest.mark.parametrize("bound", ["in_min", "in_max"])
    def test_empty_mask_retains_explicit_bound(self, bound: str) -> None:
        data = torch.tensor([0.0, 50.0, 100.0]).reshape(1, 1, 1, 3)
        transform = tio.Normalize(
            out_min=0.0,
            out_max=1.0,
            masking_method=lambda x: torch.zeros_like(x, dtype=torch.bool),
            **{bound: 50.0},
        )
        with pytest.warns(RuntimeWarning, match="mask is empty"):
            result = transform(tio.ScalarImage(data))
        expected = [0.0, 0.0, 1.0] if bound == "in_min" else [0.0, 1.0, 1.0]
        torch.testing.assert_close(result.data.flatten(), torch.tensor(expected))

    @pytest.mark.parametrize("bound", ["in_min", "in_max"])
    def test_equal_inferred_and_explicit_bounds_warn(self, bound: str) -> None:
        data = torch.tensor([0.0, 50.0, 100.0]).reshape(1, 1, 1, 3)
        value = 100.0 if bound == "in_min" else 0.0
        bounds: dict = {bound: value}
        with pytest.warns(RuntimeWarning, match="input range is zero"):
            result = tio.Normalize(**bounds)(tio.ScalarImage(data))
        torch.testing.assert_close(result.data, data)

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
    @pytest.mark.parametrize("bound", ["in_min", "in_max"])
    @pytest.mark.parametrize("dtype", [torch.float32, torch.float64, torch.int16])
    def test_partial_bound_matches_cpu_on_cuda(
        self, bound: str, dtype: torch.dtype
    ) -> None:
        data = torch.linspace(0, 100, 24).reshape(1, 4, 6, 1).to(dtype)
        transform = tio.Normalize(
            out_min=0.0,
            out_max=1.0,
            masking_method="mask",
            **{bound: 50.0},
        )
        outputs = []
        for device in ("cpu", "cuda"):
            subject = tio.Subject(
                image=tio.ScalarImage(data.to(device)),
                mask=tio.LabelMap((data > 20).to(device)),
            )
            result = transform(subject)
            assert result.image.data.device.type == device
            outputs.append(result.image.data.cpu())
        torch.testing.assert_close(outputs[0], outputs[1], atol=2e-5, rtol=2e-5)


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


class TestQuantile:
    """Tests for the `torch.kthvalue`-based quantile helper."""

    @pytest.mark.parametrize("q", [0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0])
    def test_matches_torch_quantile(self, q: float) -> None:
        values = torch.linspace(-3.0, 7.0, 101)
        expected = torch.quantile(values, q)
        result = compute_quantile(values, q)
        assert torch.allclose(result, expected, atol=1e-5)

    def test_invalid_q_raises(self) -> None:
        values = torch.arange(10, dtype=torch.float32)
        with pytest.raises(ValueError, match="0 <= q <= 1"):
            compute_quantile(values, 1.5)

    def test_large_tensor_interior_quantile(self) -> None:
        # torch.quantile raises for more than 2**24 elements; kthvalue does not.
        values = torch.arange(2**24 + 1, dtype=torch.float32)
        result = compute_quantile(values, 0.5)
        assert result.item() == pytest.approx(2**23)

    def test_rescale_intensity_large_image(self) -> None:
        # Exceeds torch.quantile's 2**24-element limit; uses min/max endpoints.
        data = torch.zeros(1, 2**24 + 1, 1, 1, dtype=torch.float32)
        # A single non-zero voxel becomes the input maximum; everything else
        # is the minimum, so the output spans the full [0, 1] range.
        input_max = 4.0
        data[0, -1] = input_max
        image = tio.ScalarImage(data)
        transform = tio.RescaleIntensity(out_min=0.0, out_max=1.0, copy=False)
        result = transform(image)
        assert result.data[0, 0, 0, 0].item() == pytest.approx(0.0)
        assert result.data[0, -1, 0, 0].item() == pytest.approx(1.0)


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
