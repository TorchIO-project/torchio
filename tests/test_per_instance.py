"""Cross-cutting tests for per-instance batch augmentation."""

from __future__ import annotations

import pytest
import torch

import torchio as tio


def _identical_batch(batch_size: int = 4) -> tio.SubjectsBatch:
    data = torch.rand(1, 8, 8, 8) + 0.1
    subjects = [
        tio.Subject(t1=tio.ScalarImage(data.clone())) for _ in range(batch_size)
    ]
    return tio.SubjectsBatch.from_subjects(subjects)


class TestCapabilityFlags:
    def test_base_defaults_false(self) -> None:
        transform = tio.transforms.Transform()
        assert transform.supports_per_instance_params is False
        assert transform.supports_per_instance_p is False

    @pytest.mark.parametrize(
        "cls_kwargs",
        [
            (tio.Noise, {"std": 0.1}),
            (tio.Gamma, {"log_gamma": 0.1}),
        ],
    )
    def test_converted_intensity_opt_in(self, cls_kwargs) -> None:
        cls, kwargs = cls_kwargs
        transform = cls(**kwargs)
        assert transform.supports_per_instance_params
        assert transform.supports_per_instance_p

    def test_spatial_opts_in(self) -> None:
        transform = tio.Affine(degrees=10.0)
        assert transform.supports_per_instance_params
        assert transform.supports_per_instance_p

    def test_resample_disables_per_instance_p(self) -> None:
        # A resampling target is shape-changing, so per-element gating is off.
        transform = tio.Resample(2)
        assert transform.supports_per_instance_params
        assert not transform.supports_per_instance_p


class TestUnconvertedTransforms:
    """A transform that does not opt in stays batch-shared on a batch."""

    def test_unconverted_transform_resolves_no_batch(self) -> None:
        class Plain(tio.transforms.IntensityTransform):
            # Does not override the capability flags, so it never samples
            # per-instance parameters even for a batch.
            def make_params(self, batch):
                return {"n": self._resolve_n(batch)}

            def apply_transform(self, batch, params):
                return batch

        batch = _identical_batch()
        result = Plain()(batch)
        params = result.applied_transforms[-1].params
        assert params["n"] is None
        assert "_batched_keys" not in params

    def test_unconverted_flags_default_false(self) -> None:
        class Plain(tio.transforms.IntensityTransform):
            def apply_transform(self, batch, params):
                return batch

        transform = Plain()
        assert not transform.supports_per_instance_params
        assert not transform.supports_per_instance_p


class TestComposePerInstance:
    def test_compose_child_is_per_instance(self) -> None:
        torch.manual_seed(0)
        batch = _identical_batch()
        pipeline = tio.Compose([tio.Gamma(log_gamma=(0.2, 0.8))])
        result = pipeline(batch)
        log_gammas = [history[-1].params["log_gamma"] for history in result.histories]
        assert all(isinstance(value, float) for value in log_gammas)
        assert len(set(log_gammas)) > 1

    def test_compose_respects_per_instance_false(self) -> None:
        torch.manual_seed(0)
        batch = _identical_batch()
        pipeline = tio.Compose([tio.Gamma(log_gamma=(0.2, 0.8), per_instance=False)])
        result = pipeline(batch)
        params = result.applied_transforms[-1].params
        assert isinstance(params["log_gamma"], float)


class TestPerInstanceHistory:
    def test_unbatch_slices_history(self) -> None:
        torch.manual_seed(0)
        batch = _identical_batch(batch_size=4)
        result = tio.Gamma(log_gamma=(0.2, 0.8))(batch)
        batch_log_gammas = [
            history[-1].params["log_gamma"] for history in result.histories
        ]
        for i, subject in enumerate(result.unbatch()):
            trace = subject.applied_transforms[-1]
            assert trace.params["log_gamma"] == batch_log_gammas[i]
            assert "_batched_keys" not in trace.params


class TestSpatialBatchSizeValidation:
    def test_mismatched_batch_size_raises(self) -> None:
        torch.manual_seed(0)
        batch = _identical_batch(batch_size=4)
        transform = tio.Affine(degrees=(20.0, 80.0), default_pad_value=0.0)
        params = transform.make_params(batch)
        smaller = _identical_batch(batch_size=2)
        with pytest.raises(RuntimeError, match="Per-instance spatial parameters"):
            transform.apply_transform(smaller, params)

    def test_history_index_out_of_range_raises(self) -> None:
        torch.manual_seed(0)
        batch = _identical_batch(batch_size=4)
        result = tio.Noise(std=(0.1, 0.5))(batch)
        with pytest.raises(IndexError):
            result.history(4)


class TestPerInstanceDtypePreservation:
    """Per-instance gating must not produce mixed-dtype batch outputs.

    Float-domain intensity transforms compute in ``float32`` internally.
    When per-element gating skips some elements, the skipped ones keep the
    input dtype while applied ones must be cast back, so ``torch.cat`` over
    the batch does not fail on a dtype mismatch for non-``float32`` inputs.
    """

    @staticmethod
    def _mixed_dtype_batch(dtype: torch.dtype, batch_size: int = 8):
        data = (torch.rand(1, 8, 8, 8) + 0.5).to(dtype)
        subjects = [
            tio.Subject(t1=tio.ScalarImage(data.clone())) for _ in range(batch_size)
        ]
        return tio.SubjectsBatch.from_subjects(subjects)

    @pytest.mark.parametrize(
        "transform",
        [
            tio.Ghosting(num_ghosts=4, intensity=1.0, p=0.5),
            tio.Spike(num_spikes=2, intensity=1.0, p=0.5),
            tio.Motion(degrees=10.0, translation=10.0, num_transforms=2, p=0.5),
        ],
    )
    def test_fft_transforms_preserve_float64(self, transform) -> None:
        torch.manual_seed(0)
        batch = self._mixed_dtype_batch(torch.float64)
        result = transform(batch)
        assert result.t1.data.dtype == torch.float64

    def test_bias_field_preserves_float16(self) -> None:
        torch.manual_seed(0)
        batch = self._mixed_dtype_batch(torch.float16)
        result = tio.BiasField(std=0.5, p=0.5)(batch)
        assert result.t1.data.dtype == torch.float16


class TestFullyGatedNoHistory:
    def test_fully_gated_records_no_history(self) -> None:
        # p=0 gates out every element, an exact no-op, so no history trace
        # is recorded (and replaying it cannot trigger a spurious resample).
        torch.manual_seed(0)
        batch = _identical_batch(batch_size=4)
        result = tio.Affine(degrees=20.0, p=0.0)(batch)
        assert result.applied_transforms == []

    def test_fully_gated_inverse_preserves_float64(self) -> None:
        torch.manual_seed(0)
        data = torch.rand(1, 8, 8, 8, dtype=torch.float64)
        batch = tio.SubjectsBatch.from_subjects(
            [tio.Subject(t1=tio.ScalarImage(data.clone())) for _ in range(4)]
        )
        original = batch.t1.data.clone()
        result = tio.Affine(degrees=20.0, p=0.0)(batch)
        restored = result.apply_inverse_transform()
        assert torch.equal(restored.t1.data, original)
