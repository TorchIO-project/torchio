"""No-arg augmentation transforms are a deterministic no-op and warn.

Transforms whose parameters are sampled from a range should, when
constructed with no arguments, default to an identity (no-op) and emit a
warning telling the user to pass arguments. Inherently-stochastic
transforms (which draw a random realisation rather than sampling a scalar
parameter) are exempt.
"""

from __future__ import annotations

import warnings

import pytest
import torch

import torchio as tio
from torchio.transforms.parameter_range import _ParameterRange

# Transforms that must be a deterministic no-op (and warn) with no args.
NOOP_TRANSFORMS = ["Affine", "Anisotropy", "Blur", "Gamma", "Ghosting", "Spike"]

# Args that activate each transform: should NOT warn and should change data.
# Ranges are kept away from the identity value to avoid flaky assertions.
ACTIVE_KWARGS: dict[str, dict] = {
    "Affine": {"degrees": (10, 15)},
    "Anisotropy": {"downsampling": (2, 5)},
    "Blur": {"std": (1, 2)},
    "Gamma": {"log_gamma": (0.3, 0.5)},
    "Ghosting": {"intensity": (0.8, 1.0)},
    "Spike": {"intensity": (2, 3)},
}

# Inherently-stochastic transforms: no-arg construction must NOT warn.
EXEMPT_TRANSFORMS = ["Noise", "ElasticDeformation", "Swap", "BiasField"]


def _subject() -> tio.Subject:
    torch.manual_seed(0)
    return tio.Subject(t1=tio.ScalarImage(torch.rand(1, 12, 12, 12) * 100))


@pytest.mark.parametrize("name", NOOP_TRANSFORMS)
def test_no_args_is_identity(name: str) -> None:
    subject = _subject()
    original = subject.t1.data.clone()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = getattr(tio, name)()(subject)
    torch.testing.assert_close(result.t1.data, original)


@pytest.mark.parametrize("name", NOOP_TRANSFORMS)
def test_no_args_warns(name: str) -> None:
    with pytest.warns(UserWarning, match=name):
        getattr(tio, name)()


@pytest.mark.parametrize("name", NOOP_TRANSFORMS)
def test_active_does_not_warn_and_changes(name: str) -> None:
    subject = _subject()
    original = subject.t1.data.clone()
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # any warning becomes an error
        transform = getattr(tio, name)(**ACTIVE_KWARGS[name])
    torch.manual_seed(0)
    result = transform(subject)
    assert not torch.allclose(result.t1.data, original)


@pytest.mark.parametrize("name", EXEMPT_TRANSFORMS)
def test_stochastic_no_args_does_not_warn(name: str) -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        getattr(tio, name)()


class TestIsConstant:
    def test_scalar(self) -> None:
        assert _ParameterRange(0.0).is_constant(0.0)
        assert _ParameterRange(1.0).is_constant(1.0)
        assert not _ParameterRange(0.0).is_constant(1.0)

    def test_degenerate_range_is_constant(self) -> None:
        assert _ParameterRange((0.0, 0.0)).is_constant(0.0)

    def test_real_range_is_not_constant(self) -> None:
        assert not _ParameterRange((0.0, 2.0)).is_constant(0.0)
