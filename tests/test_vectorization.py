"""Equivalence gate: per-instance batch transforms must be vectorized.

Each transform here has an `apply_transform` that is deterministic given the
recorded parameters (the randomness lives in `make_params`). The
`assert_vectorized` fixture checks that applying the transform to a batch
produces, for every element, the same result as applying the element's own
parameters to that element alone. A correct vectorized implementation passes;
any cross-element contamination or broadcasting mistake fails.

Transforms that sample inside `apply_transform` (Noise, LabelsToImage) are
excluded here and covered by their own tests.
"""

from __future__ import annotations

import pytest
import torch

import torchio as tio


def _batch(
    batch_size: int = 4, dtype: torch.dtype = torch.float32
) -> tio.SubjectsBatch:
    data = torch.rand(1, 12, 12, 12, dtype=dtype)
    subjects = [
        tio.Subject(t1=tio.ScalarImage(data.clone() + index))
        for index in range(batch_size)
    ]
    return tio.SubjectsBatch.from_subjects(subjects)


@pytest.mark.parametrize(
    "transform",
    [
        tio.Ghosting(num_ghosts=(2, 5), intensity=(0.5, 1.0)),
        tio.Spike(num_spikes=(1, 3), intensity=(0.3, 0.8)),
        tio.Blur(std=(0.5, 2.0)),
        tio.BiasField(std=(0.3, 0.8)),
        tio.Flip(axes=(0, 1, 2), flip_probability=0.5),
        tio.Motion(degrees=10.0, translation=10.0, num_transforms=2),
        tio.Swap(patch_size=3, num_iterations=5),
        tio.Anisotropy(downsampling=(1.5, 4.0)),
    ],
)
def test_vectorized_matches_per_element(transform, assert_vectorized) -> None:
    torch.manual_seed(0)
    assert_vectorized(transform, _batch())


@pytest.mark.parametrize(
    "transform",
    [
        tio.Ghosting(num_ghosts=4, intensity=1.0, p=0.5),
        tio.Spike(num_spikes=2, intensity=1.0, p=0.5),
        tio.Blur(std=1.5, p=0.5),
        tio.BiasField(std=0.5, p=0.5),
        tio.Flip(axes=(0, 1, 2), flip_probability=1.0, p=0.5),
        tio.Motion(degrees=10.0, translation=10.0, num_transforms=2, p=0.5),
        tio.Swap(patch_size=3, num_iterations=5, p=0.5),
    ],
)
def test_vectorized_matches_per_element_with_gating(
    transform,
    assert_vectorized,
) -> None:
    torch.manual_seed(0)
    assert_vectorized(transform, _batch(batch_size=6))


def test_anisotropy_tie_rounding_matches_scalar(assert_vectorized) -> None:
    # An odd spatial length with factor 2.0 makes length/factor land on a .5
    # tie (e.g. 9/2 = 4.5). The per-instance path uses torch.round and the
    # scalar path uses Python round; both must agree (round-half-to-even), so
    # the vectorized per-element result must match the scalar application.
    torch.manual_seed(0)
    data = torch.rand(1, 9, 9, 9)
    batch = tio.SubjectsBatch.from_subjects(
        [tio.Subject(t1=tio.ScalarImage(data.clone() + index)) for index in range(4)]
    )
    assert_vectorized(tio.Anisotropy(downsampling=2.0), batch)


def test_gating_with_prior_history(assert_vectorized) -> None:
    torch.manual_seed(0)
    batch = tio.Gamma(log_gamma=0.2, per_instance=False)(_batch(batch_size=6))

    assert_vectorized(
        tio.Flip(axes=(0, 1, 2), flip_probability=1.0, p=0.5),
        batch,
    )
