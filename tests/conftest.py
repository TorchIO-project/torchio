"""Shared pytest fixtures for the test suite."""

from __future__ import annotations

import copy
from collections.abc import Callable
from typing import Any

import pytest
import torch

from torchio.data.batch import SubjectsBatch
from torchio.data.batch import _slice_params


@pytest.fixture
def assert_vectorized() -> Callable[..., None]:
    """Return an assertion that a per-instance transform is vectorized faithfully.

    The returned helper applies *transform* to a batch (per-instance) and then,
    for every element, re-applies the same per-element parameters to that
    element alone. The two must match, which proves the vectorized whole-batch
    computation is equivalent to processing each element independently (no
    cross-element contamination or broadcasting mistakes).

    This is only valid for transforms whose `apply_transform` is deterministic
    given the recorded parameters (the randomness lives in `make_params`).
    Transforms that sample inside `apply_transform` (e.g. Noise, LabelsToImage)
    need a different check.
    """

    def _assert(
        transform: Any,
        batch: SubjectsBatch,
        *,
        rtol: float = 1e-5,
        atol: float = 1e-6,
    ) -> None:
        original = copy.deepcopy(batch)
        result = transform(batch)
        params = result.applied_transforms[-1].params
        assert "_batched_keys" in params, "per-instance path was not active"
        batched_keys = params["_batched_keys"]
        image_names = list(transform._get_images(result).keys())
        result_images = transform._get_images(result)
        original_subjects = original.unbatch()
        for index in range(original.batch_size):
            single = SubjectsBatch.from_subjects([original_subjects[index]])
            element_params = _slice_params(params, index, batched_keys)
            single = transform.apply_transform(single, element_params)
            single_images = transform._get_images(single)
            for name in image_names:
                torch.testing.assert_close(
                    result_images[name].data[index : index + 1],
                    single_images[name].data,
                    rtol=rtol,
                    atol=atol,
                )

    return _assert
