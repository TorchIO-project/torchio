"""Swap: randomly swap patches within an image for self-supervised learning."""

from __future__ import annotations

import warnings
from typing import Any

import torch
from torch import Tensor

from ..data.batch import SubjectsBatch
from ..data.image import LabelMap
from .parameter_range import to_nonneg_range
from .transform import IntensityTransform


class Swap(IntensityTransform):
    r"""Randomly swap patches within an image.

    This is typically used in
    [context restoration for self-supervised learning](https://www.sciencedirect.com/science/article/pii/S1361841518304699).
    Pairs of same-sized patches are selected at random and their
    contents are exchanged.

    Warning:
        This transform is intended for **self-supervised** or
        **unsupervised** workflows.  Because the spatial content is
        rearranged, aligned label maps become inconsistent with the
        swapped image.  A warning is emitted if ``LabelMap`` images
        are present in the subject.

    Args:
        patch_size: Spatial size of the patches to swap.  A single
            integer $n$ means $(n, n, n)$.
        num_iterations: Number of patch pairs to swap.  A 2-tuple
            $(a, b)$ samples $n \sim \mathcal{U}(a, b)$.
        **kwargs: See [`Transform`][torchio.Transform].

    Examples:
        >>> import torchio as tio
        >>> transform = tio.Swap(patch_size=15, num_iterations=100)
    """

    def __init__(
        self,
        *,
        patch_size: int | tuple[int, int, int] = 15,
        num_iterations: int | tuple[int, int] = 100,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        if isinstance(patch_size, int):
            patch_size = (patch_size, patch_size, patch_size)
        self.patch_size = patch_size
        self.num_iterations = to_nonneg_range(num_iterations)

    def make_params(self, batch: SubjectsBatch) -> dict[str, Any]:
        """Sample swap locations for each image."""
        n = max(1, round(self.num_iterations.sample_1d()))

        # Warn if label maps are present.
        for _name, img_batch in batch.images.items():
            if issubclass(img_batch._image_class, LabelMap):
                warnings.warn(
                    "Swap is applied to a subject containing LabelMap "
                    "images. The spatial rearrangement will make labels "
                    "inconsistent with the swapped image. This transform "
                    "is intended for self-supervised learning.",
                    stacklevel=2,
                )
                break

        # Sample one set of locations per image (shared across batch).
        any_img = next(iter(batch.images.values()))
        spatial_shape = any_img.data.shape[2:]  # (I, J, K)
        locations = _sample_swap_locations(
            spatial_shape,
            self.patch_size,
            n,
        )
        return {"locations": locations}

    def apply_transform(
        self,
        batch: SubjectsBatch,
        params: dict[str, Any],
    ) -> SubjectsBatch:
        """Swap patches in each selected image."""
        locations = params["locations"]
        for _name, img_batch in self._get_images(batch).items():
            img_batch.data = _apply_swaps(
                img_batch.data,
                locations,
                self.patch_size,
            )
        return batch


def _sample_swap_locations(
    spatial_shape: tuple[int, ...],
    patch_size: tuple[int, int, int],
    num_iterations: int,
) -> list[tuple[tuple[int, int, int], tuple[int, int, int]]]:
    """Sample pairs of non-overlapping patch origins.

    Args:
        spatial_shape: ``(I, J, K)`` spatial dimensions.
        patch_size: ``(pi, pj, pk)`` patch dimensions.
        num_iterations: Number of pairs to sample.

    Returns:
        List of ``(origin_a, origin_b)`` tuples.
    """
    locations: list[tuple[tuple[int, int, int], tuple[int, int, int]]] = []
    max_ini = [s - p for s, p in zip(spatial_shape, patch_size, strict=True)]
    if any(m < 0 for m in max_ini):
        msg = (
            f"Patch size {patch_size} cannot be larger than "
            f"spatial shape {tuple(spatial_shape)}"
        )
        raise ValueError(msg)

    for _ in range(num_iterations):
        first = _random_origin(max_ini)
        # Resample second until non-overlapping with first.
        for _ in range(100):
            second = _random_origin(max_ini)
            if not _patches_overlap(first, second, patch_size):
                break
        locations.append((first, second))

    return locations


def _random_origin(
    max_ini: list[int],
) -> tuple[int, int, int]:
    """Sample a random patch origin."""
    coords = []
    for m in max_ini:
        if m == 0:
            coords.append(0)
        else:
            coords.append(int(torch.randint(m + 1, (1,)).item()))
    return (coords[0], coords[1], coords[2])


def _patches_overlap(
    a: tuple[int, int, int],
    b: tuple[int, int, int],
    patch_size: tuple[int, int, int],
) -> bool:
    """Check whether two axis-aligned patches overlap."""
    for ai, bi, p in zip(a, b, patch_size, strict=True):
        if ai + p <= bi or bi + p <= ai:
            return False
    return True


def _apply_swaps(
    data: Tensor,
    locations: list[tuple[tuple[int, int, int], tuple[int, int, int]]],
    patch_size: tuple[int, int, int],
) -> Tensor:
    """Swap patch pairs in a 5D tensor.

    Args:
        data: ``(B, C, I, J, K)`` tensor.
        locations: List of ``(origin_a, origin_b)`` pairs.
        patch_size: ``(pi, pj, pk)`` patch dimensions.

    Returns:
        Tensor with patches swapped.
    """
    result = data.clone()
    pi, pj, pk = patch_size

    for (ai, aj, ak), (bi, bj, bk) in locations:
        patch_a = result[:, :, ai : ai + pi, aj : aj + pj, ak : ak + pk].clone()
        patch_b = result[:, :, bi : bi + pi, bj : bj + pj, bk : bk + pk].clone()
        result[:, :, ai : ai + pi, aj : aj + pj, ak : ak + pk] = patch_b
        result[:, :, bi : bi + pi, bj : bj + pj, bk : bk + pk] = patch_a

    return result
