"""PCA: dimensionality reduction of multi-channel images."""

from __future__ import annotations

from typing import Any

import torch
from einops import rearrange
from torch import Tensor

from ..data.batch import SubjectsBatch
from .transform import IntensityTransform


class PCA(IntensityTransform):
    r"""Apply PCA to reduce the channel dimension.

    Reshapes a $(C, I, J, K)$ image to $(N, I \cdot J \cdot K)$,
    performs PCA, and reshapes back to
    $(\text{num\_components}, I, J, K)$.

    This is useful for visualizing high-dimensional feature maps
    (e.g., neural network embeddings) as RGB images.

    The implementation uses :func:`torch.pca_lowrank`, so no
    external dependencies are needed.

    Args:
        num_components: Number of principal components to keep.
        whiten: If ``True``, normalize each component to unit
            variance.
        normalize: If ``True``, divide all components by the
            standard deviation of the first component.
        values_range: Linear mapping range for normalization to
            $[0, 1]$.  The default $(-2.3, 2.3)$ covers
            $\approx 99\%$ of a standard normal distribution.
        clip: If ``True``, clip output to $[0, 1]$ after
            normalization.
        **kwargs: See [`Transform`][torchio.Transform].

    Examples:
        >>> import torchio as tio
        >>> transform = tio.PCA(num_components=3)
    """

    def __init__(
        self,
        num_components: int = 3,
        *,
        whiten: bool = True,
        normalize: bool = True,
        values_range: tuple[float, float] = (-2.3, 2.3),
        clip: bool = True,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        if num_components < 1:
            msg = f"num_components must be >= 1, got {num_components}"
            raise ValueError(msg)
        self.num_components = num_components
        self.whiten = whiten
        self.normalize = normalize
        self.values_range = values_range
        self.clip = clip

    def make_params(self, batch: SubjectsBatch) -> dict[str, Any]:
        """No random parameters."""
        return {}

    def apply_transform(
        self,
        batch: SubjectsBatch,
        params: dict[str, Any],
    ) -> SubjectsBatch:
        """Apply PCA to each selected image."""
        for _name, img_batch in self._get_images(batch).items():
            results = []
            for i in range(img_batch.batch_size):
                results.append(self._pca_single(img_batch.data[i]))
            img_batch.data = torch.stack(results)
        return batch

    def _pca_single(self, tensor: Tensor) -> Tensor:
        """Apply PCA to a single ``(C, I, J, K)`` tensor.

        Args:
            tensor: Input with *C* channels.

        Returns:
            Tensor with *num_components* channels.
        """
        c, si, sj, sk = tensor.shape
        if c < self.num_components:
            msg = (
                f"Image has {c} channels but num_components="
                f"{self.num_components}. Need at least as many "
                "channels as components."
            )
            raise ValueError(msg)

        # (C, I*J*K) → (voxels, channels)
        flat = rearrange(tensor.float(), "c i j k -> (i j k) c")

        # Center.
        mean = flat.mean(dim=0, keepdim=True)
        centered = flat - mean

        # PCA via torch.pca_lowrank.
        _u, s, v = torch.pca_lowrank(centered, q=self.num_components)
        # Project: (voxels, channels) @ (channels, n_comp) → (voxels, n_comp)
        projected = centered @ v

        if self.whiten:
            # s are singular values; variance ≈ s² / (n - 1)
            n = flat.shape[0]
            denom = (n - 1) ** 0.5 if n > 1 else 1.0
            std = s / denom
            std = std.clamp(min=1e-8)
            projected = projected / std.unsqueeze(0)

        if self.normalize and projected.shape[1] > 0:
            first_std = projected[:, 0].std().clamp(min=1e-8)
            projected = projected / first_std

        # Map values_range to [0, 1].
        lo, hi = self.values_range
        projected = (projected - lo) / (hi - lo)

        if self.clip:
            projected = projected.clamp(0, 1)

        # Reshape back: (voxels, n_comp) → (n_comp, I, J, K)
        result = rearrange(
            projected,
            "(i j k) c -> c i j k",
            i=si,
            j=sj,
            k=sk,
        )
        return result
