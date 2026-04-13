"""Motion: simulate MRI motion artifacts via k-space corruption."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
import torch.nn.functional as functional
from einops import rearrange
from torch import Tensor

from ..data.batch import SubjectsBatch
from .parameter_range import to_range
from .transform import IntensityTransform


class Motion(IntensityTransform):
    r"""Simulate MRI motion artifacts.

    Motion during MR acquisition corrupts different segments of
    k-space with different rigid-body transforms, producing
    characteristic ringing and blurring.  This implementation follows
    [Shaw et al., 2019](http://proceedings.mlr.press/v102/shaw19a.html).

    The simulation:

    1. Splits k-space into *num_transforms* + 1 segments along a
       random axis.
    2. For each segment, applies a random rigid-body transform to
       the image and fills the corresponding k-space lines from the
       transformed image.
    3. Reconstructs the corrupted image via inverse FFT.

    Args:
        degrees: Rotation range in degrees.  A scalar $d$ means
            $\theta_i \sim \mathcal{U}(-d, d)$.  A 2-tuple $(a, b)$
            means $\theta_i \sim \mathcal{U}(a, b)$.
        translation: Translation range in mm, same convention.
        num_transforms: Number of inter-segment motion events.
            More transforms produce more distortion.
        **kwargs: See [`Transform`][torchio.Transform].

    Warning:
        Large numbers of transforms increase execution time
        significantly for 3D volumes.

    Examples:
        >>> import torchio as tio
        >>> transform = tio.Motion()
        >>> transform = tio.Motion(degrees=15, translation=10, num_transforms=4)
    """

    def __init__(
        self,
        *,
        degrees: float | tuple[float, float] = 10.0,
        translation: float | tuple[float, float] = 10.0,
        num_transforms: int = 2,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.degrees = to_range(degrees)
        self.translation = to_range(translation)
        if not isinstance(num_transforms, int) or num_transforms < 1:
            msg = f"num_transforms must be a positive int, got {num_transforms}"
            raise ValueError(msg)
        self.num_transforms = num_transforms

    def make_params(self, batch: SubjectsBatch) -> dict[str, Any]:
        """Sample motion parameters."""
        transforms = []
        for _ in range(self.num_transforms):
            transforms.append(
                {
                    "degrees": self.degrees.sample(),
                    "translation": self.translation.sample(),
                }
            )
        return {
            "transforms": transforms,
            "seed": int(torch.randint(0, 2**31, (1,)).item()),
        }

    def apply_transform(
        self,
        batch: SubjectsBatch,
        params: dict[str, Any],
    ) -> SubjectsBatch:
        """Corrupt each selected image with simulated motion."""
        motion_transforms = params["transforms"]
        for _name, img_batch in self._get_images(batch).items():
            img_batch.data = _apply_motion(
                img_batch.data,
                motion_transforms,
            )
        return batch


def _apply_motion(
    data: Tensor,
    motion_transforms: list[dict[str, tuple[float, float, float]]],
) -> Tensor:
    """Apply motion corruption to a 5D tensor.

    Args:
        data: ``(B, C, I, J, K)`` image tensor.
        motion_transforms: List of dicts with ``degrees`` and
            ``translation`` 3-tuples.

    Returns:
        Motion-corrupted ``(B, C, I, J, K)`` tensor.
    """
    result = data.float()
    num_transforms = len(motion_transforms)

    for b in range(result.shape[0]):
        for c in range(result.shape[1]):
            channel = result[b, c]
            shape = channel.shape
            spectrum = torch.fft.fftn(channel)

            # Split k-space along the first axis into segments.
            num_segments = num_transforms + 1
            segment_size = shape[0] // num_segments

            for seg_idx in range(1, num_segments):
                transform = motion_transforms[seg_idx - 1]
                moved = _apply_rigid_transform(
                    channel,
                    transform["degrees"],
                    transform["translation"],
                )
                moved_spectrum = torch.fft.fftn(moved)
                start = seg_idx * segment_size
                end = (
                    (seg_idx + 1) * segment_size
                    if seg_idx < num_segments - 1
                    else shape[0]
                )
                spectrum[start:end] = moved_spectrum[start:end]

            reconstructed = torch.fft.ifftn(spectrum)
            result[b, c] = reconstructed.real

    return result


def _apply_rigid_transform(
    tensor: Tensor,
    degrees: tuple[float, float, float],
    translation: tuple[float, float, float],
) -> Tensor:
    """Apply a rigid-body transform to a 3D tensor using affine_grid.

    Args:
        tensor: ``(I, J, K)`` tensor.
        degrees: Euler angles in degrees.
        translation: Translation in voxels (approximation).

    Returns:
        Transformed ``(I, J, K)`` tensor.
    """
    shape = tensor.shape
    radians = [np.radians(d) for d in degrees]
    rx, ry, rz = radians

    cos_x, sin_x = np.cos(rx), np.sin(rx)
    cos_y, sin_y = np.cos(ry), np.sin(ry)
    cos_z, sin_z = np.cos(rz), np.sin(rz)

    r_x = torch.tensor(
        [
            [1, 0, 0],
            [0, cos_x, -sin_x],
            [0, sin_x, cos_x],
        ],
        dtype=torch.float32,
    )

    r_y = torch.tensor(
        [
            [cos_y, 0, sin_y],
            [0, 1, 0],
            [-sin_y, 0, cos_y],
        ],
        dtype=torch.float32,
    )

    r_z = torch.tensor(
        [
            [cos_z, -sin_z, 0],
            [sin_z, cos_z, 0],
            [0, 0, 1],
        ],
        dtype=torch.float32,
    )

    rotation = (r_z @ r_y @ r_x).to(tensor.device)

    # Normalize translation to [-1, 1] grid coordinates.
    shape_t = torch.tensor(shape, dtype=torch.float32)
    t_normalized = torch.tensor(translation, dtype=torch.float32) / (shape_t / 2)
    t_normalized = t_normalized.to(tensor.device)

    # Build 3x4 affine matrix for affine_grid.
    theta = torch.zeros(1, 3, 4, device=tensor.device)
    theta[0, :3, :3] = rotation
    theta[0, :3, 3] = t_normalized

    # affine_grid expects (N, C, D, H, W) input size.
    grid = functional.affine_grid(
        theta,
        [1, 1, shape[0], shape[1], shape[2]],
        align_corners=True,
    )
    input_5d = rearrange(tensor, "i j k -> 1 1 i j k").float()
    output = functional.grid_sample(
        input_5d,
        grid,
        mode="bilinear",
        padding_mode="zeros",
        align_corners=True,
    )
    return rearrange(output, "1 1 i j k -> i j k")
