"""ToReferenceSpace: set spatial metadata to match a reference image."""

from __future__ import annotations

from typing import Any

import numpy as np
from torch import Tensor

from ...data.affine import AffineMatrix
from ...data.batch import SubjectsBatch
from ...data.image import Image
from ...types import TypeThreeInts
from ..transform import SpatialTransform


class ToReferenceSpace(SpatialTransform):
    r"""Set the spatial metadata of an image to match a reference space.

    This is useful for assigning meaningful spatial metadata to a
    tensor that has lost it, such as a neural network embedding or a
    downsampled feature map.  The data is left unchanged; only the
    affine is updated so that the (possibly lower-resolution) grid
    covers the same field of view, orientation, and physical center
    as the *reference* image.

    A typical use case is visualizing or resampling the output of a
    network whose spatial resolution differs from its input:

    Args:
        reference: Full-resolution reference image whose field of view
            and orientation will be matched.
        **kwargs: See [`Transform`][torchio.Transform].

    Examples:
        >>> import torch
        >>> import torchio as tio
        >>> reference = tio.ScalarImage(tensor=torch.rand(1, 64, 64, 64))
        >>> # A network embedding loses spatial metadata:
        >>> embedding = torch.rand(8, 16, 16, 16)
        >>> image = tio.ToReferenceSpace.from_tensor(embedding, reference)
        >>> image.spatial_shape
        (16, 16, 16)
    """

    def __init__(self, reference: Image, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        if not isinstance(reference, Image):
            msg = f"reference must be a TorchIO Image, got {type(reference).__name__}"
            raise TypeError(msg)
        self.reference = reference

    def make_params(self, batch: SubjectsBatch) -> dict[str, Any]:
        """No random parameters."""
        return {}

    def apply_transform(
        self,
        batch: SubjectsBatch,
        params: dict[str, Any],
    ) -> SubjectsBatch:
        """Replace each image's affine with the reference-space affine."""
        for _name, img_batch in self._get_images(batch).items():
            output_shape = (
                int(img_batch.data.shape[2]),
                int(img_batch.data.shape[3]),
                int(img_batch.data.shape[4]),
            )
            new_affine = _reference_space_affine(self.reference, output_shape)
            img_batch.affines[:] = [new_affine.clone() for _ in img_batch.affines]
        return batch

    @staticmethod
    def from_tensor(tensor: Tensor, reference: Image) -> Image:
        """Build a TorchIO image from a tensor and a reference image.

        Args:
            tensor: A `(C, I, J, K)` tensor (e.g., a network
                embedding) whose spatial metadata should match the
                reference space.
            reference: Reference image whose field of view and
                orientation will be matched.

        Returns:
            A new image with *tensor* as data and a reference-space
            affine.  The image class matches that of *reference*.
        """
        output_shape = (
            int(tensor.shape[-3]),
            int(tensor.shape[-2]),
            int(tensor.shape[-1]),
        )
        new_affine = _reference_space_affine(reference, output_shape)
        cls = type(reference)
        return cls(tensor, affine=new_affine)


def _reference_space_affine(
    reference: Image,
    output_shape: TypeThreeInts,
) -> AffineMatrix:
    """Compute an affine placing a grid in the reference field of view.

    The output grid shares the reference's physical center and
    orientation; the voxel spacing is scaled so the grid covers the
    same field of view, regardless of its resolution.

    Args:
        reference: The reference image.
        output_shape: Spatial shape `(I, J, K)` of the target grid.

    Returns:
        The reference-space affine for the target grid.
    """
    ref_affine = reference.affine
    rotation = ref_affine.direction.cpu().numpy().astype(np.float64)
    ref_spacing = np.asarray(ref_affine.spacing, dtype=np.float64)
    ref_origin = np.asarray(ref_affine.origin, dtype=np.float64)
    ref_shape = np.asarray(reference.spatial_shape, dtype=np.float64)
    new_shape = np.asarray(output_shape, dtype=np.float64)

    downsampling = ref_shape / new_shape
    new_spacing = ref_spacing * downsampling

    # Keep the physical center fixed so the grid covers the same FOV.
    center = ref_origin + rotation @ (((ref_shape - 1) / 2) * ref_spacing)
    new_origin = center - rotation @ (((new_shape - 1) / 2) * new_spacing)

    matrix = np.eye(4, dtype=np.float64)
    matrix[:3, :3] = rotation * new_spacing
    matrix[:3, 3] = new_origin
    return AffineMatrix(matrix)
