from __future__ import annotations

from typing import Any

import numpy as np
import torch
import torch.nn.functional as F  # noqa: N812

from ....constants import INTENSITY
from ....constants import TYPE
from ....data.subject import Subject
from ...spatial_transform import SpatialTransform
from .. import RandomTransform
from .random_affine import Affine
from .random_affine import _physical_to_grid_theta
from .random_elastic_deformation import _TORCH_INTERPOLATION_MODE
from .random_elastic_deformation import ElasticDeformation
from .random_elastic_deformation import _check_folding
from .random_elastic_deformation import _resample_with_displacement
from .random_elastic_deformation import _upsample_displacement_field


class RandomAffineElasticDeformation(RandomTransform, SpatialTransform):
    r"""Apply a RandomAffine and RandomElasticDeformation simultaneously.

    Composes both transforms into a single resampling operation using pure
    PyTorch. For additional details on the transformations, see
    [`RandomAffine`][torchio.transforms.RandomAffine]
    and [`RandomElasticDeformation`][torchio.transforms.RandomElasticDeformation]

    Args:
        affine_first: Apply affine before elastic deformation.
        affine_kwargs: See [`RandomAffine`][torchio.transforms.RandomAffine] for kwargs.
        elastic_kwargs: See [`RandomElasticDeformation`][torchio.transforms.RandomElasticDeformation]
            for kwargs.
        **kwargs: See [`Transform`][torchio.transforms.Transform] for additional
            keyword arguments.

    Examples:
        >>> import torchio as tio
        >>> image = tio.datasets.Colin27().t1
        >>> affine_kwargs = {'scales': (0.9, 1.2), 'degrees': 15}
        >>> elastic_kwargs = {'max_displacement': (17, 12, 2)}
        >>> transform = tio.RandomAffineElasticDeformation(
        ...     affine_kwargs,
        ...     elastic_kwargs
        ... )
        >>> transformed = transform(image)

    """

    def __init__(
        self,
        affine_first: bool = True,
        affine_kwargs: dict[str, Any] | None = None,
        elastic_kwargs: dict[str, Any] | None = None,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.affine_first = affine_first

        # Avoid circular imports
        from .random_affine import RandomAffine
        from .random_elastic_deformation import RandomElasticDeformation

        self.affine_kwargs = affine_kwargs or {}
        self.random_affine = RandomAffine(**self.affine_kwargs)

        self.elastic_kwargs = elastic_kwargs or {}
        self.random_elastic = RandomElasticDeformation(**self.elastic_kwargs)

    def get_params(self):
        affine_params = self.random_affine.get_params(
            self.random_affine.scales,
            self.random_affine.degrees,
            self.random_affine.translation,
            self.random_affine.isotropic,
        )
        elastic_params = self.random_elastic.get_params(
            self.random_elastic.num_control_points,
            self.random_elastic.max_displacement,
            self.random_elastic.num_locked_borders,
        )
        return affine_params, elastic_params

    def apply_transform(self, subject: Subject) -> Subject:
        affine_params, elastic_params = self.get_params()

        scaling_params, rotation_params, translation_params = affine_params
        affine_params = {
            'scales': scaling_params.tolist(),
            'degrees': rotation_params.tolist(),
            'translation': translation_params.tolist(),
            'center': self.random_affine.center,
            'default_pad_value': self.random_affine.default_pad_value,
            'image_interpolation': self.random_affine.image_interpolation,
            'label_interpolation': self.random_affine.label_interpolation,
            'check_shape': self.random_affine.check_shape,
        }

        elastic_params = {
            'control_points': elastic_params,
            'max_displacement': self.random_elastic.max_displacement,
            'image_interpolation': self.random_elastic.image_interpolation,
            'label_interpolation': self.random_elastic.label_interpolation,
        }

        transform = AffineElasticDeformation(
            affine_first=self.affine_first,
            affine_params=affine_params,
            elastic_params=elastic_params,
            **self._get_base_args(),
        )
        transformed = transform(subject)
        assert isinstance(transformed, Subject)
        return transformed


class AffineElasticDeformation(SpatialTransform):
    r"""Apply an Affine and ElasticDeformation simultaneously.

    Composes the affine and elastic transforms into a single resampling
    operation using pure PyTorch.

    Args:
        affine_first: Apply affine before elastic deformation.
        affine_params: See [`Affine`][torchio.transforms.augmentation.Affine] for params.
        elastic_params: See
            [`ElasticDeformation`][torchio.transforms.augmentation.ElasticDeformation] for params.
        **kwargs: See [`Transform`][torchio.transforms.Transform] for additional
            keyword arguments.
    """

    def __init__(
        self,
        affine_first: bool,
        affine_params: dict[str, Any],
        elastic_params: dict[str, Any],
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.affine_first = affine_first

        self.affine_params = affine_params
        self._affine = Affine(
            **self.affine_params,
            **kwargs,
        )
        self.elastic_params = elastic_params
        self._elastic = ElasticDeformation(
            **self.elastic_params,
            **kwargs,
        )

        self.args_names = ['affine_first', 'affine_params', 'elastic_params']

    def apply_transform(self, subject: Subject) -> Subject:
        if self._affine.check_shape:
            subject.check_consistent_spatial_shape()
        default_value: float

        for image in self.get_images(subject):
            # Build the affine theta (normalized 3×4)
            forward_transform = self._affine.get_affine_transform(image)
            theta = _physical_to_grid_theta(
                forward_transform,
                image.affine,
                image.spatial_shape,
            )

            # Build the elastic displacement grid
            spacing = np.sqrt((image.affine[:3, :3] ** 2).sum(axis=0))
            control_points = self._elastic.control_points.copy()
            if self._elastic.invert_transform:
                control_points *= -1
            if image.is_2d():
                control_points[..., -1] = 0
            _check_folding(
                control_points,
                self._elastic.max_displacement,
                image.spatial_shape,
                spacing,
            )
            coarse_field = torch.as_tensor(control_points, dtype=torch.float32)
            displacement = _upsample_displacement_field(
                coarse_field, image.spatial_shape
            )

            # Build the combined sampling grid
            grid = _compose_affine_displacement(
                theta,
                displacement,
                image.spatial_shape,
                spacing,
                affine_first=self.affine_first,
            )

            transformed_tensors = []
            for tensor in image.data:
                if image[TYPE] != INTENSITY:
                    interpolation = self._affine.label_interpolation
                    default_value = self._affine.default_pad_label
                else:
                    interpolation = self._affine.image_interpolation
                    default_value = self._affine.get_default_pad_value(tensor)

                mode = _TORCH_INTERPOLATION_MODE[interpolation]
                transformed_tensor = _resample_with_displacement(
                    tensor.unsqueeze(0),
                    grid,
                    mode,
                    default_value,
                )
                transformed_tensors.append(transformed_tensor.squeeze(0))
            image.set_data(torch.stack(transformed_tensors))
        return subject


def _compose_affine_displacement(
    theta: torch.Tensor,
    displacement: torch.Tensor,
    spatial_shape: tuple[int, int, int],
    spacing: np.ndarray,
    affine_first: bool,
) -> torch.Tensor:
    """Compose an affine transform with a displacement field into a single grid.

    Args:
        theta: Affine parameters (1, 3, 4) for affine_grid.
        displacement: Dense displacement field (W, H, D, 3) in mm.
        spatial_shape: Image spatial shape (W, H, D).
        spacing: Voxel spacing in mm.
        affine_first: If True, apply affine first then elastic.

    Returns:
        Combined sampling grid of shape (1, D, H, W, 3).
    """
    w, h, d = spatial_shape

    # Build affine sampling grid (in normalized coords)
    affine_grid = F.affine_grid(
        theta,
        [1, 1, d, h, w],
        align_corners=True,
    )
    # affine_grid: (1, D, H, W, 3)

    # Convert displacement from mm to normalized coords
    spacing_t = torch.as_tensor(spacing, dtype=torch.float32)
    disp_voxels = displacement / spacing_t
    sizes = torch.tensor(
        [max(w - 1, 1), max(h - 1, 1), max(d - 1, 1)],
        dtype=torch.float32,
    )
    disp_norm = 2.0 * disp_voxels / sizes  # (W, H, D, 3)
    # Permute (W, H, D, 3) → (D, H, W, 3)
    disp_dhw = disp_norm.permute(2, 1, 0, 3).unsqueeze(0)

    if affine_first:
        # Affine first: affine_grid gives us where to sample from,
        # then we add the elastic displacement on top
        grid = affine_grid + disp_dhw
    else:
        # Elastic first: start with identity + displacement, then apply affine.
        # Build identity grid
        identity = F.affine_grid(
            torch.eye(3, 4, dtype=torch.float32).unsqueeze(0),
            [1, 1, d, h, w],
            align_corners=True,
        )
        elastic_grid = identity + disp_dhw
        # Now apply affine to the elastic grid coords
        # theta maps output coords to input coords.
        # elastic_grid gives us intermediate coords; we need to apply
        # the affine mapping to those.
        mat = theta[:, :3, :3]  # (1, 3, 3)
        trans = theta[:, :3, 3:]  # (1, 3, 1)
        flat = elastic_grid.reshape(1, -1, 3)  # (1, N, 3)
        transformed = (flat @ mat.transpose(1, 2)) + trans.transpose(1, 2)
        grid = transformed.reshape(1, d, h, w, 3)

    return grid
