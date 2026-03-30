from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import TypeAlias
from typing import Union

import numpy as np
import torch
import torch.nn.functional as F  # noqa: N812

from ....data.image import Image
from ....data.image import ScalarImage
from ....data.io import get_rotation_and_spacing_from_affine
from ....data.subject import Subject
from ....types import TypeSpacing
from ....types import TypeTripletFloat
from ...spatial_transform import SpatialTransform

TypeShapeAffine: TypeAlias = tuple[Sequence[int], np.ndarray]
TypeTarget = Union[TypeSpacing, str, Path, Image, TypeShapeAffine, None]
ONE_MILLIMITER_ISOTROPIC = 1

_SUPPORTED_INTERPOLATIONS = ('nearest', 'linear')

_TORCH_INTERPOLATION_MODE = {
    'nearest': 'nearest',
    'linear': 'bilinear',  # grid_sample calls trilinear "bilinear" for 5D
}


class Resample(SpatialTransform):
    """Resample image to a different physical space.

    This is a powerful transform that can be used to change the image shape
    or spatial metadata, or to apply a spatial transformation.

    Args:
        target: Argument to define the output space. Can be one of:

            - Output spacing $(s_w, s_h, s_d)$, in mm. If only one value
              $s$ is specified, then $s_w = s_h = s_d = s$.

            - Path to an image that will be used as reference.

            - Instance of [`Image`][torchio.Image].

            - Name of an image key in the subject.

            - Tuple `(spatial_shape, affine)` defining the output space.

        pre_affine_name: Name of the *image key* (not subject key) storing an
            affine matrix that will be applied to the image header before
            resampling. If `None`, the image is resampled with an identity
            transform. See usage in the example below.
        image_interpolation: See Interpolation.
        label_interpolation: See Interpolation.
        scalars_only: Apply only to instances of [`ScalarImage`][torchio.ScalarImage].
            Used internally by [`RandomAnisotropy`][torchio.transforms.RandomAnisotropy].
        antialias: If `True`, apply Gaussian smoothing before
            downsampling along any dimension that will be downsampled. For example,
            if the input image has spacing (0.5, 0.5, 4) and the target
            spacing is (1, 1, 1), the image will be smoothed along the first two
            dimensions before resampling. Label maps are not smoothed.
            The standard deviations of the Gaussian kernels are computed according to
            the method described in Cardoso et al.,
            [Scale factor point spread function matching: beyond aliasing in image
            resampling
            ](https://link.springer.com/chapter/10.1007/978-3-319-24571-3_81),
            MICCAI 2015.
        **kwargs: See [`Transform`][torchio.transforms.Transform] for additional
            keyword arguments.

    Examples:
        >>> import torch
        >>> import torchio as tio
        >>> transform = tio.Resample()                      # resample all images to 1mm isotropic
        >>> transform = tio.Resample(2)                     # resample all images to 2mm isotropic
        >>> transform = tio.Resample('t1')                  # resample all images to 't1' image space
        >>> # Example: using a precomputed transform to MNI space
        >>> ref_path = tio.datasets.Colin27().t1.path  # this image is in the MNI space, so we can use it as reference/target
        >>> affine_matrix = tio.io.read_matrix('transform_to_mni.txt')  # from a NiftyReg registration. Would also work with e.g. .tfm from SimpleITK
        >>> image = tio.ScalarImage(tensor=torch.rand(1, 256, 256, 180), to_mni=affine_matrix)  # 'to_mni' is an arbitrary name
        >>> transform = tio.Resample(colin.t1.path, pre_affine_name='to_mni')  # nearest neighbor interpolation is used for label maps
        >>> transformed = transform(image)  # "image" is now in the MNI space

    Note:
        The `antialias` option is recommended when large (e.g. > 2×) downsampling
        factors are expected, particularly for offline (before training) preprocessing,
        when run times are not a concern.

    """

    def __init__(
        self,
        target: TypeTarget = ONE_MILLIMITER_ISOTROPIC,
        image_interpolation: str = 'linear',
        label_interpolation: str = 'nearest',
        pre_affine_name: str | None = None,
        scalars_only: bool = False,
        antialias: bool = False,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.target = target
        self.image_interpolation = _parse_resample_interpolation(
            image_interpolation,
        )
        self.label_interpolation = _parse_resample_interpolation(
            label_interpolation,
        )
        self.pre_affine_name = pre_affine_name
        self.scalars_only = scalars_only
        self.antialias = antialias
        self.args_names = [
            'target',
            'image_interpolation',
            'label_interpolation',
            'pre_affine_name',
            'scalars_only',
            'antialias',
        ]

    @staticmethod
    def _parse_spacing(spacing: TypeSpacing) -> tuple[float, float, float]:
        if isinstance(spacing, (int, float)):
            result = (float(spacing), float(spacing), float(spacing))
        elif isinstance(spacing, np.ndarray):
            flat = list(spacing.flat)
            if len(flat) != 3:
                message = (
                    'Target must be a string, a positive number'
                    f' or a sequence of positive numbers, not {type(spacing)}'
                )
                raise ValueError(message)
            result = (float(flat[0]), float(flat[1]), float(flat[2]))
        elif isinstance(spacing, Sequence):
            if len(spacing) != 3:
                message = (
                    'Target must be a string, a positive number'
                    f' or a sequence of positive numbers, not {type(spacing)}'
                )
                raise ValueError(message)
            values = []
            for value in spacing:
                if not isinstance(value, (int, float)):
                    message = (
                        'Target must be a string, a positive number'
                        f' or a sequence of positive numbers, not {type(spacing)}'
                    )
                    raise ValueError(message)
                values.append(float(value))
            result = (values[0], values[1], values[2])
        else:
            message = (
                'Target must be a string, a positive number'
                f' or a sequence of positive numbers, not {type(spacing)}'
            )
            raise ValueError(message)
        if any(value <= 0 for value in result):
            message = f'Spacing must be strictly positive, not "{spacing}"'
            raise ValueError(message)
        return result

    @staticmethod
    def check_affine(affine_name: str, image: Image):
        if not isinstance(affine_name, str):
            message = f'Affine name argument must be a string, not {type(affine_name)}'
            raise TypeError(message)
        if affine_name in image:
            matrix = image[affine_name]
            if not isinstance(matrix, (np.ndarray, torch.Tensor)):
                message = (
                    'The affine matrix must be a NumPy array or PyTorch'
                    f' tensor, not {type(matrix)}'
                )
                raise TypeError(message)
            if matrix.shape != (4, 4):
                message = f'The affine matrix shape must be (4, 4), not {matrix.shape}'
                raise ValueError(message)

    @staticmethod
    def check_affine_key_presence(affine_name: str, subject: Subject):
        for image in subject.get_images(intensity_only=False):
            if affine_name in image:
                return
        message = (
            f'An affine name was given ("{affine_name}"), but it was not found'
            ' in any image in the subject'
        )
        raise ValueError(message)

    def apply_transform(self, subject: Subject) -> Subject:
        use_pre_affine = self.pre_affine_name is not None
        if use_pre_affine:
            assert self.pre_affine_name is not None  # for mypy
            self.check_affine_key_presence(self.pre_affine_name, subject)

        for image in self.get_images(subject):
            # If the current image is the reference, don't resample it
            if self.target is image:
                continue

            # If the target is not a string, or is not an image in the subject,
            # do nothing
            if isinstance(self.target, str):
                try:
                    target_image = subject.get_image(self.target)
                except KeyError:
                    pass
                else:
                    if target_image is image:
                        continue

            # Choose interpolation
            if not isinstance(image, ScalarImage):
                if self.scalars_only:
                    continue
                interpolation = self.label_interpolation
            else:
                interpolation = self.image_interpolation

            # Apply given affine matrix if found in image
            if use_pre_affine and self.pre_affine_name in image:
                assert self.pre_affine_name is not None  # for mypy
                self.check_affine(self.pre_affine_name, image)
                matrix = image[self.pre_affine_name]
                if isinstance(matrix, torch.Tensor):
                    matrix = matrix.numpy()
                image.affine = matrix @ image.affine

            # Resolve the target output space
            output_shape, output_affine = self._resolve_target(
                self.target,
                image,
                subject,
            )

            # Anti-aliasing: smooth before downsampling
            input_data = image.data
            if self.antialias and isinstance(image, ScalarImage):
                input_spacing = np.sqrt((image.affine[:3, :3] ** 2).sum(axis=0))
                output_spacing = np.sqrt((output_affine[:3, :3] ** 2).sum(axis=0))
                factors = output_spacing / input_spacing
                no_downsample = factors <= 1
                factors[no_downsample] = np.nan
                sigmas = self._get_sigmas(factors, input_spacing)
                input_data = _gaussian_smooth(input_data, sigmas)

            # Build the sampling grid mapping output voxels → input voxels
            theta = _resample_grid_theta(
                image.affine,
                output_affine,
                output_shape,
            )

            mode = _TORCH_INTERPOLATION_MODE[interpolation]
            resampled = _resample_tensor(
                input_data,
                theta,
                output_shape,
                mode,
            )

            image.set_data(resampled)
            image.affine = output_affine
        return subject

    def _resolve_target(
        self,
        target: TypeTarget,
        image: Image,
        subject: Subject,
    ) -> tuple[tuple[int, int, int], np.ndarray]:
        """Resolve the target to (output_shape, output_affine).

        Args:
            target: The target specification.
            image: The image being resampled.
            subject: The parent subject.

        Returns:
            Tuple of (spatial_shape, affine_4x4).
        """
        if target is None:
            raise RuntimeError('Target cannot be None')

        if isinstance(target, (str, Path, Image)):
            if isinstance(target, Image):
                ref = target
            elif Path(target).is_file():
                ref = ScalarImage(target)
            else:
                try:
                    ref = subject.get_image(str(target))
                except KeyError as error:
                    message = (
                        f'Image name "{target}" not found in subject.'
                        f' If "{target}" is a path, it does not exist or'
                        ' permission has been denied'
                    )
                    raise ValueError(message) from error
            return ref.spatial_shape, ref.affine.copy()

        if isinstance(target, (int, float)):
            spacing = self._parse_spacing(target)
            return _compute_new_shape_affine(image, spacing)

        if isinstance(target, tuple) and len(target) == 2:
            shape = target[0]
            affine = target[1]
            if not (isinstance(shape, (list, tuple)) and len(shape) == 3):
                message = (
                    'Target shape must be a sequence of three integers, but'
                    f' "{shape}" was passed'
                )
                raise RuntimeError(message)
            if not isinstance(affine, np.ndarray) or affine.shape != (4, 4):
                message = (
                    'Target affine must have shape (4, 4) but the following'
                    f' was passed:\n{shape}'
                )
                raise RuntimeError(message)
            shape_list = list(shape)
            return (
                int(shape_list[0]),
                int(shape_list[1]),
                int(shape_list[2]),
            ), affine.copy()

        if isinstance(target, list) and len(target) == 3:
            parsed = self._parse_spacing(target)
            return _compute_new_shape_affine(image, parsed)

        if isinstance(target, np.ndarray) and target.size == 3:
            flat = target.flat
            parsed = self._parse_spacing(
                (float(flat[0]), float(flat[1]), float(flat[2]))
            )
            return _compute_new_shape_affine(image, parsed)

        if (
            isinstance(target, tuple)
            and len(target) == 3
            and all(isinstance(v, (int, float)) for v in target)
        ):
            target_list: list[float] = [
                float(v) for v in target if isinstance(v, (int, float))
            ]
            parsed = self._parse_spacing(
                (target_list[0], target_list[1], target_list[2])
            )
            return _compute_new_shape_affine(image, parsed)

        raise RuntimeError(f'Target not understood: "{target}"')

    @staticmethod
    def _get_sigmas(downsampling_factor: np.ndarray, spacing: np.ndarray) -> np.ndarray:
        """Compute optimal standard deviation for Gaussian kernel.

        From Cardoso et al., [Scale factor point spread function matching:
        beyond aliasing in image resampling
        ](https://link.springer.com/chapter/10.1007/978-3-319-24571-3_81),
        MICCAI 2015.

        Args:
            downsampling_factor: Array with the downsampling factor for each
                dimension.
            spacing: Array with the spacing of the input image in mm.
        """
        k = downsampling_factor
        # Equation from top of page 678 of proceedings (4/9 in the PDF)
        variance = (k**2 - 1) * (2 * np.sqrt(2 * np.log(2))) ** (-2)
        sigma = spacing * np.sqrt(variance)
        return sigma


def _compute_new_shape_affine(
    image: Image,
    spacing: TypeTripletFloat,
) -> tuple[tuple[int, int, int], np.ndarray]:
    """Compute output shape and affine for a target spacing.

    Args:
        image: The input image.
        spacing: Target spacing in mm.

    Returns:
        Tuple of (spatial_shape, affine_4x4).
    """
    old_spacing = np.sqrt((image.affine[:3, :3] ** 2).sum(axis=0))
    new_spacing = np.array(spacing, dtype=float)
    old_shape = np.array(image.spatial_shape, dtype=float)

    # Compute new size, keeping singleton dimensions
    new_shape = np.floor(old_shape * old_spacing / new_spacing)
    new_shape[old_shape == 1] = 1

    # Compute rotation/direction from affine
    rotation, _ = get_rotation_and_spacing_from_affine(image.affine)
    old_origin = image.affine[:3, 3]

    # Recompute origin to keep the image centered
    old_center = old_origin + rotation @ ((old_shape - 1) / 2 * old_spacing)
    new_origin = old_center - rotation @ ((new_shape - 1) / 2 * new_spacing)

    # Build new affine
    new_affine = np.eye(4)
    new_affine[:3, :3] = rotation * new_spacing
    new_affine[:3, 3] = new_origin

    w, h, d = int(new_shape[0]), int(new_shape[1]), int(new_shape[2])
    return (w, h, d), new_affine


def _resample_grid_theta(
    input_affine: np.ndarray,
    output_affine: np.ndarray,
    output_shape: tuple[int, int, int],
) -> torch.Tensor:
    """Build a 3×4 theta for affine_grid to resample from input to output space.

    Maps output normalized coords → input normalized coords.

    Args:
        input_affine: 4×4 input image affine (voxel → world).
        output_affine: 4×4 output image affine (voxel → world).
        output_shape: (W, H, D) output spatial shape.

    Returns:
        A (1, 3, 4) theta tensor.
    """
    # Output voxel → world → input voxel
    input_affine_inv = np.linalg.inv(input_affine)
    m_voxel = input_affine_inv @ output_affine  # maps output voxel → input voxel

    # We don't use affine_grid's theta because the input and output shapes
    # differ. Instead we'll build the grid manually in _resample_tensor.
    # Return m_voxel as a tensor for use there.
    return torch.as_tensor(m_voxel, dtype=torch.float32).unsqueeze(0)


def _resample_tensor(
    tensor: torch.Tensor,
    m_voxel_batch: torch.Tensor,
    output_shape: tuple[int, int, int],
    mode: str,
) -> torch.Tensor:
    """Resample a tensor from input space to output space.

    Args:
        tensor: Input tensor of shape (C, W_in, H_in, D_in).
        m_voxel_batch: (1, 4, 4) matrix mapping output voxels → input voxels.
        output_shape: Target (W_out, H_out, D_out).
        mode: Interpolation mode for grid_sample.

    Returns:
        Resampled tensor of shape (C, W_out, H_out, D_out).
    """
    m_voxel = m_voxel_batch.squeeze(0).numpy()
    in_w, in_h, in_d = tensor.shape[1], tensor.shape[2], tensor.shape[3]
    out_w, out_h, out_d = output_shape

    # Build grid of output voxel coordinates (out_W, out_H, out_D, 3)
    gw = torch.arange(out_w, dtype=torch.float32)
    gh = torch.arange(out_h, dtype=torch.float32)
    gd = torch.arange(out_d, dtype=torch.float32)
    grid_w, grid_h, grid_d = torch.meshgrid(gw, gh, gd, indexing='ij')
    # (out_W, out_H, out_D, 4) homogeneous coords
    ones = torch.ones_like(grid_w)
    output_coords = torch.stack([grid_w, grid_h, grid_d, ones], dim=-1)

    # Transform to input voxel coordinates
    m_t = torch.as_tensor(m_voxel, dtype=torch.float32)
    # (W, H, D, 4) @ (4, 4).T → (W, H, D, 4)
    input_coords = output_coords @ m_t.T
    input_voxels = input_coords[..., :3]  # (out_W, out_H, out_D, 3)

    # Normalize input voxels to [-1, 1] for grid_sample
    sizes = torch.tensor(
        [max(in_w - 1, 1), max(in_h - 1, 1), max(in_d - 1, 1)],
        dtype=torch.float32,
    )
    grid_norm = 2.0 * input_voxels / sizes - 1.0  # (out_W, out_H, out_D, 3)

    # Permute to grid_sample layout: (D, H, W, 3) with coords (x=W, y=H, z=D)
    grid_dhw = grid_norm.permute(2, 1, 0, 3).unsqueeze(0)  # (1, out_D, out_H, out_W, 3)

    # Prepare input: (C, W, H, D) → (1, C, D, H, W)
    input_5d = tensor.permute(0, 3, 2, 1).unsqueeze(0).float()

    sampled = F.grid_sample(
        input_5d,
        grid_dhw,
        mode=mode,
        padding_mode='zeros',
        align_corners=True,
    )

    # (1, C, out_D, out_H, out_W) → (C, out_W, out_H, out_D)
    return sampled.squeeze(0).permute(0, 3, 2, 1)


def _gaussian_smooth(
    tensor: torch.Tensor,
    sigmas: np.ndarray,
) -> torch.Tensor:
    """Apply Gaussian smoothing to a tensor.

    Args:
        tensor: Input tensor of shape (C, W, H, D).
        sigmas: Standard deviations in mm for each spatial dimension.
            NaN means no smoothing in that dimension.

    Returns:
        Smoothed tensor.
    """
    sigmas = sigmas.copy()
    sigmas[np.isnan(sigmas)] = 0.0

    if np.all(sigmas == 0):
        return tensor

    result = tensor.float()
    for dim_idx in range(3):
        sigma = sigmas[dim_idx]
        if sigma <= 0:
            continue
        # Kernel radius: 3 sigma, must be odd
        radius = max(int(np.ceil(3 * sigma)), 1)
        kernel_size = 2 * radius + 1
        x = torch.arange(kernel_size, dtype=torch.float32) - radius
        kernel_1d = torch.exp(-0.5 * (x / sigma) ** 2)
        kernel_1d = kernel_1d / kernel_1d.sum()

        # Reshape for conv along spatial dim (dim_idx + 1 because of C dim)
        # tensor is (C, W, H, D), spatial dims are 1, 2, 3
        shape = [1] * 4
        shape[dim_idx + 1] = kernel_size
        kernel = kernel_1d.reshape(shape)

        # Expand for per-channel conv
        c = result.shape[0]
        # We'll apply per-channel using F.conv3d
        # (C, W, H, D) → (1, C, W, H, D)
        # Actually easier to pad and convolve along the right dim
        pad_amounts = [
            0
        ] * 6  # (D_before, D_after, H_before, H_after, W_before, W_after)
        # F.pad pads from last dim backwards
        pad_idx = 2 * (2 - dim_idx)
        pad_amounts[pad_idx] = radius
        pad_amounts[pad_idx + 1] = radius

        padded = F.pad(
            result.unsqueeze(0),  # (1, C, W, H, D)
            pad_amounts,
            mode='replicate',
        )

        # Build 3D kernel: (C_out, C_in/groups, kW, kH, kD)
        k3d = torch.zeros(c, 1, 1, 1, 1)
        if dim_idx == 0:
            k3d = kernel.unsqueeze(0).expand(c, 1, kernel_size, 1, 1)
        elif dim_idx == 1:
            k3d = kernel.unsqueeze(0).expand(c, 1, 1, kernel_size, 1)
        else:
            k3d = kernel.unsqueeze(0).expand(c, 1, 1, 1, kernel_size)

        result = F.conv3d(padded, k3d, groups=c).squeeze(0)

    return result


def _parse_resample_interpolation(interpolation: str) -> str:
    """Validate that interpolation is supported by the PyTorch backend."""
    if not isinstance(interpolation, str):
        itype = type(interpolation)
        raise TypeError(f'Interpolation must be a string, not {itype}')
    interpolation = interpolation.lower()
    if interpolation not in _SUPPORTED_INTERPOLATIONS:
        message = (
            f'Interpolation "{interpolation}" is not supported.'
            f' Supported values are: {list(_SUPPORTED_INTERPOLATIONS)}'
        )
        raise ValueError(message)
    return interpolation
