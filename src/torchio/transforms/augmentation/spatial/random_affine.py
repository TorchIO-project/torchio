from __future__ import annotations

from numbers import Number

import numpy as np
import torch
import torch.nn.functional as F  # noqa: N812

from ....constants import INTENSITY
from ....constants import TYPE
from ....data.subject import Subject
from ....types import TypeRangeFloat
from ....types import TypeSextetFloat
from ....types import TypeTripletFloat
from ....utils import to_tuple
from ...spatial_transform import SpatialTransform
from .. import RandomTransform

TypeOneToSixFloat = TypeRangeFloat | TypeTripletFloat | TypeSextetFloat

_SUPPORTED_INTERPOLATIONS = ('nearest', 'linear')

_TORCH_INTERPOLATION_MODE = {
    'nearest': 'nearest',
    'linear': 'bilinear',  # grid_sample calls trilinear "bilinear" for 5D
}


def _euler_to_rotation_matrix(
    degrees: np.ndarray,
) -> np.ndarray:
    """Build a 3×3 rotation matrix from Euler angles (X→Y→Z order).

    This matches SimpleITK's Euler3DTransform convention.

    Args:
        degrees: Array of shape (3,) with rotation angles in degrees
            around X, Y, Z axes respectively.

    Returns:
        A 3×3 rotation matrix as a numpy array.
    """
    radians = np.radians(degrees)
    rx, ry, rz = radians

    cos_x, sin_x = np.cos(rx), np.sin(rx)
    cos_y, sin_y = np.cos(ry), np.sin(ry)
    cos_z, sin_z = np.cos(rz), np.sin(rz)

    # Rotation around X
    r_x = np.array(
        [
            [1, 0, 0],
            [0, cos_x, -sin_x],
            [0, sin_x, cos_x],
        ]
    )

    # Rotation around Y
    r_y = np.array(
        [
            [cos_y, 0, sin_y],
            [0, 1, 0],
            [-sin_y, 0, cos_y],
        ]
    )

    # Rotation around Z
    r_z = np.array(
        [
            [cos_z, -sin_z, 0],
            [sin_z, cos_z, 0],
            [0, 0, 1],
        ]
    )

    return r_z @ r_y @ r_x


def _build_physical_transform(
    scales: np.ndarray,
    degrees: np.ndarray,
    translation: np.ndarray,
    center_ras: np.ndarray | None,
) -> np.ndarray:
    """Build a 4×4 forward physical-space transform (input world → output world).

    Composes: translate to center → scale → rotate → translate back + translate.

    Args:
        scales: Array of shape (3,) with scaling factors per axis.
        degrees: Array of shape (3,) with rotation angles in degrees.
        translation: Array of shape (3,) with translation in mm.
        center_ras: Array of shape (3,) with center of rotation/scaling
            in RAS world coordinates, or None for origin.

    Returns:
        A 4×4 forward transform matrix.
    """
    rotation = _euler_to_rotation_matrix(degrees)
    scale_matrix = np.diag(scales)

    # Combined rotation and scaling
    rs = rotation @ scale_matrix

    transform = np.eye(4)
    transform[:3, :3] = rs

    if center_ras is not None:
        center = np.asarray(center_ras, dtype=float)
        # T_center @ RS @ T_{-center}: shift to center, apply RS, shift back
        transform[:3, 3] = center - rs @ center

    transform[:3, 3] += translation

    return transform


def _physical_to_grid_theta(
    forward_transform: np.ndarray,
    image_affine: np.ndarray,
    spatial_shape: tuple[int, int, int],
) -> torch.Tensor:
    """Convert a 4×4 physical transform to a 3×4 theta for affine_grid.

    The chain is:
        output voxel → output world → input world → input voxel
        M_voxel = A⁻¹ @ T⁻¹ @ A

    Then convert to the normalized [-1, 1] coordinate system that
    ``torch.nn.functional.affine_grid`` expects.

    Args:
        forward_transform: 4×4 forward physical transform (input → output).
        image_affine: 4×4 image affine matrix (voxel → world).
        spatial_shape: (W, H, D) spatial dimensions.

    Returns:
        A (1, 3, 4) float32 theta tensor.
    """
    inverse_transform = np.linalg.inv(forward_transform)
    affine_inv = np.linalg.inv(image_affine)

    # Output voxel → input voxel
    m_voxel = affine_inv @ inverse_transform @ image_affine

    # Normalization: voxel index i ∈ [0, N-1] → normalized coord ∈ [-1, 1]
    # n = 2*i/(N-1) - 1  →  i = (n + 1) * (N-1) / 2
    # N_to_norm maps voxel to normalized; N_from_norm maps normalized to voxel
    w, h, d = spatial_shape

    # Matrix that converts voxel indices to [-1, 1] coords
    # and its inverse
    n_to_norm = np.diag(
        [2.0 / max(w - 1, 1), 2.0 / max(h - 1, 1), 2.0 / max(d - 1, 1), 1.0]
    )
    n_to_norm[:3, 3] = -1.0

    n_from_norm = np.linalg.inv(n_to_norm)

    # Full chain: normalized output → voxel output → voxel input → normalized input
    # But affine_grid maps FROM output normalized TO input normalized
    # grid[i,j,k] = coordinates in input normalized space
    theta_4x4 = n_to_norm @ m_voxel @ n_from_norm

    # grid_sample uses (x, y, z) = (W, H, D) indexing for coordinates
    # but grid shape is (N, D, H, W, 3), and the 3 coords are (x=W, y=H, z=D)
    # TorchIO stores data as (C, W, H, D), which we'll permute to (C, D, H, W)
    # for grid_sample. So after permutation:
    #   dim0=D, dim1=H, dim2=W
    #   grid coords: x=W(dim2), y=H(dim1), z=D(dim0)
    #
    # Our theta_4x4 maps (W, H, D) normalized coords to (W, H, D) normalized coords.
    # We need to reorder to (x=W, y=H, z=D) which is the same ordering.
    # However, affine_grid expects theta that maps output grid positions
    # to input sampling positions, where positions are (x, y, z).
    # Since our data after permute is (D, H, W), the grid has shape (D, H, W, 3)
    # and coords are (x=W, y=H, z=D).
    #
    # Our m_voxel is in (W, H, D) space. After permute, the axes are (D, H, W).
    # We need theta in (W, H, D) order since grid coords are (x=W, y=H, z=D).
    # So theta_4x4 is already in the right order!

    theta = torch.as_tensor(theta_4x4[:3, :], dtype=torch.float32)
    return theta.unsqueeze(0)  # (1, 3, 4)


def _resample_tensor(
    tensor: torch.Tensor,
    theta: torch.Tensor,
    mode: str,
    padding_value: float,
) -> torch.Tensor:
    """Resample a 3D tensor using affine_grid + grid_sample.

    Handles TorchIO's (C, W, H, D) layout by permuting to PyTorch's
    (N, C, D, H, W) convention.

    Args:
        tensor: Input tensor of shape (C, W, H, D).
        theta: Affine parameters of shape (1, 3, 4).
        mode: Interpolation mode for grid_sample ('nearest' or 'trilinear').
        padding_value: Fill value for out-of-bounds regions.

    Returns:
        Resampled tensor of shape (C, W, H, D).
    """
    # (C, W, H, D) → (C, D, H, W) for grid_sample
    tensor_dhw = tensor.permute(0, 3, 2, 1)

    # Add batch dimension: (1, C, D, H, W)
    input_5d = tensor_dhw.unsqueeze(0).float()

    grid = F.affine_grid(theta, list(input_5d.shape), align_corners=True)

    sampled = F.grid_sample(
        input_5d,
        grid,
        mode=mode,
        padding_mode='zeros',
        align_corners=True,
    )

    if padding_value != 0.0:
        # Create a mask of valid (in-bounds) regions by sampling an all-ones tensor
        ones = torch.ones_like(input_5d)
        mask = F.grid_sample(
            ones,
            grid,
            mode='nearest',
            padding_mode='zeros',
            align_corners=True,
        )
        # Where mask is 0, fill with padding_value
        sampled = torch.where(mask > 0.5, sampled, torch.tensor(padding_value))

    # Remove batch dim and permute back: (1, C, D, H, W) → (C, W, H, D)
    result = sampled.squeeze(0).permute(0, 3, 2, 1)
    return result


def _get_borders_mean(tensor: torch.Tensor, filter_otsu: bool = False) -> float:
    """Compute mean of border voxels, optionally filtered by Otsu threshold.

    Args:
        tensor: 3D tensor of shape (W, H, D) or (1, W, H, D).
        filter_otsu: If True, only average values below Otsu threshold.

    Returns:
        Mean value of border voxels.
    """
    if tensor.dim() == 4:
        tensor = tensor[0]
    borders = torch.cat(
        [
            tensor[0, :, :].ravel(),
            tensor[-1, :, :].ravel(),
            tensor[:, 0, :].ravel(),
            tensor[:, -1, :].ravel(),
            tensor[:, :, 0].ravel(),
            tensor[:, :, -1].ravel(),
        ]
    )
    borders = borders.float()

    if not filter_otsu:
        return borders.mean().item()

    threshold = _otsu_threshold(borders)
    values = borders[borders < threshold]
    if values.numel() > 0:
        return values.mean().item()
    return borders.mean().item()


def _otsu_threshold(values: torch.Tensor) -> float:
    """Compute Otsu's threshold for a 1D tensor of values.

    Args:
        values: 1D tensor of values.

    Returns:
        The optimal threshold value.
    """
    sorted_vals, _ = values.sort()
    n = sorted_vals.numel()
    if n == 0:
        return 0.0

    total_sum = sorted_vals.sum().item()

    best_threshold = sorted_vals[0].item()
    best_variance = 0.0

    sum_bg = 0.0
    count_bg = 0

    for i in range(n - 1):
        val = sorted_vals[i].item()
        count_bg += 1
        count_fg = n - count_bg
        sum_bg += val

        mean_bg = sum_bg / count_bg
        mean_fg = (total_sum - sum_bg) / count_fg

        weight_bg = count_bg / n
        weight_fg = count_fg / n

        between_variance = weight_bg * weight_fg * (mean_bg - mean_fg) ** 2

        if between_variance > best_variance:
            best_variance = between_variance
            best_threshold = val

    return best_threshold


class RandomAffine(RandomTransform, SpatialTransform):
    r"""Apply a random affine transformation and resample the image.

    Args:
        scales: Tuple $(a_1, b_1, a_2, b_2, a_3, b_3)$ defining the
            scaling ranges.
            The scaling values along each dimension are $(s_1, s_2, s_3)$,
            where $s_i \sim \mathcal{U}(a_i, b_i)$.
            If two values $(a, b)$ are provided,
            then $s_i \sim \mathcal{U}(a, b)$.
            If only one value $x$ is provided,
            then $s_i \sim \mathcal{U}(1 - x, 1 + x)$.
            If three values $(x_1, x_2, x_3)$ are provided,
            then $s_i \sim \mathcal{U}(1 - x_i, 1 + x_i)$.
            For example, using `scales=(0.5, 0.5)` will zoom out the image,
            making the objects inside look twice as small while preserving
            the physical size and position of the image bounds.
        degrees: Tuple $(a_1, b_1, a_2, b_2, a_3, b_3)$ defining the
            rotation ranges in degrees.
            Rotation angles around each axis are
            $(\theta_1, \theta_2, \theta_3)$,
            where $\theta_i \sim \mathcal{U}(a_i, b_i)$.
            If two values $(a, b)$ are provided,
            then $\theta_i \sim \mathcal{U}(a, b)$.
            If only one value $x$ is provided,
            then $\theta_i \sim \mathcal{U}(-x, x)$.
            If three values $(x_1, x_2, x_3)$ are provided,
            then $\theta_i \sim \mathcal{U}(-x_i, x_i)$.
        translation: Tuple $(a_1, b_1, a_2, b_2, a_3, b_3)$ defining the
            translation ranges in mm.
            Translation along each axis is $(t_1, t_2, t_3)$,
            where $t_i \sim \mathcal{U}(a_i, b_i)$.
            If two values $(a, b)$ are provided,
            then $t_i \sim \mathcal{U}(a, b)$.
            If only one value $x$ is provided,
            then $t_i \sim \mathcal{U}(-x, x)$.
            If three values $(x_1, x_2, x_3)$ are provided,
            then $t_i \sim \mathcal{U}(-x_i, x_i)$.
            For example, if the image is in RAS+ orientation (e.g., after
            applying [`ToCanonical`][torchio.transforms.preprocessing.ToCanonical])
            and the translation is $(10, 20, 30)$, the sample will move
            10 mm to the right, 20 mm to the front, and 30 mm upwards.
            If the image was in, e.g., PIR+ orientation, the sample will move
            10 mm to the back, 20 mm downwards, and 30 mm to the right.
        isotropic: If `True`, only one scaling factor will be sampled for all dimensions,
            i.e. $s_1 = s_2 = s_3$.
            If one value $x$ is provided in `scales`, the scaling factor along all
            dimensions will be $s \sim \mathcal{U}(1 - x, 1 + x)$.
            If two values provided $(a, b)$ in `scales`, the scaling factor along all
            dimensions will be $s \sim \mathcal{U}(a, b)$.
        center: If `'image'`, rotations and scaling will be performed around
            the image center. If `'origin'`, rotations and scaling will be
            performed around the origin in world coordinates.
        default_pad_value: As the image is rotated, some values near the
            borders will be undefined.
            If `'minimum'`, the fill value will be the image minimum.
            If `'mean'`, the fill value is the mean of the border values.
            If `'otsu'`, the fill value is the mean of the values at the
            border that lie under an
            [Otsu threshold ](https://ieeexplore.ieee.org/document/4310076).
            If it is a number, that value will be used.
            This parameter applies to intensity images only.
        default_pad_label: As the label map is rotated, some values near the
            borders will be undefined. This numeric value will be used to fill
            those undefined regions. This parameter applies to label maps only.
        image_interpolation: Interpolation mode for intensity images.
            Must be `'nearest'` or `'linear'`.
        label_interpolation: Interpolation mode for label maps.
            Must be `'nearest'` or `'linear'`.
        check_shape: If `True` an error will be raised if the images are in
            different physical spaces. If `False`, `center` should
            probably not be `'image'` but `'center'`.
        **kwargs: See [`Transform`][torchio.transforms.Transform] for additional
            keyword arguments.

    Examples:
        >>> import torchio as tio
        >>> image = tio.datasets.Colin27().t1
        >>> transform = tio.RandomAffine(
        ...     scales=(0.9, 1.2),
        ...     degrees=15,
        ... )
        >>> transformed = transform(image)

    """

    def __init__(
        self,
        scales: TypeOneToSixFloat = 0.1,
        degrees: TypeOneToSixFloat = 10,
        translation: TypeOneToSixFloat = 0,
        isotropic: bool = False,
        center: str = 'image',
        default_pad_value: str | float = 'minimum',
        default_pad_label: int | float = 0,
        image_interpolation: str = 'linear',
        label_interpolation: str = 'nearest',
        check_shape: bool = True,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.isotropic = isotropic
        _parse_scales_isotropic(scales, isotropic)
        self.scales = self.parse_params(scales, 1, 'scales', min_constraint=0)
        self.degrees = self.parse_params(degrees, 0, 'degrees')
        self.translation = self.parse_params(translation, 0, 'translation')
        if center not in ('image', 'origin'):
            message = f'Center argument must be "image" or "origin", not "{center}"'
            raise ValueError(message)
        self.center = center
        self.default_pad_value = _parse_default_value(default_pad_value)
        if not isinstance(default_pad_label, (int, float)):
            message = 'default_pad_label must be a number, '
            message += f'but it is "{default_pad_label}"'
            raise ValueError(message)
        self.default_pad_label = float(default_pad_label)
        self.image_interpolation = _parse_affine_interpolation(
            image_interpolation,
        )
        self.label_interpolation = _parse_affine_interpolation(
            label_interpolation,
        )
        self.check_shape = check_shape

    @staticmethod
    def get_params(
        scales: TypeSextetFloat,
        degrees: TypeSextetFloat,
        translation: TypeSextetFloat,
        isotropic: bool,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        scaling_params = torch.as_tensor(
            RandomTransform.sample_uniform_sextet(scales),
            dtype=torch.float64,
        )
        if isotropic:
            scaling_params.fill_(scaling_params[0])
        rotation_params = torch.as_tensor(
            RandomTransform.sample_uniform_sextet(degrees),
            dtype=torch.float64,
        )
        translation_params = torch.as_tensor(
            RandomTransform.sample_uniform_sextet(translation),
            dtype=torch.float64,
        )
        return scaling_params, rotation_params, translation_params

    def apply_transform(self, subject: Subject) -> Subject:
        scaling_params, rotation_params, translation_params = self.get_params(
            self.scales,
            self.degrees,
            self.translation,
            self.isotropic,
        )
        scaling_values = [float(value) for value in scaling_params.tolist()]
        rotation_values = [float(value) for value in rotation_params.tolist()]
        translation_values = [float(value) for value in translation_params.tolist()]
        transform = Affine(
            scales=(scaling_values[0], scaling_values[1], scaling_values[2]),
            degrees=(rotation_values[0], rotation_values[1], rotation_values[2]),
            translation=(
                translation_values[0],
                translation_values[1],
                translation_values[2],
            ),
            center=self.center,
            default_pad_value=self.default_pad_value,
            default_pad_label=self.default_pad_label,
            image_interpolation=self.image_interpolation,
            label_interpolation=self.label_interpolation,
            check_shape=self.check_shape,
            **self._get_base_args(),
        )
        transformed = transform(subject)
        assert isinstance(transformed, Subject)
        return transformed


class Affine(SpatialTransform):
    r"""Apply affine transformation.

    Args:
        scales: Tuple $(s_1, s_2, s_3)$ defining the
            scaling values along each dimension.
        degrees: Tuple $(\theta_1, \theta_2, \theta_3)$ defining the
            rotation around each axis.
        translation: Tuple $(t_1, t_2, t_3)$ defining the
            translation in mm along each axis.
        center: If `'image'`, rotations and scaling will be performed around
            the image center. If `'origin'`, rotations and scaling will be
            performed around the origin in world coordinates.
        default_pad_value: As the image is rotated, some values near the
            borders will be undefined.
            If `'minimum'`, the fill value will be the image minimum.
            If `'mean'`, the fill value is the mean of the border values.
            If `'otsu'`, the fill value is the mean of the values at the
            border that lie under an
            [Otsu threshold ](https://ieeexplore.ieee.org/document/4310076).
            If it is a number, that value will be used.
            This parameter applies to intensity images only.
        default_pad_label: As the label map is rotated, some values near the
            borders will be undefined. This numeric value will be used to fill
            those undefined regions. This parameter applies to label maps only.
        image_interpolation: Interpolation mode for intensity images.
            Must be `'nearest'` or `'linear'`.
        label_interpolation: Interpolation mode for label maps.
            Must be `'nearest'` or `'linear'`.
        check_shape: If `True` an error will be raised if the images are in
            different physical spaces. If `False`, `center` should
            probably not be `'image'` but `'center'`.
        **kwargs: See [`Transform`][torchio.transforms.Transform] for additional
            keyword arguments.
    """

    def __init__(
        self,
        scales: TypeTripletFloat,
        degrees: TypeTripletFloat,
        translation: TypeTripletFloat,
        center: str = 'image',
        default_pad_value: str | float = 'minimum',
        default_pad_label: int | float = 0,
        image_interpolation: str = 'linear',
        label_interpolation: str = 'nearest',
        check_shape: bool = True,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.scales = self.parse_params(
            scales,
            None,
            'scales',
            make_ranges=False,
            min_constraint=0,
        )
        self.degrees = self.parse_params(
            degrees,
            None,
            'degrees',
            make_ranges=False,
        )
        self.translation = self.parse_params(
            translation,
            None,
            'translation',
            make_ranges=False,
        )
        if center not in ('image', 'origin'):
            message = f'Center argument must be "image" or "origin", not "{center}"'
            raise ValueError(message)
        self.center = center
        self.use_image_center = center == 'image'
        self.default_pad_value = _parse_default_value(default_pad_value)
        if not isinstance(default_pad_label, (int, float)):
            message = 'default_pad_label must be a number, '
            message += f'but it is "{default_pad_label}"'
            raise ValueError(message)
        self.default_pad_label = float(default_pad_label)
        self.image_interpolation = _parse_affine_interpolation(
            image_interpolation,
        )
        self.label_interpolation = _parse_affine_interpolation(
            label_interpolation,
        )
        self.invert_transform = False
        self.check_shape = check_shape
        self.args_names = [
            'scales',
            'degrees',
            'translation',
            'center',
            'default_pad_value',
            'default_pad_label',
            'image_interpolation',
            'label_interpolation',
            'check_shape',
        ]

    def get_affine_transform(self, image) -> np.ndarray:
        """Build the 4x4 forward physical transform for an image.

        Args:
            image: A TorchIO Image instance.

        Returns:
            A 4x4 numpy array representing the forward transform
            (input world -> output world).
        """
        scaling = np.asarray(self.scales).copy()
        rotation = np.asarray(self.degrees).copy()
        translation = np.asarray(self.translation).copy()

        if image.is_2d():
            scaling[2] = 1
            rotation[:-1] = 0

        if self.use_image_center:
            center_ras = np.array(image.get_center(lps=False))
        else:
            center_ras = None

        transform = _build_physical_transform(
            scaling,
            rotation,
            translation,
            center_ras,
        )

        if self.invert_transform:
            transform = np.linalg.inv(transform)

        return transform

    def get_default_pad_value(self, tensor: torch.Tensor) -> float:
        """Compute the padding value for out-of-bounds regions.

        Args:
            tensor: The image tensor (single channel, no batch dim).

        Returns:
            The fill value for out-of-bounds regions.
        """
        default_value: float
        if self.default_pad_value == 'minimum':
            default_value = tensor.min().item()
        elif self.default_pad_value == 'mean':
            default_value = _get_borders_mean(tensor, filter_otsu=False)
        elif self.default_pad_value == 'otsu':
            default_value = _get_borders_mean(tensor, filter_otsu=True)
        else:
            assert isinstance(self.default_pad_value, Number)
            default_value = float(self.default_pad_value)
        return default_value

    def apply_transform(self, subject: Subject) -> Subject:
        if self.check_shape:
            subject.check_consistent_spatial_shape()
        default_value: float
        for image in self.get_images(subject):
            forward_transform = self.get_affine_transform(image)
            theta = _physical_to_grid_theta(
                forward_transform,
                image.affine,
                image.spatial_shape,
            )
            transformed_tensors = []
            for tensor in image.data:
                if image[TYPE] != INTENSITY:
                    interpolation = self.label_interpolation
                    default_value = self.default_pad_label
                else:
                    interpolation = self.image_interpolation
                    default_value = self.get_default_pad_value(tensor)
                mode = _TORCH_INTERPOLATION_MODE[interpolation]
                transformed_tensor = _resample_tensor(
                    tensor.unsqueeze(0),
                    theta,
                    mode,
                    default_value,
                )
                transformed_tensors.append(transformed_tensor.squeeze(0))
            image.set_data(torch.stack(transformed_tensors))
        return subject


def _parse_affine_interpolation(interpolation: str) -> str:
    """Validate that interpolation is supported by the PyTorch backend.

    Args:
        interpolation: Interpolation mode string.

    Returns:
        The validated interpolation string (lowercased).

    Raises:
        TypeError: If interpolation is not a string.
        ValueError: If interpolation is not 'nearest' or 'linear'.
    """
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


def _parse_scales_isotropic(scales, isotropic):
    scales = to_tuple(scales)
    if isotropic and len(scales) in (3, 6):
        message = (
            'If "isotropic" is True, the value for "scales" must have'
            f' length 1 or 2, but "{scales}" was passed.'
            ' If you want to set isotropic scaling, use a single value or two values as a range'
            ' for the scaling factor. Refer to the documentation for more information.'
        )
        raise ValueError(message)


def _parse_default_value(value: str | float) -> str | float:
    if isinstance(value, Number) or value in ('minimum', 'otsu', 'mean'):
        return value
    message = (
        'Value for default_pad_value must be "minimum", "otsu", "mean" or a number'
    )
    raise ValueError(message)
