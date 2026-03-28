"""I/O helpers for spatial transforms and matrices."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import SimpleITK as sitk
import torch
from einops import rearrange
from torch import Tensor

from .types import TypePath

# Matrices used to switch between LPS and RAS
FLIPXY_33 = np.diag([-1.0, -1.0, 1.0])
FLIPXY_44 = np.diag([-1.0, -1.0, 1.0, 1.0])


def read_matrix(path: TypePath) -> Tensor:
    """Read an affine transform from a file and return a 4x4 tensor in RAS.

    Supported formats:

    - ``.tfm``, ``.h5``: ITK transforms (read via SimpleITK)
    - ``.txt``, ``.trsf``: NiftyReg / blockmatching matrices

    Args:
        path: Path to the transform file.

    Returns:
        A ``(4, 4)`` float64 tensor representing the affine in RAS
        convention.
    """
    path = Path(path)
    suffix = path.suffix
    if suffix in (".tfm", ".h5"):
        return _read_itk_matrix(path)
    if suffix in (".txt", ".trsf"):
        return _read_niftyreg_matrix(path)
    msg = f'Unknown suffix for transform file: "{suffix}"'
    raise ValueError(msg)


def write_matrix(matrix: Tensor, path: TypePath) -> None:
    """Write a 4x4 affine matrix to a file.

    Args:
        matrix: A ``(4, 4)`` tensor in RAS convention.
        path: Destination path. Suffix determines format.
    """
    path = Path(path)
    suffix = path.suffix
    if suffix in (".tfm", ".h5"):
        _write_itk_matrix(matrix, path)
    elif suffix in (".txt", ".trsf"):
        _write_niftyreg_matrix(matrix, path)
    else:
        msg = f'Unknown suffix for transform file: "{suffix}"'
        raise ValueError(msg)


# --- ITK (LPS convention) ---------------------------------------------------


def _to_itk_convention(matrix: Tensor | np.ndarray) -> np.ndarray:
    """Convert a RAS affine to ITK's LPS convention."""
    if isinstance(matrix, Tensor):
        matrix = matrix.numpy()
    matrix = FLIPXY_44 @ matrix @ FLIPXY_44
    return np.linalg.inv(matrix)


def _from_itk_convention(matrix: np.ndarray) -> np.ndarray:
    """Convert an ITK LPS affine to RAS convention."""
    matrix = matrix @ FLIPXY_44
    matrix = FLIPXY_44 @ matrix
    return np.linalg.inv(matrix)


def _read_itk_matrix(path: TypePath) -> Tensor:
    """Read an affine transform in ITK's ``.tfm`` or ``.h5`` format."""
    transform = sitk.ReadTransform(str(path))
    parameters = transform.GetParameters()
    rotation_parameters = parameters[:9]
    rotation_matrix = rearrange(np.array(rotation_parameters), "(i j) -> i j", i=3)
    translation_parameters = parameters[9:]
    translation_vector = rearrange(np.array(translation_parameters), "i -> i 1")
    matrix = np.hstack([rotation_matrix, translation_vector])
    homogeneous_matrix_lps = np.vstack([matrix, [0, 0, 0, 1]])
    homogeneous_matrix_ras = _from_itk_convention(homogeneous_matrix_lps)
    return torch.as_tensor(homogeneous_matrix_ras)


def _write_itk_matrix(matrix: Tensor, path: TypePath) -> None:
    """Write a RAS affine as an ITK ``.tfm`` file."""
    itk_matrix = _to_itk_convention(matrix)
    rotation = itk_matrix[:3, :3].ravel().tolist()
    translation = itk_matrix[:3, 3].tolist()
    transform = sitk.AffineTransform(rotation, translation)
    transform.WriteTransform(str(path))


# --- NiftyReg / blockmatching ------------------------------------------------


def _read_niftyreg_matrix(path: TypePath) -> Tensor:
    """Read a NiftyReg matrix (reference → floating, inverted to RAS)."""
    matrix = np.loadtxt(path).astype(np.float64)
    inverted = np.linalg.inv(matrix)
    return torch.from_numpy(inverted)


def _write_niftyreg_matrix(matrix: Tensor, path: TypePath) -> None:
    """Write a RAS affine as a NiftyReg ``.txt`` file."""
    if isinstance(matrix, Tensor):
        matrix = matrix.numpy()
    inverted = np.linalg.inv(matrix)
    np.savetxt(path, inverted, fmt="%.8f")
