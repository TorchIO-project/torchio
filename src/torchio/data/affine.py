"""Affine matrix class."""

from __future__ import annotations

import contextlib
from typing import Any

import nibabel as nib
import numpy as np
import numpy.typing as npt
import torch
from torch import Tensor

from ..types import TypeDirection
from ..types import TypeOrientationCodes
from ..types import TypeOrigin
from ..types import TypeSpacing


class Affine:
    r"""$4 \times 4$ affine matrix mapping voxel indices to world coordinates.

    Stores the matrix as a ``torch.Tensor`` so it can live on the same
    device as the image data. Named properties expose spacing, origin,
    direction, and orientation. Composition uses the ``@`` operator.

    Args:
        matrix: $4 \times 4$ array-like, ``torch.Tensor``, or ``None``
            (identity). NumPy arrays are converted to tensors.

    Examples:
        >>> import torchio as tio
        >>> affine = tio.Affine()
        >>> affine.spacing
        (1.0, 1.0, 1.0)
        >>> affine.orientation
        ('R', 'A', 'S')
    """

    __slots__ = ("_matrix",)

    def __init__(
        self,
        matrix: Tensor | npt.ArrayLike | None = None,
    ) -> None:
        if matrix is None:
            self._matrix = torch.eye(4, dtype=torch.float64)
        elif isinstance(matrix, Tensor):
            if matrix.shape != (4, 4):
                msg = f"Affine must be 4x4, got {tuple(matrix.shape)}"
                raise ValueError(msg)
            self._matrix = matrix.to(torch.float64).clone()
        else:
            m = np.asarray(matrix, dtype=np.float64)
            if m.shape != (4, 4):
                msg = f"Affine must be 4x4, got {m.shape}"
                raise ValueError(msg)
            self._matrix = torch.as_tensor(m.copy(), dtype=torch.float64)

    # --- Construction helpers ---

    @classmethod
    def from_spacing(
        cls,
        spacing: TypeSpacing,
        *,
        origin: TypeOrigin = (0.0, 0.0, 0.0),
        direction: npt.ArrayLike | Tensor | None = None,
    ) -> Affine:
        """Create an affine from spacing, origin, and direction.

        Args:
            spacing: Voxel size in mm along each axis.
            origin: World coordinates of the first voxel center.
            direction: 3x3 rotation/direction matrix. Identity if not given.
        """
        matrix = torch.eye(4, dtype=torch.float64)
        if direction is not None:
            if isinstance(direction, Tensor):
                matrix[:3, :3] = direction.to(torch.float64)
            else:
                matrix[:3, :3] = torch.as_tensor(
                    np.asarray(direction, dtype=np.float64),
                )
        sp = torch.as_tensor(spacing, dtype=torch.float64)
        matrix[:3, :3] *= sp
        matrix[:3, 3] = torch.as_tensor(origin, dtype=torch.float64)
        return cls(matrix)

    # --- Properties ---

    @property
    def data(self) -> Tensor:
        """The underlying 4x4 tensor."""
        return self._matrix

    @property
    def device(self) -> torch.device:
        """Device the affine matrix resides on."""
        return self._matrix.device

    @property
    def spacing(self) -> TypeSpacing:
        """Voxel spacing in mm, derived from the rotation-zoom block."""
        rz = self._matrix[:3, :3]
        sp = torch.sqrt(torch.sum(rz**2, dim=0))
        return (float(sp[0]), float(sp[1]), float(sp[2]))

    @property
    def origin(self) -> TypeOrigin:
        """World coordinates of the first voxel center."""
        o = self._matrix[:3, 3]
        return (float(o[0]), float(o[1]), float(o[2]))

    @property
    def direction(self) -> TypeDirection:
        """3x3 direction (rotation) matrix, with spacing factored out."""
        rz = self._matrix[:3, :3]
        sp = torch.sqrt(torch.sum(rz**2, dim=0))
        return (rz / sp).cpu().numpy()

    @property
    def orientation(self) -> TypeOrientationCodes:
        """Anatomical orientation codes (e.g., `('R', 'A', 'S')`)."""
        codes = nib.orientations.aff2axcodes(self._matrix.cpu().numpy())
        return (codes[0], codes[1], codes[2])

    # --- Methods ---

    def to(self, *args: Any, **kwargs: Any) -> Affine:
        """Move the affine to a device.

        The affine always stays in float64 for precision. On devices
        that don't support float64 (e.g., MPS), it remains on CPU.

        Returns:
            ``self`` (modified in-place).
        """
        with contextlib.suppress(TypeError):
            # MPS doesn't support float64 — keep on CPU
            self._matrix = self._matrix.to(*args, **kwargs).to(torch.float64)
        return self

    def clone(self) -> Affine:
        """Return a deep copy."""
        return Affine(self._matrix.clone())

    def inverse(self) -> Affine:
        """Return the inverse affine."""
        return Affine(torch.linalg.inv(self._matrix))

    def compose(self, other: Affine) -> Affine:
        """Return `self @ other` as a new `Affine`.

        Equivalent to using the `@` operator.
        """
        return Affine(self._matrix @ other._matrix)

    def apply(self, points: Tensor | npt.ArrayLike) -> Tensor:
        """Apply the affine to an (N, 3) set of points.

        Args:
            points: Tensor or array of shape (N, 3).

        Returns:
            Transformed points as a tensor, shape (N, 3).
        """
        if not isinstance(points, Tensor):
            pts = torch.as_tensor(
                np.asarray(points, dtype=np.float64),
                dtype=torch.float64,
            )
        else:
            pts = points.to(torch.float64)
        pts = pts.to(self._matrix.device)
        ones = torch.ones(pts.shape[0], 1, dtype=torch.float64, device=pts.device)
        homogeneous = torch.cat([pts, ones], dim=1)
        transformed = (self._matrix @ homogeneous.T).T
        return transformed[:, :3]

    def numpy(self) -> npt.NDArray[np.float64]:
        """Return the underlying 4x4 matrix as a numpy array."""
        return self._matrix.cpu().numpy()

    # --- Dunder methods ---

    def __matmul__(self, other: object) -> Affine:
        """Compose two affines via the `@` operator."""
        if not isinstance(other, Affine):
            return NotImplemented
        return self.compose(other)

    def __array__(
        self,
        dtype: npt.DTypeLike | None = None,
        copy: bool | None = None,
    ) -> npt.NDArray[np.float64]:
        arr = self._matrix.cpu().numpy()
        if dtype is not None:
            return np.array(arr, dtype=dtype, copy=copy)
        if copy:
            return arr.copy()
        return arr

    def __repr__(self) -> str:
        sp = ", ".join(f"{s:.2f}" for s in self.spacing)
        ori = "".join(self.orientation)
        o = ", ".join(f"{v:.2f}" for v in self.origin)
        return f"Affine(spacing=({sp}), origin=({o}), orientation={ori}+)"

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Affine):
            return NotImplemented
        return torch.equal(self._matrix, other._matrix)

    def __copy__(self) -> Affine:
        return self.clone()

    def __deepcopy__(self, memo: dict) -> Affine:
        new = self.clone()
        memo[id(self)] = new
        return new
