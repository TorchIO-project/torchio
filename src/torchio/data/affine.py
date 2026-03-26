"""Affine matrix class."""

from __future__ import annotations

import nibabel as nib
import numpy as np
import numpy.typing as npt

from ..types import TypeAffineMatrix
from ..types import TypeDirection
from ..types import TypeOrientationCodes
from ..types import TypeOrigin
from ..types import TypeSpacing
from ..types import TypeWorldPoints


class Affine:
    r"""$4 \times 4$ affine matrix mapping voxel indices to world coordinates.

    Thin wrapper around a numpy array providing named access to spacing,
    origin, direction, and orientation, plus composition and inversion.

    Supports the `__array__` protocol, so it can be used directly with
    numpy operations (e.g., `np.asarray(affine)`).

    Composition is supported via the `@` operator:

        combined = affine_a @ affine_b

    Args:
        matrix: $4 \times 4$ array-like. Defaults to the identity matrix.

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
        matrix: npt.ArrayLike | None = None,
    ) -> None:
        if matrix is None:
            self._matrix: npt.NDArray[np.float64] = np.eye(4)
        else:
            m = np.asarray(matrix, dtype=np.float64)
            if m.shape != (4, 4):
                msg = f"Affine must be 4x4, got {m.shape}"
                raise ValueError(msg)
            self._matrix = m.copy()

    # --- Construction helpers ---

    @classmethod
    def from_spacing(
        cls,
        spacing: TypeSpacing,
        *,
        origin: TypeOrigin = (0.0, 0.0, 0.0),
        direction: npt.ArrayLike | None = None,
    ) -> Affine:
        """Create an affine from spacing, origin, and direction.

        Args:
            spacing: Voxel size in mm along each axis.
            origin: World coordinates of the first voxel center.
            direction: 3x3 rotation/direction matrix. Identity if not given.
        """
        matrix = np.eye(4, dtype=np.float64)
        if direction is not None:
            matrix[:3, :3] = np.asarray(direction, dtype=np.float64)
        matrix[:3, :3] *= np.asarray(spacing, dtype=np.float64)
        matrix[:3, 3] = origin
        return cls(matrix)

    # --- Properties ---

    @property
    def spacing(self) -> TypeSpacing:
        """Voxel spacing in mm, derived from the rotation-zoom block."""
        rz = self._matrix[:3, :3]
        sp = np.sqrt(np.sum(rz**2, axis=0))
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
        sp = np.sqrt(np.sum(rz**2, axis=0))
        return rz / sp

    @property
    def orientation(self) -> TypeOrientationCodes:
        """Anatomical orientation codes (e.g., `('R', 'A', 'S')`)."""
        codes = nib.orientations.aff2axcodes(self._matrix)
        return (codes[0], codes[1], codes[2])

    # --- Methods ---

    def inverse(self) -> Affine:
        """Return the inverse affine."""
        return Affine(np.linalg.inv(self._matrix))

    def compose(self, other: Affine) -> Affine:
        """Return `self @ other` as a new `Affine`.

        Equivalent to using the `@` operator.
        """
        return Affine(self._matrix @ other._matrix)

    def apply(self, points: npt.ArrayLike) -> TypeWorldPoints:
        """Apply the affine to an (N, 3) array of points.

        Args:
            points: Array of shape (N, 3) in the source coordinate system.

        Returns:
            Transformed points, shape (N, 3).
        """
        pts = np.asarray(points, dtype=np.float64)
        ones = np.ones((pts.shape[0], 1), dtype=np.float64)
        homogeneous = np.hstack([pts, ones])
        transformed = (self._matrix @ homogeneous.T).T
        return transformed[:, :3]

    def numpy(self) -> TypeAffineMatrix:
        """Return the underlying 4x4 matrix as a numpy array."""
        return self._matrix

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
        if dtype is not None:
            return np.array(self._matrix, dtype=dtype, copy=copy)
        if copy:
            return self._matrix.copy()
        return self._matrix

    def __repr__(self) -> str:
        sp = ", ".join(f"{s:.2f}" for s in self.spacing)
        ori = "".join(self.orientation)
        o = ", ".join(f"{v:.2f}" for v in self.origin)
        return f"Affine(spacing=({sp}), origin=({o}), orientation={ori}+)"

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Affine):
            return NotImplemented
        return np.array_equal(self._matrix, other._matrix)

    def __copy__(self) -> Affine:
        return Affine(self._matrix)

    def __deepcopy__(self, memo: dict) -> Affine:
        new = Affine(self._matrix)
        memo[id(self)] = new
        return new
