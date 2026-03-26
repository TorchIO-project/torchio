"""Lazy image data backends.

Backends provide a uniform interface for accessing image data without
requiring full materialization into memory. The `Image` class uses backends
internally — users interact with images via `.data` (materialized tensor)
and `.dataobj` (lazy backend for advanced use like slicing).
"""

from __future__ import annotations

from typing import Protocol
from typing import runtime_checkable

import nibabel as nib
import numpy as np
import torch
from einops import rearrange
from torch import Tensor


@runtime_checkable
class ImageDataBackend(Protocol):
    """Protocol for lazy image data access.

    Implementations wrap different storage formats (in-memory arrays,
    NIfTI files via nibabel, NIfTI-Zarr via dask) behind a uniform
    interface.
    """

    @property
    def shape(self) -> tuple[int, ...]:
        """Shape as (C, I, J, K)."""
        ...

    @property
    def affine(self) -> np.ndarray:
        """$4 \\times 4$ affine matrix."""
        ...

    def __getitem__(self, slices: object) -> np.ndarray:
        """Slice the data, returning a numpy array."""
        ...

    def to_tensor(self) -> Tensor:
        """Materialize the full data as a float32 tensor."""
        ...


class NumpyBackend:
    """Backend wrapping an in-memory numpy array.

    Used for images created via `Image.from_tensor()` or after full
    materialization from another backend.

    Args:
        data: 4D array with shape (C, I, J, K).
        affine: $4 \\times 4$ affine matrix. Identity if not given.
    """

    __slots__ = ("_affine", "_data")

    def __init__(
        self,
        data: np.ndarray,
        affine: np.ndarray | None = None,
    ) -> None:
        self._data = np.asarray(data)
        self._affine = affine if affine is not None else np.eye(4)

    @property
    def shape(self) -> tuple[int, ...]:
        return tuple(self._data.shape)

    @property
    def affine(self) -> np.ndarray:
        return self._affine

    def __getitem__(self, slices: object) -> np.ndarray:
        return self._data[slices]

    def to_tensor(self) -> Tensor:
        return torch.from_numpy(self._data.copy())


class NibabelBackend:
    """Backend wrapping a nibabel image for lazy NIfTI access.

    Data is accessed through nibabel's `dataobj` proxy, which supports
    memory-mapped reads. Shape and affine are read from the header
    without loading data.

    This backend also works with NIfTI-Zarr files loaded via `niizarr`,
    since `zarr2nii` returns a standard `nibabel.Nifti1Image` whose
    `dataobj` is a dask array.

    Args:
        nii: A nibabel image (typically from `nib.load()` or `zarr2nii()`).
    """

    __slots__ = ("_nii", "_shape")

    def __init__(self, nii: nib.spatialimages.SpatialImage) -> None:
        self._nii = nii
        header_shape = nii.header.get_data_shape()
        ndim = len(header_shape)
        if ndim == 3:
            si, sj, sk = header_shape
            self._shape = (1, int(si), int(sj), int(sk))
        elif ndim == 4:
            si, sj, sk, c = header_shape
            self._shape = (int(c), int(si), int(sj), int(sk))
        elif ndim == 5 and header_shape[3] == 1:
            # 5D vector NIfTI written by SimpleITK: (I, J, K, 1, C)
            si, sj, sk, _, c = header_shape
            self._shape = (int(c), int(si), int(sj), int(sk))
        else:
            msg = f"Expected 3D or 4D NIfTI, got {ndim}D"
            raise ValueError(msg)

    @property
    def shape(self) -> tuple[int, ...]:
        return self._shape

    @property
    def affine(self) -> np.ndarray:
        return np.asarray(self._nii.header.get_best_affine())

    def __getitem__(self, slices: object) -> np.ndarray:
        """Slice in (C, I, J, K) space.

        Translates the (C, I, J, K) indexing to the on-disk layout
        (I, J, K) for 3D or (I, J, K, C) for 4D NIfTI files.
        """
        if not isinstance(slices, tuple):
            slices = (slices,)
        ndim_on_disk = len(self._nii.header.get_data_shape())
        if ndim_on_disk == 3:
            # (C, I, J, K) -> (I, J, K), then add channel dim back
            disk_slices = slices[1:] if len(slices) > 1 else slices
            data = np.asarray(self._nii.dataobj[disk_slices])
            return rearrange(data, "... -> 1 ...")
        # 4D: (C, I, J, K) -> (I, J, K, C)
        c_slice = slices[0] if len(slices) > 0 else slice(None)
        spatial_slices = slices[1:] if len(slices) > 1 else ()
        disk_slices = (*spatial_slices, c_slice)
        data = np.asarray(self._nii.dataobj[disk_slices])
        if data.ndim == 3:
            # Single channel selected
            return rearrange(data, "i j k -> 1 i j k")
        return rearrange(data, "i j k c -> c i j k")

    def to_tensor(self) -> Tensor:
        """Materialize the full image as a float32 tensor."""
        data = np.asarray(self._nii.dataobj)
        ndim = data.ndim
        if ndim == 3:
            data = rearrange(data, "i j k -> 1 i j k")
        elif ndim == 4:
            data = rearrange(data, "i j k c -> c i j k")
        elif ndim == 5 and data.shape[3] == 1:
            # 5D vector NIfTI written by SimpleITK: (I, J, K, 1, C)
            data = rearrange(data, "i j k 1 c -> c i j k")
        else:
            msg = f"Expected 3D or 4D data, got {ndim}D"
            raise ValueError(msg)
        return torch.as_tensor(data.copy(), dtype=torch.float32)


class ZarrBackend:
    """Backend wrapping a NIfTI-Zarr file for chunked lazy access.

    NIfTI-Zarr files are loaded via `niizarr.zarr2nii()`, which returns
    a nibabel image with a dask array as its `dataobj`. This backend
    delegates to `NibabelBackend` for the actual data access.

    Requires the `nifti-zarr` package.

    Args:
        path: Path to a `.nii.zarr` directory.
    """

    __slots__ = ("_nibabel_backend",)

    def __init__(self, path: str | object) -> None:
        from ..imports import get_niizarr

        niizarr = get_niizarr()
        nii = niizarr.zarr2nii(str(path))
        self._nibabel_backend = NibabelBackend(nii)

    @property
    def shape(self) -> tuple[int, ...]:
        return self._nibabel_backend.shape

    @property
    def affine(self) -> np.ndarray:
        return self._nibabel_backend.affine

    def __getitem__(self, slices: object) -> np.ndarray:
        return self._nibabel_backend[slices]

    def to_tensor(self) -> Tensor:
        return self._nibabel_backend.to_tensor()
