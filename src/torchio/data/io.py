"""Image I/O: readers, source resolution, format detection."""

from __future__ import annotations

import tempfile
from io import IOBase
from pathlib import Path
from typing import Any
from typing import cast

import fsspec
import nibabel as nib
import numpy as np
import SimpleITK as sitk
import torch
from einops import rearrange

from ..types import TypeImageData

# NIfTI/TorchIO use RAS+; SimpleITK uses LPS+.  Multiplying the first
# two rows of the 4x4 affine by -1 converts between the two conventions.
_RAS_TO_LPS = np.diag([-1.0, -1.0, 1.0, 1.0])

#: Input types accepted by the Image constructor.
ImageSource = str | Path | IOBase | fsspec.core.OpenFile


# ── Source resolution ────────────────────────────────────────────────


def resolve_source(
    source: ImageSource,
    *,
    suffix: str | None = None,
) -> Path:
    """Resolve an ImageSource to a local Path.

    - Local paths and ``Path`` objects are returned directly.
    - Remote URIs (``http://``, ``s3://``, ``az://``, etc.) are
      fetched via fsspec and cached to a temp file.
    - ``fsspec.core.OpenFile`` objects are opened and written to a
      temp file.
    - File-like objects (``io.BytesIO``, open files) are written to a
      temp file. A *suffix* is required so readers can detect the
      format.
    """
    if isinstance(source, Path):
        return source
    if isinstance(source, str):
        if _is_remote(source):
            return _fetch_remote(source)
        return Path(source)
    if isinstance(source, fsspec.core.OpenFile):
        return _materialize_openfile(source)
    if isinstance(source, IOBase):
        if suffix is None:
            msg = (
                "A 'suffix' (e.g. '.nii.gz') is required when passing"
                " a file-like object so the reader can detect the format"
            )
            raise ValueError(msg)
        return _materialize_filelike(source, suffix=suffix)
    msg = (
        "Expected path, URL, fsspec.OpenFile, or file-like,"
        f" got {type(source).__name__}"
    )
    raise TypeError(msg)


# ── Format detection ─────────────────────────────────────────────────


def is_nifti(path: Path) -> bool:
    """Check if a path looks like a NIfTI file."""
    name = path.name.lower()
    return name.endswith(".nii") or name.endswith(".nii.gz")


def is_nifti_zarr(path: Path) -> bool:
    """Check if a path looks like a NIfTI-Zarr file."""
    return str(path).endswith(".nii.zarr")


def is_remote_nifti_zarr(uri: str) -> bool:
    """Check if a string is a remote NIfTI-Zarr URI."""
    clean = uri.split("?", maxsplit=1)[0].split("#", maxsplit=1)[0]
    clean = clean.rstrip("/").lower()
    return _is_remote(clean) and clean.endswith(".nii.zarr")


# ── Dtype helper ─────────────────────────────────────────────────────


# `torch.from_numpy` does not support these numpy dtypes directly, so we
# upcast to the next PyTorch-compatible signed/unsigned type while
# preserving integer semantics.
_NUMPY_DTYPE_PROMOTIONS: dict[np.dtype, np.dtype] = {
    np.dtype("bool"): np.dtype("uint8"),
    np.dtype("uint16"): np.dtype("int32"),
    np.dtype("uint32"): np.dtype("int64"),
    np.dtype("uint64"): np.dtype("int64"),
}


def _numpy_to_tensor(array: np.ndarray) -> torch.Tensor:
    """Convert a numpy array to a torch tensor preserving dtype where possible.

    Unsigned integer dtypes (``uint16``, ``uint32``, ``uint64``) and
    ``bool`` are not supported by ``torch.from_numpy``; they are upcast
    to the smallest signed/unsigned type that PyTorch supports while
    preserving the integer range. All other dtypes keep their native
    representation.
    """
    promotion = _NUMPY_DTYPE_PROMOTIONS.get(array.dtype)
    if promotion is not None:
        array = array.astype(promotion, copy=False)
    if not array.flags.writeable or not array.flags.c_contiguous:
        array = np.ascontiguousarray(array)
    return torch.from_numpy(array)


# ── Readers ──────────────────────────────────────────────────────────


def read_nibabel(path: Path, **kwargs: Any) -> tuple[TypeImageData, np.ndarray]:
    """Read a NIfTI image using NiBabel.

    Args:
        path: Path to the NIfTI file.
        **kwargs: Forwarded to ``nibabel.load()``.
    """
    img = cast(nib.Nifti1Image, nib.load(path, **kwargs))
    data = np.asarray(img.dataobj)
    affine = np.asarray(img.affine)
    if data.ndim == 3:
        data = rearrange(data, "i j k -> 1 i j k")
    elif data.ndim == 4:
        data = rearrange(data, "i j k c -> c i j k")
    elif data.ndim == 5 and data.shape[3] == 1:
        data = rearrange(data, "i j k 1 c -> c i j k")
    else:
        msg = f"Expected 3D or 4D data, got {data.ndim}D"
        raise ValueError(msg)
    tensor = _numpy_to_tensor(data.copy())
    return tensor, affine


def read_sitk(path: Path, **kwargs: Any) -> tuple[TypeImageData, np.ndarray]:
    """Read an image using SimpleITK (for non-NIfTI formats).

    Args:
        path: Path to the image file.
        **kwargs: Forwarded to ``SimpleITK.ReadImage()``.
    """
    sitk_image = sitk.ReadImage(str(path), **kwargs)
    data = sitk.GetArrayFromImage(sitk_image)
    n_components = sitk_image.GetNumberOfComponentsPerPixel()
    if data.ndim == 3 and n_components == 1:
        data = rearrange(data, "k j i -> 1 i j k")
    elif data.ndim == 4 and n_components > 1:
        data = rearrange(data, "k j i c -> c i j k")
    else:
        msg = f"Expected 3D data, got {data.ndim}D with {n_components} components"
        raise ValueError(msg)
    spacing = np.array(sitk_image.GetSpacing())
    origin = np.array(sitk_image.GetOrigin())
    direction = rearrange(np.array(sitk_image.GetDirection()), "(i j) -> i j", i=3)
    # SimpleITK returns LPS; convert to RAS for TorchIO's convention.
    lps_affine = np.eye(4, dtype=np.float64)
    lps_affine[:3, :3] = direction * spacing
    lps_affine[:3, 3] = origin
    affine = _RAS_TO_LPS @ lps_affine
    tensor = _numpy_to_tensor(data.copy())
    return tensor, affine


def default_reader(path: Path, **kwargs: Any) -> tuple[TypeImageData, np.ndarray]:
    """Read an image, dispatching to NiBabel, NIfTI-Zarr, or SimpleITK.

    Args:
        path: Path to the image file.
        **kwargs: Forwarded to the underlying reader.
    """
    if is_nifti_zarr(path):
        return read_nifti_zarr(path, **kwargs)
    if is_nifti(path):
        return read_nibabel(path, **kwargs)
    return read_sitk(path, **kwargs)


def read_nifti_zarr(
    path: Path,
    **kwargs: Any,
) -> tuple[TypeImageData, np.ndarray]:
    """Read a NIfTI-Zarr image using ``niizarr``.

    Requires the ``zarr`` extra: ``pip install torchio[zarr]``.

    Args:
        path: Path to a ``.nii.zarr`` directory.
        **kwargs: Forwarded to ``niizarr.zarr2nii()``.
    """
    from ..external.imports import get_niizarr

    niizarr = get_niizarr()
    nii = niizarr.zarr2nii(str(path), **kwargs)
    data = np.asarray(nii.dataobj)
    affine = np.asarray(nii.header.get_best_affine())
    if data.ndim == 3:
        data = rearrange(data, "i j k -> 1 i j k")
    elif data.ndim == 4:
        data = rearrange(data, "i j k c -> c i j k")
    else:
        msg = f"Expected 3D or 4D NIfTI-Zarr data, got {data.ndim}D"
        raise ValueError(msg)
    tensor = _numpy_to_tensor(data.copy())
    return tensor, affine


# ── Internal helpers ─────────────────────────────────────────────────


def _is_remote(path_str: str) -> bool:
    return "://" in path_str


def _write_to_tempfile(data: bytes, *, suffix: str) -> Path:
    fd, name = tempfile.mkstemp(suffix=suffix)
    try:
        with open(fd, "wb") as f:
            f.write(data)
    except BaseException:
        Path(name).unlink(missing_ok=True)
        raise
    return Path(name)


def _fetch_remote(uri: str) -> Path:
    with fsspec.open(uri, "rb") as remote:
        suffix = _guess_suffix(uri)
        return _write_to_tempfile(remote.read(), suffix=suffix)


def _materialize_openfile(of: fsspec.core.OpenFile) -> Path:
    with of as f:
        suffix = _guess_suffix(of.path)
        return _write_to_tempfile(f.read(), suffix=suffix)


def _materialize_filelike(f: IOBase, *, suffix: str) -> Path:
    return _write_to_tempfile(f.read(), suffix=suffix)


def _guess_suffix(path_str: str) -> str:
    clean = path_str.split("?")[0].split("#")[0]
    if ".nii.gz" in clean:
        return ".nii.gz"
    p = Path(clean)
    return p.suffix or ".nii.gz"
