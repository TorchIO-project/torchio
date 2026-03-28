"""Image I/O: readers, source resolution, format detection."""

from __future__ import annotations

import tempfile
from io import IOBase
from pathlib import Path
from typing import cast

import fsspec
import nibabel as nib
import numpy as np
import SimpleITK as sitk
import torch
from einops import rearrange

from ..types import TypeImageData

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


# ── Readers ──────────────────────────────────────────────────────────


def read_nibabel(path: Path) -> tuple[TypeImageData, np.ndarray]:
    """Read a NIfTI image using NiBabel."""
    img = cast(nib.Nifti1Image, nib.load(path))
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
    tensor = torch.as_tensor(data.copy(), dtype=torch.float32)
    return tensor, affine


def read_sitk(path: Path) -> tuple[TypeImageData, np.ndarray]:
    """Read an image using SimpleITK (for non-NIfTI formats)."""
    sitk_image = sitk.ReadImage(str(path))
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
    affine = np.eye(4, dtype=np.float64)
    affine[:3, :3] = direction * spacing
    affine[:3, 3] = origin
    tensor = torch.as_tensor(data.copy(), dtype=torch.float32)
    return tensor, affine


def default_reader(path: Path) -> tuple[TypeImageData, np.ndarray]:
    """Read an image, dispatching to NiBabel or SimpleITK by extension."""
    if is_nifti(path):
        return read_nibabel(path)
    return read_sitk(path)


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
