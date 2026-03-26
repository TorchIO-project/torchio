"""Image classes for TorchIO."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any
from typing import Self
from typing import cast

import nibabel as nib
import numpy as np
import numpy.typing as npt
import SimpleITK as sitk
import torch
from einops import rearrange
from torch import Tensor

from ..types import TypeImageData
from ..types import TypePath
from ..types import TypeSpatialShape
from ..types import TypeTensorShape
from .affine import Affine


def _is_nifti(path: Path) -> bool:
    name = path.name.lower()
    return name.endswith(".nii") or name.endswith(".nii.gz")


def _read_nibabel(path: Path) -> tuple[TypeImageData, np.ndarray]:
    """Read a NIfTI image using NiBabel."""
    img = cast(nib.Nifti1Image, nib.load(path))
    data = np.asarray(img.dataobj)
    affine = np.asarray(img.affine)
    if data.ndim == 3:
        data = rearrange(data, "i j k -> 1 i j k")
    elif data.ndim == 4:
        data = rearrange(data, "i j k c -> c i j k")
    elif data.ndim == 5 and data.shape[3] == 1:
        # 5D vector NIfTI written by SimpleITK: (I, J, K, 1, C)
        data = rearrange(data, "i j k 1 c -> c i j k")
    else:
        msg = f"Expected 3D or 4D data, got {data.ndim}D"
        raise ValueError(msg)
    tensor = torch.as_tensor(data.copy(), dtype=torch.float32)
    return tensor, affine


def _read_sitk(path: Path) -> tuple[TypeImageData, np.ndarray]:
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
    direction = np.array(sitk_image.GetDirection()).reshape(3, 3)
    affine = np.eye(4, dtype=np.float64)
    affine[:3, :3] = direction * spacing
    affine[:3, 3] = origin
    tensor = torch.as_tensor(data.copy(), dtype=torch.float32)
    return tensor, affine


def _default_reader(path: Path) -> tuple[TypeImageData, np.ndarray]:
    """Read an image, dispatching to NiBabel or SimpleITK by extension."""
    if _is_nifti(path):
        return _read_nibabel(path)
    return _read_sitk(path)


class Image:
    r"""Base image class.

    TorchIO images are
    [lazy loaders](https://en.wikipedia.org/wiki/Lazy_loading):
    data is only read from disk when first accessed.

    Use [`ScalarImage`][torchio.ScalarImage] for intensity data and
    [`LabelMap`][torchio.LabelMap] for segmentations.
    Transforms use `isinstance` checks to decide behavior (e.g., nearest-
    neighbor interpolation for [`LabelMap`][torchio.LabelMap]).

    Use `from_tensor` to create an image from an in-memory tensor.

    Args:
        path: Path to an image file. NIfTI files are read with
            [NiBabel](https://nipy.org/nibabel/); all other formats are read
            with [SimpleITK](https://simpleitk.org/).
        reader: Callable that takes a path and returns a tuple
            `(tensor, affine_array)`. Overrides the default reader.
        affine: $4 \times 4$ affine matrix or
            [`Affine`][torchio.Affine] instance. If given, overrides
            the affine read from the file.
        metadata: Arbitrary metadata dict.

    Examples:
        >>> import torchio as tio
        >>> image = tio.ScalarImage("t1.nii.gz")  # lazy, not loaded yet
        >>> image.data  # triggers load
        >>> image.spacing
        (1.0, 1.0, 1.0)
        >>> image = tio.ScalarImage.from_tensor(torch.randn(1, 256, 256, 176))
    """

    def __init__(
        self,
        path: TypePath,
        *,
        reader: Callable[[Path], tuple[TypeImageData, np.ndarray]] | None = None,
        affine: Affine | npt.ArrayLike | None = None,
        metadata: dict[str, Any] | None = None,
    ):
        self._path: Path | None = Path(path)
        self._reader = reader or _default_reader
        self._metadata: dict[str, Any] = dict(metadata) if metadata else {}
        self._data: Tensor | None = None
        self._affine: Affine | None = (
            self._parse_affine(affine) if affine is not None else None
        )

    @classmethod
    def from_tensor(
        cls,
        tensor: TypeImageData | np.ndarray,
        *,
        affine: Affine | npt.ArrayLike | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> Self:
        r"""Create an image from an in-memory tensor.

        Args:
            tensor: 4D [`torch.Tensor`][torch.Tensor] or NumPy array
                with shape $(C, I, J, K)$.
            affine: $4 \times 4$ affine matrix or
                [`Affine`][torchio.Affine] instance. Identity if `None`.
            metadata: Arbitrary metadata dict.
        """
        instance = object.__new__(cls)
        instance._path = None
        instance._reader = _default_reader
        instance._metadata = dict(metadata) if metadata else {}
        instance._data = Image._parse_tensor(tensor)
        instance._affine = Image._parse_affine(affine)
        return instance

    @staticmethod
    def _parse_tensor(tensor: Tensor | np.ndarray) -> Tensor:
        if isinstance(tensor, np.ndarray):
            tensor = torch.as_tensor(tensor.copy(), dtype=torch.float32)
        if tensor.ndim != 4:
            msg = f"Tensor must be 4D (C, I, J, K), got {tensor.ndim}D"
            raise ValueError(msg)
        return tensor

    @staticmethod
    def _parse_affine(affine: Affine | npt.ArrayLike | None) -> Affine:
        if affine is None:
            return Affine()
        if isinstance(affine, Affine):
            return affine
        return Affine(affine)

    # --- Properties ---

    @property
    def path(self) -> Path | None:
        """Path to the image file, if any."""
        return self._path

    @property
    def is_loaded(self) -> bool:
        """Whether the image data is loaded into memory."""
        return self._data is not None

    @property
    def data(self) -> TypeImageData:
        """Tensor data with shape (C, I, J, K). Triggers lazy load if needed."""
        if self._data is None:
            self.load()
        assert self._data is not None
        return self._data

    @property
    def affine(self) -> Affine:
        """4x4 affine matrix mapping voxel indices to world coordinates."""
        if self._affine is None:
            self.load()
        assert self._affine is not None
        return self._affine

    @property
    def metadata(self) -> dict[str, Any]:
        """Arbitrary metadata dict."""
        return self._metadata

    @property
    def shape(self) -> TypeTensorShape:
        """Tensor shape as (C, I, J, K)."""
        if self._data is not None:
            c, si, sj, sk = self._data.shape
            return (c, si, sj, sk)
        if self._path is not None:
            if self._reader is not _default_reader:
                self.load()
                return self.shape
            return self._read_shape_from_header()
        msg = "Cannot determine shape: no data or path"
        raise RuntimeError(msg)

    @property
    def spatial_shape(self) -> TypeSpatialShape:
        """Spatial dimensions as (I, J, K)."""
        return self.shape[1:]

    @property
    def num_channels(self) -> int:
        """Number of channels."""
        return self.shape[0]

    @property
    def spacing(self) -> tuple[float, float, float]:
        """Voxel spacing in mm, derived from the affine."""
        return self.affine.spacing

    @property
    def origin(self) -> tuple[float, float, float]:
        """Center of the first voxel in world coordinates."""
        return self.affine.origin

    @property
    def memory(self) -> int:
        """Number of bytes the tensor occupies in RAM."""
        c, si, sj, sk = self.shape
        element_size = self.data.element_size() if self.is_loaded else 4
        return c * si * sj * sk * element_size

    @property
    def orientation(self) -> tuple[str, str, str]:
        """Orientation codes from the affine."""
        return self.affine.orientation

    # --- Methods ---

    def load(self) -> None:
        """Load data from disk into memory."""
        if self._data is not None:
            return
        if self._path is None:
            msg = "Cannot load: no path set"
            raise RuntimeError(msg)
        tensor, affine_array = self._reader(self._path)
        self._data = tensor
        if self._affine is None:
            self._affine = Affine(affine_array)

    def set_data(self, tensor: TypeImageData | np.ndarray) -> None:
        """Replace the image data with a new tensor.

        Args:
            tensor: 4D tensor with shape (C, I, J, K).
        """
        self._data = self._parse_tensor(tensor)

    def new_like(
        self,
        *,
        data: TypeImageData,
        affine: Affine | npt.ArrayLike | None = None,
    ) -> Self:
        r"""Create a new image of the same class with new data.

        Preserves metadata and affine. Uses the existing affine
        unless a new one is provided. Works correctly with custom subclasses.

        Args:
            data: New 4D [`torch.Tensor`][torch.Tensor] with shape
                $(C, I, J, K)$.
            affine: New $4 \times 4$ affine. If `None`, uses `self.affine`.
        """
        new_affine = (
            self._parse_affine(affine)
            if affine is not None
            else Affine(self.affine.numpy().copy())
        )
        return type(self).from_tensor(
            data,
            affine=new_affine,
            metadata=dict(self._metadata),
        )

    def save(self, path: str | Path) -> None:
        """Save the image using SimpleITK.

        Supports any format SimpleITK can write (NIfTI, NRRD, MHA, etc.).

        Args:
            path: Output file path. The format is inferred from the extension.
        """
        path = Path(path)
        data = self.data.numpy()
        n_channels = data.shape[0]
        if n_channels == 1:
            array = rearrange(data, "1 i j k -> k j i")
            sitk_image = sitk.GetImageFromArray(array)
        else:
            array = rearrange(data, "c i j k -> k j i c")
            sitk_image = sitk.GetImageFromArray(array, isVector=True)
        sitk_image.SetSpacing(self.affine.spacing)
        sitk_image.SetOrigin(self.affine.origin)
        sitk_image.SetDirection(self.affine.direction.flatten().tolist())
        sitk.WriteImage(sitk_image, str(path))

    def _read_shape_from_header(self) -> TypeTensorShape:
        """Read shape from file metadata without loading data."""
        assert self._path is not None
        if _is_nifti(self._path):
            return self._read_shape_nibabel()
        return self._read_shape_sitk()

    def _read_shape_nibabel(self) -> TypeTensorShape:
        assert self._path is not None
        img = cast(nib.Nifti1Image, nib.load(self._path))
        header_shape = img.header.get_data_shape()
        if len(header_shape) == 3:
            si, sj, sk = header_shape
            return (1, int(si), int(sj), int(sk))
        elif len(header_shape) == 4:
            si, sj, sk, c = header_shape
            return (int(c), int(si), int(sj), int(sk))
        elif len(header_shape) == 5 and header_shape[3] == 1:
            # 5D vector NIfTI written by SimpleITK: (I, J, K, 1, C)
            si, sj, sk, _, c = header_shape
            return (int(c), int(si), int(sj), int(sk))
        msg = f"Expected 3D or 4D shape, got {len(header_shape)}D"
        raise ValueError(msg)

    def _read_shape_sitk(self) -> TypeTensorShape:
        assert self._path is not None
        reader = sitk.ImageFileReader()
        reader.SetFileName(str(self._path))
        reader.ReadImageInformation()
        size = reader.GetSize()
        n_components = reader.GetNumberOfComponents()
        ndim = reader.GetDimension()
        if ndim == 3:
            return (n_components, size[0], size[1], size[2])
        msg = f"Expected 3D image, got {ndim}D"
        raise ValueError(msg)

    def __repr__(self) -> str:
        cls_name = type(self).__name__
        if self.is_loaded:
            sp = ", ".join(f"{s:.2f}" for s in self.spacing)
            orient = "".join(self.orientation)
            return (
                f"{cls_name}("
                f"shape: {self.shape}; "
                f"spacing: ({sp}); "
                f"orientation: {orient}+)"
            )
        if self._path is not None:
            return f'{cls_name}(path: "{self._path}")'
        return f"{cls_name}()"

    def __copy__(self) -> Self:
        return self.new_like(data=self.data.clone())

    def __deepcopy__(self, memo: dict) -> Self:
        import copy as copy_mod

        affine_copy = (
            copy_mod.deepcopy(self._affine) if self._affine is not None else None
        )
        meta_copy = dict(self._metadata)

        if self._path is not None:
            new = type(self)(
                self._path,
                reader=self._reader,
                affine=affine_copy,
                metadata=meta_copy,
            )
            if self._data is not None:
                new._data = self._data.clone()
                new._affine = affine_copy
        elif self._data is not None:
            new = type(self).from_tensor(
                self._data.clone(),
                affine=affine_copy,
                metadata=meta_copy,
            )
        else:
            new = object.__new__(type(self))
            new._path = None
            new._data = None
            new._affine = affine_copy
            new._reader = self._reader
            new._metadata = meta_copy
        memo[id(self)] = new
        return new


class ScalarImage(Image):
    """Image with scalar (intensity) data.

    Transforms use `isinstance(image, ScalarImage)` to identify intensity
    images for operations like normalization or augmentation.

    Examples:
        >>> import torchio as tio
        >>> image = tio.ScalarImage("t1.nii.gz")
        >>> image = tio.ScalarImage.from_tensor(torch.randn(1, 256, 256, 176))
    """


class LabelMap(Image):
    """Image with label (segmentation) data.

    Transforms use `isinstance(image, LabelMap)` to apply nearest-neighbor
    interpolation during spatial transforms.

    Examples:
        >>> import torchio as tio
        >>> label = tio.LabelMap("seg.nii.gz")
        >>> label = tio.LabelMap.from_tensor(torch.randint(0, 5, (1, 256, 256, 176)))
    """
