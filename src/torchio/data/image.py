"""Image classes for TorchIO."""

from __future__ import annotations

import copy
import types
from collections.abc import Callable
from pathlib import Path
from typing import Any
from typing import Self
from typing import TypeVar

import nibabel as nib
import nibabel.spatialimages
import numpy as np
import numpy.typing as npt
import SimpleITK as sitk
import torch
from einops import rearrange
from torch import Tensor

from ..types import TypeImageData
from ..types import TypeSpatialShape
from ..types import TypeTensorShape
from .affine import Affine
from .backends import ImageDataBackend
from .backends import NibabelBackend
from .backends import NumpyBackend
from .bboxes import BoundingBoxes
from .invertible import Invertible
from .io import ImageSource
from .io import default_reader
from .io import is_nifti
from .io import is_nifti_zarr
from .io import resolve_source
from .points import Points

_AnnotationType = TypeVar("_AnnotationType", Points, BoundingBoxes)


def _backend_label(backend: object | None) -> str:
    """Short label for the backend type, used in ``__repr__``."""
    if backend is None:
        return "unknown"
    name = type(backend).__name__
    if "Nibabel" in name:
        return "NIfTI"
    if "Zarr" in name:
        return "NIfTI-Zarr"
    if "Numpy" in name:
        return "NumPy"
    return name


def _expand_ellipsis(
    items: tuple[int | slice | types.EllipsisType, ...],
    *,
    ndim: int,
) -> tuple[int | slice, ...]:
    """Replace a single `Ellipsis` with enough `slice(None)` to fill *ndim*."""
    n_ellipsis = sum(1 for s in items if s is Ellipsis)
    if n_ellipsis == 0:
        result: tuple[int | slice, ...] = tuple(s for s in items if s is not Ellipsis)
        return result
    if n_ellipsis > 1:
        msg = "Only one ellipsis is allowed"
        raise IndexError(msg)
    idx = items.index(Ellipsis)
    n_explicit = len(items) - 1
    n_fill = max(ndim - n_explicit, 0)
    expanded = items[:idx] + (slice(None),) * n_fill + items[idx + 1 :]
    result = tuple(s for s in expanded if s is not Ellipsis)
    return result


class Image(Invertible):
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
        path: Path to an image file, URL, ``fsspec.OpenFile``, or
            file-like object. NIfTI files are read with
            [NiBabel](https://nipy.org/nibabel/); all other formats
            with [SimpleITK](https://simpleitk.org/).
        reader: Callable that takes a path and returns a tuple
            `(tensor, affine_array)`. Overrides the default reader.
        reader_kwargs: Extra keyword arguments forwarded to the reader
            function. For the default reader these are passed to
            ``nibabel.load()`` or ``SimpleITK.ReadImage()``.
        affine: $4 \times 4$ affine matrix or
            [`Affine`][torchio.Affine] instance. If given, overrides
            the affine read from the file.
        points: Named sets of [`Points`][torchio.Points] attached to
            this image.
        bounding_boxes: Named sets of
            [`BoundingBoxes`][torchio.BoundingBoxes] attached to this
            image.
        **kwargs: Arbitrary metadata, accessible via attribute or
            dict-style lookup (e.g., ``protocol="MPRAGE"``).

    Examples:
        >>> import torchio as tio
        >>> image = tio.ScalarImage("t1.nii.gz")  # lazy, not loaded yet
        >>> image.data  # triggers load
        >>> image.spacing
        (1.0, 1.0, 1.0)
        >>> image = tio.ScalarImage.from_tensor(torch.randn(1, 256, 256, 176))
    """

    # Known __init__ kwargs — everything else goes to metadata.
    _INIT_KWARGS = frozenset(
        {
            "reader",
            "reader_kwargs",
            "affine",
            "channels_last",
            "suffix",
            "points",
            "bounding_boxes",
        }
    )

    def __init__(
        self,
        path: ImageSource,
        *,
        reader: Callable[[Path], tuple[TypeImageData, np.ndarray]] | None = None,
        reader_kwargs: dict[str, Any] | None = None,
        affine: Affine | npt.ArrayLike | None = None,
        channels_last: bool = False,
        suffix: str | None = None,
        points: dict[str, Points] | None = None,
        bounding_boxes: dict[str, BoundingBoxes] | None = None,
        **kwargs: Any,
    ):
        self._path: Path | None = resolve_source(path, suffix=suffix)
        self._reader = reader or default_reader
        self._reader_kwargs: dict[str, Any] = dict(reader_kwargs or {})
        self._channels_last = channels_last
        self._metadata: dict[str, Any] = dict(kwargs)
        self._data: Tensor | None = None
        self._backend: ImageDataBackend | None = None
        self._affine: Affine | None = (
            self._parse_affine(affine) if affine is not None else None
        )
        self._points = self._parse_annotations(points, "Points")
        self._bounding_boxes = self._parse_annotations(
            bounding_boxes,
            "BoundingBoxes",
        )
        self.applied_transforms: list[Any] = []

    @classmethod
    def from_tensor(
        cls,
        tensor: TypeImageData | np.ndarray,
        *,
        affine: Affine | npt.ArrayLike | None = None,
        channels_last: bool = False,
        points: dict[str, Points] | None = None,
        bounding_boxes: dict[str, BoundingBoxes] | None = None,
        **kwargs: Any,
    ) -> Self:
        r"""Create an image from an in-memory tensor.

        Args:
            tensor: 4D [`torch.Tensor`][torch.Tensor] or NumPy array
                with shape $(C, I, J, K)$, or $(I, J, K, C)$ if
                *channels_last* is ``True``.
            affine: $4 \times 4$ affine matrix or
                [`Affine`][torchio.Affine] instance. Identity if `None`.
            channels_last: If ``True``, the tensor is assumed to have
                shape $(I, J, K, C)$ and will be permuted to
                $(C, I, J, K)$.
            points: Named sets of [`Points`][torchio.Points] attached to
                this image.
            bounding_boxes: Named sets of
                [`BoundingBoxes`][torchio.BoundingBoxes] attached to this
                image.
            **kwargs: Arbitrary metadata (e.g., ``protocol="MPRAGE"``).
        """
        instance = object.__new__(cls)
        instance._path = None
        instance._reader = default_reader
        instance._reader_kwargs = {}
        instance._channels_last = False  # already permuted below
        instance._metadata = dict(kwargs)
        parsed = Image._parse_tensor(tensor)
        if channels_last:
            parsed = rearrange(parsed, "i j k c -> c i j k")
        instance._data = parsed
        parsed_affine = Image._parse_affine(affine)
        instance._affine = parsed_affine
        instance._backend = NumpyBackend(
            instance._data.detach().cpu().numpy(),
            affine=parsed_affine.numpy(),
        )
        instance._points = Image._parse_annotations(points, "Points")
        instance._bounding_boxes = Image._parse_annotations(
            bounding_boxes,
            "BoundingBoxes",
        )
        instance.applied_transforms = []
        return instance

    @classmethod
    def from_sitk(cls, sitk_image: sitk.Image, **kwargs: Any) -> Self:
        """Create an image from a SimpleITK Image.

        Preserves spacing, origin, and direction.

        Args:
            sitk_image: A SimpleITK Image.
            **kwargs: Forwarded to ``from_tensor`` (``points``,
                ``bounding_boxes``, ``metadata``).
        """
        data = sitk.GetArrayFromImage(sitk_image)
        n_components = sitk_image.GetNumberOfComponentsPerPixel()
        data = data[np.newaxis] if n_components == 1 else np.moveaxis(data, -1, 0)
        tensor = torch.as_tensor(data.copy(), dtype=torch.float32)
        spacing = np.array(sitk_image.GetSpacing())
        origin = np.array(sitk_image.GetOrigin())
        direction = rearrange(np.array(sitk_image.GetDirection()), "(i j) -> i j", i=3)
        affine_matrix = np.eye(4)
        affine_matrix[:3, :3] = direction * spacing
        affine_matrix[:3, 3] = origin
        return cls.from_tensor(tensor, affine=Affine(affine_matrix), **kwargs)

    @classmethod
    def from_nifti(cls, nifti_image: nib.Nifti1Image, **kwargs: Any) -> Self:
        """Create an image from a NiBabel Nifti1Image.

        Preserves the affine matrix.

        Args:
            nifti_image: A NiBabel Nifti1Image.
            **kwargs: Forwarded to ``from_tensor`` (``points``,
                ``bounding_boxes``, ``metadata``).
        """
        data = np.asarray(nifti_image.dataobj)
        if data.ndim == 3:
            data = data[np.newaxis]
        elif data.ndim == 4:
            data = np.moveaxis(data, -1, 0)
        tensor = torch.as_tensor(data.copy(), dtype=torch.float32)
        affine_matrix = np.asarray(nifti_image.affine)
        return cls.from_tensor(tensor, affine=Affine(affine_matrix), **kwargs)

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

    @staticmethod
    def _parse_annotations(
        annotations: dict[str, _AnnotationType] | None,
        type_name: str,
    ) -> dict[str, _AnnotationType]:
        """Validate and copy an annotation dict.

        Args:
            annotations: Mapping of names to annotation objects, or ``None``.
            type_name: Expected class name (``"Points"`` or
                ``"BoundingBoxes"``) used for validation and error messages.

        Returns:
            A shallow copy of the dict, or an empty dict if *annotations*
            is ``None``.

        Raises:
            TypeError: If any value is not an instance of the expected class.
        """
        if annotations is None:
            return {}
        expected_type = Points if type_name == "Points" else BoundingBoxes
        for key, value in annotations.items():
            if not isinstance(value, expected_type):
                msg = (
                    f"Expected {type_name} for key {key!r}, got {type(value).__name__}"
                )
                raise TypeError(msg)
        return dict(annotations)

    def _deep_copy_annotations(
        self,
    ) -> tuple[dict[str, Points], dict[str, BoundingBoxes]]:
        """Deep-copy both annotation dicts."""
        points_copy: dict[str, Points] = {
            k: copy.deepcopy(v) for k, v in self._points.items()
        }
        bboxes_copy: dict[str, BoundingBoxes] = {
            k: copy.deepcopy(v) for k, v in self._bounding_boxes.items()
        }
        return points_copy, bboxes_copy

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
            # Try backend first to avoid full data load
            if self._path is not None and self._reader is default_reader:
                self._ensure_backend()
                if self._backend is not None:
                    self._affine = Affine(self._backend.affine)
            if self._affine is None:
                self.load()
        assert self._affine is not None
        return self._affine

    @property
    def metadata(self) -> dict[str, Any]:
        """Arbitrary metadata dict."""
        return self._metadata

    @property
    def dataobj(self) -> ImageDataBackend:
        """Lazy data backend for advanced operations like slicing.

        Returns the underlying backend without materializing the full tensor.
        For NIfTI files this is a `NibabelBackend`; for NIfTI-Zarr files a
        `ZarrBackend`; for in-memory images a `NumpyBackend`.
        """
        if self._backend is None:
            self._ensure_backend()
        assert self._backend is not None
        return self._backend

    @property
    def shape(self) -> TypeTensorShape:
        """Tensor shape as (C, I, J, K)."""
        if self._data is not None:
            c, si, sj, sk = self._data.shape
            return (c, si, sj, sk)
        if self._backend is not None:
            s = self._backend.shape
            return (int(s[0]), int(s[1]), int(s[2]), int(s[3]))
        if self._path is not None:
            if self._reader is not default_reader:
                self.load()
                return self.shape
            # Try to create a lazy backend (NIfTI, Zarr)
            self._ensure_backend()
            if self._backend is not None:
                s = self._backend.shape
                return (int(s[0]), int(s[1]), int(s[2]), int(s[3]))
            # Non-NIfTI: read shape from header via SimpleITK
            return self._read_shape_sitk(self._path)
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
        """Number of bytes the tensor would occupy in RAM."""
        c, si, sj, sk = self.shape
        return c * si * sj * sk * self.dtype.itemsize

    @property
    def dtype(self) -> torch.dtype | np.dtype:
        """Data type of the image.

        Returns the PyTorch dtype if the image is loaded, otherwise
        reads the on-disk dtype from the header without loading data.
        """
        if self._data is not None:
            return self._data.dtype
        if self._backend is not None:
            return self._backend.dtype
        if self._path is not None:
            self._ensure_backend()
            if self._backend is not None:
                return self._backend.dtype
            # Non-NIfTI fallback: read via SimpleITK header
            return self._read_dtype_sitk(self._path)
        msg = "Cannot determine dtype: no data or path"
        raise RuntimeError(msg)

    @property
    def orientation(self) -> tuple[str, str, str]:
        """Orientation codes from the affine."""
        return self.affine.orientation

    @property
    def points(self) -> dict[str, Points]:
        """Named sets of points attached to this image."""
        return self._points

    @property
    def bounding_boxes(self) -> dict[str, BoundingBoxes]:
        """Named sets of bounding boxes attached to this image."""
        return self._bounding_boxes

    # --- Methods ---

    def load(self) -> None:
        """Load data from disk into memory."""
        if self._data is not None:
            return
        if self._path is None:
            msg = "Cannot load: no path set"
            raise RuntimeError(msg)
        # If we already have a lazy backend, materialize from it
        if self._backend is not None:
            self._data = self._backend.to_tensor()
            if self._affine is None:
                self._affine = Affine(self._backend.affine)
            self._apply_channels_last()
            return
        # For NIfTI-Zarr or NIfTI with default reader, create backend first
        if self._reader is default_reader and (
            is_nifti_zarr(self._path) or is_nifti(self._path)
        ):
            self._ensure_backend()
            if self._backend is not None:
                self._data = self._backend.to_tensor()
                if self._affine is None:
                    self._affine = Affine(self._backend.affine)
                self._apply_channels_last()
                return
        # Otherwise use the reader (custom reader or non-NIfTI formats)
        tensor, affine_array = self._reader(self._path, **self._reader_kwargs)
        self._data = tensor
        if self._affine is None:
            self._affine = Affine(affine_array)
        self._apply_channels_last()

    def _apply_channels_last(self) -> None:
        """Permute data from (I, J, K, C) to (C, I, J, K) if needed."""
        if self._channels_last and self._data is not None:
            self._data = rearrange(self._data, "i j k c -> c i j k")
            self._channels_last = False  # only do it once

    def set_data(self, tensor: TypeImageData | np.ndarray) -> None:
        """Replace the image data with a new tensor.

        Args:
            tensor: 4D tensor with shape (C, I, J, K).
        """
        self._data = self._parse_tensor(tensor)

    @property
    def device(self) -> torch.device:
        """Device the image data resides on."""
        return self.data.device

    def to(self, *args: Any, **kwargs: Any) -> Self:
        """Move image data and affine to a device and/or cast to a dtype.

        Accepts the same arguments as ``torch.Tensor.to()``.

        Returns:
            ``self`` (modified in-place).
        """
        self._data = self.data.to(*args, **kwargs)
        if self._affine is not None:
            self._affine.to(*args, **kwargs)
        return self

    def new_like(
        self,
        *,
        data: TypeImageData,
        affine: Affine | npt.ArrayLike | None = None,
    ) -> Self:
        r"""Create a new image of the same class with new data.

        Preserves metadata, annotations, and affine. Uses the existing
        affine unless a new one is provided. Works correctly with custom
        subclasses.

        Args:
            data: New 4D [`torch.Tensor`][torch.Tensor] with shape
                $(C, I, J, K)$.
            affine: New $4 \times 4$ affine. If `None`, uses `self.affine`.
        """
        new_affine = (
            self._parse_affine(affine) if affine is not None else self.affine.clone()
        )
        points_copy, bboxes_copy = self._deep_copy_annotations()
        return type(self).from_tensor(
            data,
            affine=new_affine,
            points=points_copy,
            bounding_boxes=bboxes_copy,
            **dict(self._metadata),
        )

    def save(
        self,
        path: str | Path,
        *,
        writer_kwargs: dict[str, Any] | None = None,
    ) -> None:
        """Save the image to a file.

        NIfTI-Zarr (`.nii.zarr`) files are written via `niizarr`
        (requires the `zarr` extra). All other formats are written
        with [SimpleITK](https://simpleitk.org/).

        Args:
            path: Output file path. The format is inferred from the
                extension.
            writer_kwargs: Extra keyword arguments forwarded to the
                writer. For SimpleITK formats these are passed to
                ``SimpleITK.WriteImage()``.
        """
        path = Path(path)
        if is_nifti_zarr(path):
            self._save_nii_zarr(path)
        else:
            self._save_sitk(path, **(writer_kwargs or {}))

    def _save_sitk(self, path: Path, **kwargs: Any) -> None:
        from .io import _RAS_TO_LPS

        data = self.data.numpy()
        n_channels = data.shape[0]
        if n_channels == 1:
            array = rearrange(data, "1 i j k -> k j i")
            sitk_image = sitk.GetImageFromArray(array)
        else:
            array = rearrange(data, "c i j k -> k j i c")
            sitk_image = sitk.GetImageFromArray(array, isVector=True)
        # Convert from RAS (TorchIO) to LPS (SimpleITK) before setting metadata.
        lps_affine = _RAS_TO_LPS @ self.affine.numpy()
        lps_spacing = np.sqrt(np.sum(lps_affine[:3, :3] ** 2, axis=0))
        lps_direction = lps_affine[:3, :3] / lps_spacing
        lps_origin = lps_affine[:3, 3]
        sitk_image.SetSpacing(lps_spacing.tolist())
        sitk_image.SetOrigin(lps_origin.tolist())
        sitk_image.SetDirection(lps_direction.ravel().tolist())
        sitk.WriteImage(sitk_image, str(path), **kwargs)

    def _save_nii_zarr(self, path: Path) -> None:
        from ..external.imports import get_niizarr

        niizarr = get_niizarr()
        data = self.data.numpy()
        n_channels = data.shape[0]
        if n_channels == 1:
            array = rearrange(data, "1 i j k -> i j k")
        else:
            array = rearrange(data, "c i j k -> i j k c")
        nii = nib.Nifti1Image(array, self.affine.numpy())
        niizarr.nii2zarr(nii, str(path))

    def _ensure_backend(self) -> None:
        """Create the lazy backend from path, without loading data.

        For NIfTI and NIfTI-Zarr files, creates a lazy backend that supports
        header-only reads. For other formats (NRRD, MHA, etc.), no lazy
        backend is available — callers should fall back to other methods.
        """
        if self._backend is not None:
            return
        if self._path is None:
            msg = "Cannot create backend: no path set"
            raise RuntimeError(msg)
        if is_nifti_zarr(self._path):
            from .backends import ZarrBackend

            self._backend = ZarrBackend(self._path)
        elif is_nifti(self._path):
            nii = nib.load(self._path, **self._reader_kwargs)
            assert isinstance(nii, nib.spatialimages.SpatialImage)
            self._backend = NibabelBackend(nii)

    @staticmethod
    def _read_shape_sitk(path: Path) -> TypeTensorShape:
        """Read shape from a SimpleITK-readable file without loading data."""
        reader = sitk.ImageFileReader()
        reader.SetFileName(str(path))
        reader.ReadImageInformation()
        size = reader.GetSize()
        n_components = reader.GetNumberOfComponents()
        ndim = reader.GetDimension()
        if ndim == 3:
            return (n_components, size[0], size[1], size[2])
        msg = f"Expected 3D image, got {ndim}D"
        raise ValueError(msg)

    @staticmethod
    def _read_dtype_sitk(path: Path) -> np.dtype:
        """Read dtype from a SimpleITK-readable file without loading data."""
        reader = sitk.ImageFileReader()
        reader.SetFileName(str(path))
        reader.ReadImageInformation()
        pixel_id = reader.GetPixelID()
        # Map SimpleITK pixel IDs to numpy dtypes
        sitk_to_numpy = {
            sitk.sitkUInt8: np.dtype("uint8"),
            sitk.sitkInt8: np.dtype("int8"),
            sitk.sitkUInt16: np.dtype("uint16"),
            sitk.sitkInt16: np.dtype("int16"),
            sitk.sitkUInt32: np.dtype("uint32"),
            sitk.sitkInt32: np.dtype("int32"),
            sitk.sitkUInt64: np.dtype("uint64"),
            sitk.sitkInt64: np.dtype("int64"),
            sitk.sitkFloat32: np.dtype("float32"),
            sitk.sitkFloat64: np.dtype("float64"),
            sitk.sitkVectorUInt8: np.dtype("uint8"),
            sitk.sitkVectorInt8: np.dtype("int8"),
            sitk.sitkVectorUInt16: np.dtype("uint16"),
            sitk.sitkVectorInt16: np.dtype("int16"),
            sitk.sitkVectorUInt32: np.dtype("uint32"),
            sitk.sitkVectorInt32: np.dtype("int32"),
            sitk.sitkVectorFloat32: np.dtype("float32"),
            sitk.sitkVectorFloat64: np.dtype("float64"),
        }
        return sitk_to_numpy.get(pixel_id, np.dtype("float32"))

    def __getitem__(
        self,
        item: str | int | slice | tuple[int | slice, ...],
    ) -> Any:
        """Slice the image or look up metadata by key.

        When *item* is a ``str``, the metadata value with that key is
        returned. Otherwise, the image is sliced along channel and/or
        spatial dimensions.

        Indexing follows the tensor layout `(C, I, J, K)`. Up to four
        indices may be provided; unspecified trailing dimensions keep
        their full extent. The affine origin is updated to reflect the
        spatial crop. Ellipsis (`...`) expands to fill unspecified
        dimensions with full slices. Negative indices and steps are
        supported.

        When the image has not been loaded yet, slicing reads only the
        requested region through the lazy backend — the full tensor is
        never materialized. For uncompressed NIfTI (`.nii`) this uses
        memory-mapping; for NIfTI-Zarr (`.nii.zarr`) chunked reads.
        Even for compressed NIfTI (`.nii.gz`), nibabel's proxy avoids
        materializing the full array, so slicing a small region from a
        large volume is much faster than loading everything first.

        Args:
            item: Integer, slice, or tuple of slices/ints/ellipsis for
                the C, I, J, K dimensions.

        Returns:
            A new image of the same class containing the sliced data.

        Examples:
            >>> image = tio.ScalarImage.from_tensor(torch.randn(3, 256, 256, 176))
            >>> image[0].shape              # first channel
            (1, 256, 256, 176)
            >>> image[:, 100:200].shape     # spatial range, all channels
            (3, 100, 256, 176)
            >>> image[..., 50:100].shape    # last spatial dim
            (3, 256, 256, 50)
            >>> image[1:3, 10:20, 10:20, 10:20].shape
            (2, 10, 10, 10)
        """
        # String key → metadata lookup
        if isinstance(item, str):
            if item in self._metadata:
                return self._metadata[item]
            msg = f"{type(self).__name__} has no metadata key {item!r}"
            raise KeyError(msg)

        if isinstance(item, (int, slice)) or item is Ellipsis:
            items: tuple[int | slice | types.EllipsisType, ...] = (item,)
        elif isinstance(item, tuple):
            items = item
        else:
            msg = f"Index type {type(item).__name__} not understood"
            raise TypeError(msg)

        # Expand ellipsis
        items = _expand_ellipsis(items, ndim=4)

        if len(items) > 4:
            msg = f"Too many indices: expected at most 4 (C, I, J, K), got {len(items)}"
            raise IndexError(msg)

        full_shape = self.shape  # (C, I, J, K)
        parsed: list[slice] = []
        for dim, s in enumerate(items):
            if isinstance(s, int):
                idx = s if s >= 0 else full_shape[dim] + s
                s = slice(idx, idx + 1)
            if not isinstance(s, slice):
                msg = f"Index type {type(s).__name__} not understood"
                raise TypeError(msg)
            parsed.append(s)

        # Pad with full slices for unspecified dimensions
        while len(parsed) < 4:
            parsed.append(slice(None))

        sc, si, sj, sk = parsed

        # Use backend for lazy slicing when possible (avoids full load)
        if self._data is not None:
            cropped_data = self._data[sc, si, sj, sk]
        else:
            self._ensure_backend()
            if self._backend is not None:
                array = self._backend[sc, si, sj, sk]
                cropped_data = torch.as_tensor(
                    np.asarray(array).copy(),
                    dtype=torch.float32,
                )
            else:
                cropped_data = self.data[sc, si, sj, sk]

        # Update affine origin: shift by the spatial start offset
        affine_matrix = self.affine.data.clone()
        i_start, _, _ = si.indices(full_shape[1])
        j_start, _, _ = sj.indices(full_shape[2])
        k_start, _, _ = sk.indices(full_shape[3])
        start_voxel = torch.tensor(
            [i_start, j_start, k_start],
            dtype=torch.float64,
        )
        affine_matrix[:3, 3] += affine_matrix[:3, :3] @ start_voxel

        return self.new_like(data=cropped_data, affine=Affine(affine_matrix))

    def __repr__(self) -> str:
        import humanize

        cls_name = type(self).__name__
        lines: list[str] = []
        try:
            sp = ", ".join(f"{s:.2f}" for s in self.spacing)
            ori = ", ".join(f"{o:.2f}" for o in self.origin)
            angles = ", ".join(f"{a:.1f}°" for a in self.affine.euler_angles)
            dt = str(self.dtype).replace("torch.", "")
            mem = humanize.naturalsize(self.memory, binary=True)

            # Path / loading status (after header read so backend is set)
            if self._path is not None:
                name = self._path.name
                if self.is_loaded:
                    lines.append(f"    path:        {name} (loaded)")
                else:
                    fmt = _backend_label(self._backend)
                    lines.append(f"    path:        {name} (lazy, {fmt})")
            else:
                lines.append("    path:        (in memory)")

            lines.append(f"    channels:    {self.num_channels}")
            lines.append(f"    spatial:     {self.spatial_shape}")
            lines.append(f"    spacing:     ({sp}) mm")
            lines.append(f"    origin:      ({ori}) mm")
            lines.append(f"    orientation: {''.join(self.orientation)}+")
            lines.append(f"    angles:      ({angles})")
            lines.append(f"    dtype:       {dt}")
            lines.append(f"    memory:      {mem}")
        except Exception:
            if self._path is not None:
                lines.append(f'    path: "{self._path}"')

        if self._points:
            names = ", ".join(self._points)
            lines.append(f"    points:      {{{names}}}")
        if self._bounding_boxes:
            names = ", ".join(self._bounding_boxes)
            lines.append(f"    bboxes:      {{{names}}}")

        body = "\n".join(lines)
        return f"{cls_name}(\n{body}\n)"

    def _repr_html_(self) -> str:
        """Rich HTML representation for Jupyter notebooks."""
        from ..repr_html import image_to_html

        return image_to_html(self)

    def plot(self, **kwargs: Any) -> Any:
        """Plot 3 orthogonal slices of the image.

        Requires the ``[plot]`` extras (``pip install torchio[plot]``).
        See [`plot_image`][torchio.visualization.plot_image] for the
        full list of keyword arguments.
        """
        from ..visualization import plot_image

        return plot_image(self, **kwargs)

    def __getattr__(self, name: str) -> Any:
        """Look up metadata by attribute name."""
        if name.startswith("_"):
            raise AttributeError(name)
        if name in self._metadata:
            return self._metadata[name]
        msg = f"{type(self).__name__} has no attribute {name!r}"
        raise AttributeError(msg)

    def __copy__(self) -> Self:
        return self.new_like(data=self.data.clone())

    def __deepcopy__(self, memo: dict) -> Self:
        affine_copy = copy.deepcopy(self._affine) if self._affine is not None else None
        meta_copy = dict(self._metadata)
        points_copy, bboxes_copy = self._deep_copy_annotations()

        if self._path is not None:
            new = type(self)(
                self._path,
                reader=self._reader,
                reader_kwargs=dict(self._reader_kwargs),
                affine=affine_copy,
                points=points_copy,
                bounding_boxes=bboxes_copy,
                **meta_copy,
            )
            if self._data is not None:
                new._data = self._data.clone()
                new._affine = affine_copy
            # Backend will be lazily recreated from path when needed
        elif self._data is not None:
            new = type(self).from_tensor(
                self._data.clone(),
                affine=affine_copy,
                points=points_copy,
                bounding_boxes=bboxes_copy,
                **meta_copy,
            )
        else:
            new = object.__new__(type(self))
            new._path = None
            new._data = None
            new._backend = None
            new._affine = affine_copy
            new._reader = self._reader
            new._reader_kwargs = dict(self._reader_kwargs)
            new._metadata = meta_copy
            new._points = points_copy
            new._bounding_boxes = bboxes_copy
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
