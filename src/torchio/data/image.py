"""Image classes for TorchIO."""

from __future__ import annotations

import copy
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
from .backends import ImageDataBackend
from .backends import NibabelBackend
from .backends import NumpyBackend
from .bboxes import BoundingBoxes
from .points import Points


def _is_nifti(path: Path) -> bool:
    name = path.name.lower()
    return name.endswith(".nii") or name.endswith(".nii.gz")


def _is_nifti_zarr(path: Path) -> bool:
    return str(path).endswith(".nii.zarr")


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


def _expand_ellipsis(
    items: tuple[int | slice | type(Ellipsis), ...],
    *,
    ndim: int,
) -> tuple[int | slice, ...]:
    """Replace a single `Ellipsis` with enough `slice(None)` to fill *ndim*."""
    n_ellipsis = sum(1 for s in items if s is Ellipsis)
    if n_ellipsis == 0:
        return items  # type: ignore[return-value]
    if n_ellipsis > 1:
        msg = "Only one ellipsis is allowed"
        raise IndexError(msg)
    idx = items.index(Ellipsis)
    n_explicit = len(items) - 1
    n_fill = max(ndim - n_explicit, 0)
    expanded = items[:idx] + (slice(None),) * n_fill + items[idx + 1 :]
    return expanded  # type: ignore[return-value]


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
        points: Named sets of [`Points`][torchio.Points] attached to this
            image. Keys are names, values are `Points` instances.
        bounding_boxes: Named sets of
            [`BoundingBoxes`][torchio.BoundingBoxes] attached to this
            image.
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
        points: dict[str, Points] | None = None,
        bounding_boxes: dict[str, BoundingBoxes] | None = None,
        metadata: dict[str, Any] | None = None,
    ):
        self._path: Path | None = Path(path)
        self._reader = reader or _default_reader
        self._metadata: dict[str, Any] = dict(metadata) if metadata else {}
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

    @classmethod
    def from_tensor(
        cls,
        tensor: TypeImageData | np.ndarray,
        *,
        affine: Affine | npt.ArrayLike | None = None,
        points: dict[str, Points] | None = None,
        bounding_boxes: dict[str, BoundingBoxes] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> Self:
        r"""Create an image from an in-memory tensor.

        Args:
            tensor: 4D [`torch.Tensor`][torch.Tensor] or NumPy array
                with shape $(C, I, J, K)$.
            affine: $4 \times 4$ affine matrix or
                [`Affine`][torchio.Affine] instance. Identity if `None`.
            points: Named sets of [`Points`][torchio.Points] attached to
                this image.
            bounding_boxes: Named sets of
                [`BoundingBoxes`][torchio.BoundingBoxes] attached to this
                image.
            metadata: Arbitrary metadata dict.
        """
        instance = object.__new__(cls)
        instance._path = None
        instance._reader = _default_reader
        instance._metadata = dict(metadata) if metadata else {}
        instance._data = Image._parse_tensor(tensor)
        parsed_affine = Image._parse_affine(affine)
        instance._affine = parsed_affine
        instance._backend = NumpyBackend(
            instance._data.numpy(),
            affine=parsed_affine.numpy(),
        )
        instance._points = Image._parse_annotations(points, "Points")
        instance._bounding_boxes = Image._parse_annotations(
            bounding_boxes,
            "BoundingBoxes",
        )
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

    @staticmethod
    def _parse_annotations(
        annotations: dict[str, Points | BoundingBoxes] | None,
        type_name: str,
    ) -> dict[str, Points | BoundingBoxes]:
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
        points_copy = {k: copy.deepcopy(v) for k, v in self._points.items()}
        bboxes_copy = {k: copy.deepcopy(v) for k, v in self._bounding_boxes.items()}
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
            if self._path is not None and self._reader is _default_reader:
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
            if self._reader is not _default_reader:
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
        """Number of bytes the tensor occupies in RAM."""
        c, si, sj, sk = self.shape
        element_size = self.data.element_size() if self.is_loaded else 4
        return c * si * sj * sk * element_size

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
            return
        # For NIfTI-Zarr or NIfTI with default reader, create backend first
        if self._reader is _default_reader and (
            _is_nifti_zarr(self._path) or _is_nifti(self._path)
        ):
            self._ensure_backend()
            if self._backend is not None:
                self._data = self._backend.to_tensor()
                if self._affine is None:
                    self._affine = Affine(self._backend.affine)
                return
        # Otherwise use the reader (custom reader or non-NIfTI formats)
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

        Preserves metadata, annotations, and affine. Uses the existing
        affine unless a new one is provided. Works correctly with custom
        subclasses.

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
        points_copy, bboxes_copy = self._deep_copy_annotations()
        return type(self).from_tensor(
            data,
            affine=new_affine,
            points=points_copy,
            bounding_boxes=bboxes_copy,
            metadata=dict(self._metadata),
        )

    def save(self, path: str | Path) -> None:
        """Save the image to a file.

        NIfTI-Zarr (`.nii.zarr`) files are written via `niizarr`
        (requires the `zarr` extra). All other formats are written
        with [SimpleITK](https://simpleitk.org/).

        Args:
            path: Output file path. The format is inferred from the extension.
        """
        path = Path(path)
        if _is_nifti_zarr(path):
            self._save_nii_zarr(path)
        else:
            self._save_sitk(path)

    def _save_sitk(self, path: Path) -> None:
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

    def _save_nii_zarr(self, path: Path) -> None:
        from ..imports import get_niizarr

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
        if _is_nifti_zarr(self._path):
            from .backends import ZarrBackend

            self._backend = ZarrBackend(self._path)
        elif _is_nifti(self._path):
            nii = nib.load(self._path)
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

    def __getitem__(self, item: int | slice | tuple[int | slice, ...]) -> Self:
        """Slice the image along channel and/or spatial dimensions.

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
        if isinstance(item, (int, slice)) or item is Ellipsis:
            items: tuple[int | slice | type(Ellipsis), ...] = (item,)
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
        affine_array = self.affine.numpy().copy()
        i_start, _, _ = si.indices(full_shape[1])
        j_start, _, _ = sj.indices(full_shape[2])
        k_start, _, _ = sk.indices(full_shape[3])
        start_voxel = np.array(
            [i_start, j_start, k_start],
            dtype=np.float64,
        )
        affine_array[:3, 3] += affine_array[:3, :3] @ start_voxel

        return self.new_like(data=cropped_data, affine=affine_array)

    def __repr__(self) -> str:
        cls_name = type(self).__name__
        parts: list[str] = []
        if self.is_loaded:
            sp = ", ".join(f"{s:.2f}" for s in self.spacing)
            orient = "".join(self.orientation)
            parts.append(f"shape: {self.shape}")
            parts.append(f"spacing: ({sp})")
            parts.append(f"orientation: {orient}+")
        elif self._path is not None:
            parts.append(f'path: "{self._path}"')
        if self._points:
            names = ", ".join(self._points)
            parts.append(f"points: {{{names}}}")
        if self._bounding_boxes:
            names = ", ".join(self._bounding_boxes)
            parts.append(f"bounding_boxes: {{{names}}}")
        return f"{cls_name}({'; '.join(parts)})"

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
                affine=affine_copy,
                points=points_copy,
                bounding_boxes=bboxes_copy,
                metadata=meta_copy,
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
                metadata=meta_copy,
            )
        else:
            new = object.__new__(type(self))
            new._path = None
            new._data = None
            new._backend = None
            new._affine = affine_copy
            new._reader = self._reader
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
