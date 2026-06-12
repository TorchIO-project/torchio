"""Lazy image data backends.

Backends provide a uniform interface for accessing image data without
requiring full materialization into memory. The `Image` class uses backends
internally. Users interact with images via `.data` (materialized tensor)
and `.dataobj` (lazy backend for advanced use like slicing).
"""

from __future__ import annotations

import types
from collections.abc import Callable
from collections.abc import Mapping
from dataclasses import dataclass
from dataclasses import field
from pathlib import Path
from typing import Any
from typing import Protocol
from typing import runtime_checkable

import nibabel as nib
import nibabel.spatialimages
import numpy as np
import torch
from einops import rearrange
from torch import Tensor

from ..types import SliceIndex
from ..types import TypeAffineMatrix
from ..types import TypeTensorShape


def _expand_ellipsis(
    items: tuple[int | slice | types.EllipsisType, ...],
    *,
    ndim: int,
) -> tuple[int | slice, ...]:
    """Replace a single `Ellipsis` with enough full slices to reach *ndim*."""
    n_ellipsis = sum(1 for s in items if s is Ellipsis)
    if n_ellipsis == 0:
        return tuple(s for s in items if s is not Ellipsis)
    if n_ellipsis > 1:
        msg = "Only one ellipsis is allowed"
        raise IndexError(msg)
    idx = items.index(Ellipsis)
    n_explicit = len(items) - 1
    n_fill = max(ndim - n_explicit, 0)
    expanded = items[:idx] + (slice(None),) * n_fill + items[idx + 1 :]
    return tuple(s for s in expanded if s is not Ellipsis)


def normalize_index(item: SliceIndex, *, ndim: int = 4) -> tuple[slice, ...]:
    """Normalize an index into a tuple of exactly *ndim* slices.

    Integer indices are converted to size-1 slices so that dimensions are
    never dropped, a single `Ellipsis` expands to full slices, and missing
    trailing indices are right-padded with full slices. This guarantees that
    direct backend slicing (`image.dataobj[...]`) follows the same shape rules
    as [`Image.__getitem__`][torchio.Image.__getitem__] and always returns
    `(C, I, J, K)` data.

    Args:
        item: Integer, slice, `Ellipsis`, or a tuple of those.
        ndim: Number of dimensions to produce (4 for `(C, I, J, K)`).

    Returns:
        A tuple of exactly *ndim* `slice` objects.

    Raises:
        IndexError: If more than *ndim* indices or several ellipses are given.
        TypeError: If an index element is not an `int`, `slice`, or `Ellipsis`.
    """
    match item:
        case int() | slice():
            items: tuple[int | slice | types.EllipsisType, ...] = (item,)
        case tuple():
            items = item
        case _ if item is Ellipsis:
            items = (item,)
        case _:
            msg = f"Index type {type(item).__name__} not understood"
            raise TypeError(msg)

    items = _expand_ellipsis(items, ndim=ndim)

    if len(items) > ndim:
        msg = (
            f"Too many indices: expected at most {ndim} (C, I, J, K), got {len(items)}"
        )
        raise IndexError(msg)

    parsed: list[slice] = []
    for s in items:
        match s:
            case int():
                # Keep the axis (size 1) instead of dropping it. ``slice(-1,
                # 0)`` would be empty, so the last element needs a special case.
                parsed.append(slice(s, None) if s == -1 else slice(s, s + 1))
            case slice():
                parsed.append(s)
            case _:
                msg = f"Index type {type(s).__name__} not understood"
                raise TypeError(msg)
    while len(parsed) < ndim:
        parsed.append(slice(None))
    return tuple(parsed)


@runtime_checkable
class ImageDataBackend(Protocol):
    """Protocol for lazy image data access.

    Implementations wrap different storage formats (in-memory tensors, NIfTI
    files via nibabel, NIfTI-Zarr via dask) behind a uniform, lazy I/O
    interface. This is an I/O adapter layer, not a lazy computation framework:
    it speeds up metadata reads and region slicing, but does not defer
    arithmetic or transforms.

    Contract:
        - `shape` is always 4D `(C, I, J, K)`, even for 3D NIfTI (channel
          dimension 1).
        - `affine` is a `float64` `torch.Tensor` of shape `(4, 4)`.
        - `dtype` reports the on-disk (or in-memory) `numpy` dtype.
        - `__getitem__` accepts the same indexing as
          [`Image.__getitem__`][torchio.Image.__getitem__], always returns a
          4D `torch.Tensor` in `(C, I, J, K)` layout, and never drops axes
          (integer indices keep a size-1 dimension).
        - `to_tensor` materializes the full volume as a `torch.Tensor`
          preserving the on-disk dtype where PyTorch supports it.
    """

    @property
    def shape(self) -> TypeTensorShape:
        """Shape as (C, I, J, K)."""
        ...

    @property
    def affine(self) -> TypeAffineMatrix:
        """$4 \\times 4$ affine matrix as a float64 tensor."""
        ...

    @property
    def dtype(self) -> np.dtype:
        """Data type of the image on disk."""
        ...

    def __getitem__(self, slices: SliceIndex) -> Tensor:
        """Slice the data, returning a 4D `(C, I, J, K)` tensor.

        Integer indices keep a size-1 dimension (axes are never dropped).
        Tensor-backed images preserve their device and dtype; lazy backends
        read only the requested region and convert it to a tensor.
        """
        ...

    def to_tensor(self) -> Tensor:
        """Materialize the full data as a tensor preserving the on-disk dtype."""
        ...


class TensorBackend:
    """Backend wrapping an in-memory PyTorch tensor.

    Used for images created from tensors or NumPy arrays (NumPy arrays are
    converted to tensors first).

    Args:
        data: 4D tensor with shape (C, I, J, K).
        affine: $4 \\times 4$ affine tensor. Identity if not given.
    """

    __slots__ = ("_affine", "_data")

    def __init__(
        self,
        data: Tensor,
        affine: TypeAffineMatrix | None = None,
    ) -> None:
        self._data = data
        if affine is not None:
            self._affine = affine
        else:
            self._affine = torch.eye(4, dtype=torch.float64)

    @property
    def shape(self) -> TypeTensorShape:
        s = self._data.shape
        return (int(s[0]), int(s[1]), int(s[2]), int(s[3]))

    @property
    def affine(self) -> TypeAffineMatrix:
        return self._affine

    @property
    def dtype(self) -> np.dtype:
        # Map torch dtype to numpy for protocol compatibility
        return torch.empty(0, dtype=self._data.dtype).numpy().dtype

    def __getitem__(self, slices: SliceIndex) -> Tensor:
        """Slice the tensor, preserving device, dtype, and 4D layout."""
        return self._data[normalize_index(slices)]

    def to_tensor(self) -> Tensor:
        return self._data.clone()


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
        affine: Optional $4 \\times 4$ affine that overrides the affine stored
            in the NIfTI header. Used when the user passes an explicit affine
            to `Image`, so that `image.affine` and `image.dataobj.affine` agree.
    """

    __slots__ = ("_affine_override", "_nii", "_shape")

    def __init__(
        self,
        nii: nib.spatialimages.SpatialImage,
        affine: TypeAffineMatrix | None = None,
    ) -> None:
        self._nii = nii
        self._affine_override = (
            torch.as_tensor(affine, dtype=torch.float64) if affine is not None else None
        )
        header_shape = nii.header.get_data_shape()
        ndim = len(header_shape)
        if ndim == 3:
            si, sj, sk = header_shape
            self._shape: TypeTensorShape = (1, int(si), int(sj), int(sk))
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
    def shape(self) -> TypeTensorShape:
        return self._shape

    @property
    def affine(self) -> TypeAffineMatrix:
        if self._affine_override is not None:
            return self._affine_override
        return torch.as_tensor(
            self._nii.header.get_best_affine(),
            dtype=torch.float64,
        )

    @property
    def dtype(self) -> np.dtype:
        return self._nii.header.get_data_dtype()

    def __getitem__(self, slices: SliceIndex) -> Tensor:
        """Slice in (C, I, J, K) space, returning a 4D tensor.

        The (C, I, J, K) index is translated to the on-disk layout, which is
        (I, J, K) for 3D, (I, J, K, C) for 4D, and (I, J, K, 1, C) for 5D
        vector NIfTI. Integer indices keep their axis (size 1), so the result
        is always 4D.
        """
        sc, si, sj, sk = normalize_index(slices)
        ndim_on_disk = len(self._nii.header.get_data_shape())
        if ndim_on_disk == 3:
            # On disk (I, J, K); channel axis is synthetic (size 1).
            data = np.asarray(self._nii.dataobj[si, sj, sk])
            array = rearrange(data, "i j k -> 1 i j k")[sc]
        elif ndim_on_disk == 4:
            # On disk (I, J, K, C).
            data = np.asarray(self._nii.dataobj[si, sj, sk, sc])
            array = rearrange(data, "i j k c -> c i j k")
        elif ndim_on_disk == 5:
            # 5D vector NIfTI written by SimpleITK: (I, J, K, 1, C).
            data = np.asarray(self._nii.dataobj[si, sj, sk, :, sc])
            array = rearrange(data, "i j k 1 c -> c i j k")
        else:
            msg = f"Expected 3D, 4D, or 5D NIfTI, got {ndim_on_disk}D"
            raise ValueError(msg)
        from .io import _numpy_to_tensor

        array = np.ascontiguousarray(array)
        if not array.flags.writeable:
            # Proxy reads (e.g. memory-mapped NIfTI) can be read-only, which
            # PyTorch does not support; copy so the resulting tensor is safe
            # to mutate.
            array = array.copy()
        return _numpy_to_tensor(array)

    def to_tensor(self) -> Tensor:
        """Materialize the full image preserving the on-disk dtype."""
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
        from .io import _numpy_to_tensor

        return _numpy_to_tensor(np.ascontiguousarray(data))


class ZarrBackend:
    """Backend wrapping a NIfTI-Zarr file for chunked lazy access.

    NIfTI-Zarr files are loaded via `niizarr.zarr2nii()`, which returns
    a nibabel image with a dask array as its `dataobj`. This backend
    delegates to `NibabelBackend` for the actual data access.

    Requires the `nifti-zarr` package.

    Args:
        path: Path to a `.nii.zarr` directory or a remote `.nii.zarr` URI.
        affine: Optional $4 \\times 4$ affine that overrides the affine stored
            in the NIfTI-Zarr metadata.
        **kwargs: Extra keyword arguments forwarded to `niizarr.zarr2nii()`.
    """

    __slots__ = ("_nibabel_backend",)

    def __init__(
        self,
        path: str | object,
        affine: TypeAffineMatrix | None = None,
        **kwargs: Any,
    ) -> None:
        from ..external.imports import get_niizarr

        niizarr = get_niizarr()
        nii = niizarr.zarr2nii(str(path), **kwargs)
        self._nibabel_backend = NibabelBackend(nii, affine=affine)

    @property
    def shape(self) -> TypeTensorShape:
        return self._nibabel_backend.shape

    @property
    def affine(self) -> TypeAffineMatrix:
        return self._nibabel_backend.affine

    @property
    def dtype(self) -> np.dtype:
        return self._nibabel_backend.dtype

    def __getitem__(self, slices: SliceIndex) -> Tensor:
        return self._nibabel_backend[slices]

    def to_tensor(self) -> Tensor:
        return self._nibabel_backend.to_tensor()


# ── Backend resolution and registration ──────────────────────────────


@dataclass(frozen=True)
class BackendRequest:
    """Description of an image source used to resolve a lazy backend.

    A request decouples backend selection from the `Image` class: resolvers
    and custom backends receive a `BackendRequest` instead of an `Image`.

    Attributes:
        path: Resolved filesystem (or fsspec) path to the image, if any.
        remote_zarr_uri: Remote NIfTI-Zarr URI, if the source is remote.
        zarr_store: An open Zarr store object, if the source is a store.
        affine: Optional $4 \\times 4$ affine override to apply to the backend.
        reader_kwargs: Extra keyword arguments forwarded to the loader.
        reader: The reader configured on the `Image`. Custom readers that
            implement [`LazyReader`][torchio.data.backends.LazyReader] can
            build a lazy backend instead of loading the whole volume.
    """

    path: Path | None = None
    remote_zarr_uri: str | None = None
    zarr_store: Any = None
    affine: TypeAffineMatrix | None = None
    reader_kwargs: Mapping[str, Any] = field(default_factory=dict)
    reader: Any = None


@runtime_checkable
class LazyReader(Protocol):
    """A custom reader that can build a lazy backend.

    Readers passed to `Image` are normally simple callables returning
    `(tensor, affine)`, which always load the whole volume. A reader that also
    implements `create_backend` opts in to lazy access: `Image.shape`,
    `affine`, `dtype`, and slicing then go through the returned backend without
    materializing the full tensor.
    """

    def create_backend(self, request: BackendRequest) -> ImageDataBackend:
        """Build a lazy backend for *request*."""
        ...


BackendMatcher = Callable[[BackendRequest], bool]
"""Predicate deciding whether a backend can handle a `BackendRequest`."""

BackendFactory = Callable[[BackendRequest], ImageDataBackend]
"""Callable that builds an `ImageDataBackend` from a `BackendRequest`."""


@dataclass(frozen=True)
class _BackendEntry:
    name: str
    matcher: BackendMatcher
    factory: BackendFactory


_BACKEND_REGISTRY: list[_BackendEntry] = []


def register_backend(
    name: str,
    matcher: BackendMatcher,
    factory: BackendFactory,
    *,
    prepend: bool = True,
) -> None:
    """Register a lazy image data backend.

    Registered backends are consulted by
    [`resolve_backend`][torchio.data.backends.resolve_backend] in order. This is
    the extension point for supporting new formats without editing the `Image`
    class.

    Args:
        name: Identifier for the backend, used to unregister it later.
            Registering a new backend with an existing name replaces it.
        matcher: Predicate returning `True` if this backend can handle the
            given [`BackendRequest`][torchio.data.backends.BackendRequest].
        factory: Callable that builds the backend from the request.
        prepend: If `True` (default), the backend is consulted before existing
            registrations, so custom backends take priority over the built-ins.
    """
    unregister_backend(name)
    entry = _BackendEntry(name=name, matcher=matcher, factory=factory)
    if prepend:
        _BACKEND_REGISTRY.insert(0, entry)
    else:
        _BACKEND_REGISTRY.append(entry)


def unregister_backend(name: str) -> None:
    """Remove a previously registered backend by name (no-op if absent).

    Args:
        name: The name passed to
            [`register_backend`][torchio.data.backends.register_backend].
    """
    _BACKEND_REGISTRY[:] = [e for e in _BACKEND_REGISTRY if e.name != name]


def resolve_backend(request: BackendRequest) -> ImageDataBackend | None:
    """Resolve a lazy backend for *request*.

    Args:
        request: The source description.

    Returns:
        The first matching backend, or `None` if no registered backend can
        handle the request (for example a non-NIfTI file path, where the
        caller falls back to a full read).
    """
    for entry in _BACKEND_REGISTRY:
        if entry.matcher(request):
            return entry.factory(request)
    return None


# -- Built-in backends ----------------------------------------------------


def _match_custom_reader(request: BackendRequest) -> bool:
    return request.reader is not None and isinstance(request.reader, LazyReader)


def _build_custom_reader(request: BackendRequest) -> ImageDataBackend:
    reader: LazyReader = request.reader
    return reader.create_backend(request)


def _match_remote_zarr(request: BackendRequest) -> bool:
    return request.remote_zarr_uri is not None


def _build_remote_zarr(request: BackendRequest) -> ImageDataBackend:
    assert request.remote_zarr_uri is not None
    return ZarrBackend(
        request.remote_zarr_uri,
        affine=request.affine,
        **dict(request.reader_kwargs),
    )


def _match_zarr_store(request: BackendRequest) -> bool:
    return request.zarr_store is not None


def _build_zarr_store(request: BackendRequest) -> ImageDataBackend:
    from ..external.imports import get_niizarr

    niizarr = get_niizarr()
    nii = niizarr.zarr2nii(request.zarr_store, **dict(request.reader_kwargs))
    return NibabelBackend(nii, affine=request.affine)


def _match_nifti_zarr_path(request: BackendRequest) -> bool:
    from .io import is_nifti_zarr

    return request.path is not None and is_nifti_zarr(request.path)


def _build_nifti_zarr_path(request: BackendRequest) -> ImageDataBackend:
    assert request.path is not None
    return ZarrBackend(
        request.path,
        affine=request.affine,
        **dict(request.reader_kwargs),
    )


def _match_nifti_path(request: BackendRequest) -> bool:
    from .io import is_nifti

    return request.path is not None and is_nifti(request.path)


def _build_nifti_path(request: BackendRequest) -> ImageDataBackend:
    assert request.path is not None
    nii = nib.load(request.path, **dict(request.reader_kwargs))
    assert isinstance(nii, nib.spatialimages.SpatialImage)
    return NibabelBackend(nii, affine=request.affine)


def _register_builtin_backends() -> None:
    """Register the backends shipped with TorchIO (NIfTI and NIfTI-Zarr).

    A custom-reader entry is registered first so that a
    [`LazyReader`][torchio.data.backends.LazyReader] takes priority over the
    format-based built-ins.
    """
    register_backend(
        "custom-reader", _match_custom_reader, _build_custom_reader, prepend=False
    )
    register_backend(
        "remote-nifti-zarr", _match_remote_zarr, _build_remote_zarr, prepend=False
    )
    register_backend("zarr-store", _match_zarr_store, _build_zarr_store, prepend=False)
    register_backend(
        "nifti-zarr", _match_nifti_zarr_path, _build_nifti_zarr_path, prepend=False
    )
    register_backend("nifti", _match_nifti_path, _build_nifti_path, prepend=False)


_register_builtin_backends()
