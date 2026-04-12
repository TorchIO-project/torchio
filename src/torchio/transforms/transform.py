"""Transform base classes."""

from __future__ import annotations

import contextlib
import copy as _copy
import inspect
from dataclasses import dataclass
from dataclasses import field
from typing import Any
from typing import overload

import nibabel as nib
import numpy as np
import SimpleITK as sitk
import torch
from einops import rearrange
from torch import Tensor
from torch import nn

from ..data.batch import ImagesBatch
from ..data.batch import SubjectsBatch
from ..data.image import Image
from ..data.image import ScalarImage
from ..data.subject import Subject


@dataclass
class AppliedTransform:
    """Record of a transform application, stored in Subject history.

    Attributes:
        name: Class name of the transform.
        params: Sampled parameters (JSON-serializable).
    """

    name: str
    params: dict[str, Any] = field(default_factory=dict)


#: Registry mapping transform class names to classes, for inverse lookup.
_TRANSFORM_REGISTRY: dict[str, type[Transform]] = {}


class Transform(nn.Module):
    """Abstract class for all TorchIO transforms.

    When called, the input can be an instance of
    [`Subject`][torchio.Subject],
    [`Image`][torchio.Image],
    [`torch.Tensor`][torch.Tensor],
    [`numpy.ndarray`][numpy.ndarray],
    [`SimpleITK.Image`](https://simpleitk.org/doxygen/latest/html/classitk_1_1simple_1_1Image.html),
    [`nibabel.Nifti1Image`](https://nipy.org/nibabel/reference/nibabel.nifti1.html),
    [`dict`][dict] containing 4D tensors as values,
    [`ImagesBatch`][torchio.ImagesBatch], or
    [`SubjectsBatch`][torchio.SubjectsBatch].
    The output type always matches the input type.

    All subclasses must override
    [`apply_transform()`][torchio.Transform.apply_transform],
    which receives a [`SubjectsBatch`][torchio.SubjectsBatch] and
    returns the transformed batch.

    Args:
        p: Probability that this transform will be applied.
        copy: Make a deep copy of the input before applying the
            transform. When transforms are composed with
            [`Compose`][torchio.Compose], the outer ``Compose``
            copies once and sets ``copy=False`` on inner transforms
            to avoid redundant copies.
        include: Sequence of strings with the names of the only images
            to which the transform will be applied.
        exclude: Sequence of strings with the names of the images to
            which the transform will *not* be applied.
    """

    def __init__(
        self,
        *,
        p: float = 1.0,
        copy: bool = True,
        include: list[str] | None = None,
        exclude: list[str] | None = None,
    ) -> None:
        super().__init__()
        if not 0 <= p <= 1:
            msg = f"Probability must be in [0, 1], got {p}"
            raise ValueError(msg)
        self.p = p
        self.copy = copy
        self.include = include
        self.exclude = exclude

    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)
        _TRANSFORM_REGISTRY[cls.__name__] = cls

    def __repr__(self) -> str:
        """Show only non-default fields for a compact repr."""
        from .parameter_range import ParameterRange

        parts = []
        for name, default in _collect_init_params(type(self)).items():
            value = getattr(self, name, default)
            if isinstance(value, ParameterRange):
                if value._original == default:
                    continue
            elif value == default:
                continue
            parts.append(f"{name}={value!r}")
        return f"{type(self).__name__}({', '.join(parts)})"

    def __add__(self, other: object) -> Transform:
        """Compose two transforms: ``t1 + t2`` → ``Compose([t1, t2])``."""
        if not isinstance(other, Transform):
            return NotImplemented
        from .compose import Compose

        left = self.transforms if isinstance(self, Compose) else [self]
        right = other.transforms if isinstance(other, Compose) else [other]
        return Compose([*left, *right])

    def __or__(self, other: object) -> Transform:
        """Random choice: ``t1 | t2`` → ``OneOf([t1, t2])``."""
        if not isinstance(other, Transform):
            return NotImplemented
        from .compose import OneOf

        left = self.transforms if isinstance(self, OneOf) else [self]
        right = other.transforms if isinstance(other, OneOf) else [other]
        return OneOf([*left, *right])

    @overload
    def forward(self, data: Subject) -> Subject: ...
    @overload
    def forward(self, data: Image) -> Image: ...
    @overload
    def forward(self, data: Tensor) -> Tensor: ...
    @overload
    def forward(self, data: np.ndarray) -> np.ndarray: ...
    @overload
    def forward(self, data: sitk.Image) -> sitk.Image: ...
    @overload
    def forward(self, data: nib.Nifti1Image) -> nib.Nifti1Image: ...
    @overload
    def forward(self, data: dict) -> dict: ...
    @overload
    def forward(self, data: ImagesBatch) -> ImagesBatch: ...
    @overload
    def forward(self, data: SubjectsBatch) -> SubjectsBatch: ...

    def forward(self, data: Any) -> Any:
        """Apply the transform.

        The output type always matches the input type.

        Args:
            data: Input data to transform.
        """
        if self.copy:
            data = _copy.deepcopy(data)
        batch, unwrap = self._wrap(data)
        if torch.rand(1).item() > self.p:
            return unwrap(batch)
        params = self.make_params(batch)
        batch = self.apply_transform(batch, params)
        # Record history on the batch
        trace = AppliedTransform(name=type(self).__name__, params=params)
        if not hasattr(batch, "applied_transforms"):
            batch.applied_transforms = []
        batch.applied_transforms.append(trace)
        result = unwrap(batch)
        # Propagate history to outputs that can carry it
        if (
            hasattr(batch, "applied_transforms")
            and not isinstance(result, (SubjectsBatch, Tensor, np.ndarray))
            and not isinstance(result, dict)
        ):
            with contextlib.suppress(AttributeError):
                result.applied_transforms = list(batch.applied_transforms)
        return result

    def make_params(self, batch: SubjectsBatch) -> dict[str, Any]:
        """Sample random parameters for this transform.

        Override in subclasses that have random behavior.

        Args:
            batch: A ``SubjectsBatch``.

        Returns:
            Dict of sampled parameters.
        """
        return {}

    def apply_transform(
        self,
        batch: SubjectsBatch,
        params: dict[str, Any],
    ) -> SubjectsBatch:
        """Apply the transform with the given parameters.

        Must be overridden by subclasses. Receives a ``SubjectsBatch``
        whose ``ImagesBatch`` entries contain 5D tensors
        ``(B, C, I, J, K)``. Use negative indexing (``-3``, ``-2``,
        ``-1``) for spatial dims.

        Args:
            batch: A ``SubjectsBatch`` to transform.
            params: Parameters from ``make_params``.

        Returns:
            Transformed ``SubjectsBatch``.
        """
        raise NotImplementedError

    @property
    def invertible(self) -> bool:
        """Whether this transform can be inverted."""
        return False

    def inverse(self, params: dict[str, Any]) -> Transform:
        """Return a transform that undoes this one.

        Override in invertible subclasses. The returned transform,
        when applied, reverses the effect of the forward pass with
        the given parameters.

        Args:
            params: The parameters recorded in the forward pass.

        Returns:
            A new ``Transform`` instance that inverts this one.
        """
        msg = f"{type(self).__name__} is not invertible"
        raise NotImplementedError(msg)

    def _get_images(self, batch: SubjectsBatch) -> dict[str, ImagesBatch]:
        """Get image batches filtered by include/exclude."""
        images = batch.images
        if self.include is not None:
            images = {k: v for k, v in images.items() if k in self.include}
        if self.exclude is not None:
            images = {k: v for k, v in images.items() if k not in self.exclude}
        return images

    def to_hydra(self) -> dict[str, Any]:
        """Export as a Hydra-compatible config dict.

        Returns a dict with ``_target_`` set to the fully qualified
        class name and only non-default field values included.

        Returns:
            Dict suitable for ``hydra.utils.instantiate()``.
        """
        from .parameter_range import ParameterRange

        cls = type(self)
        target = f"torchio.{cls.__qualname__}"
        cfg: dict[str, Any] = {"_target_": target}

        for name, default in _collect_init_params(cls).items():
            value = getattr(self, name, default)
            if isinstance(value, ParameterRange):
                if value._original == default:
                    continue
                value = _hydra_value(value._original)
            elif value == default:
                continue
            else:
                value = _hydra_value(value)
            cfg[name] = value
        return cfg

    @staticmethod
    def _wrap(
        data: Any,
    ) -> tuple[Any, Any]:
        """Wrap any input into a SubjectsBatch; return (batch, unwrap_fn)."""
        from ..data.batch import ImagesBatch
        from ..data.batch import SubjectsBatch

        match data:
            case SubjectsBatch():
                return data, _unwrap_subjects_batch
            case ImagesBatch():
                sb = SubjectsBatch({"tio_default_image": data})
                return sb, _unwrap_images_batch
            case Subject():
                sb = SubjectsBatch.from_subjects([data])
                return sb, _unwrap_subject
            case dict():
                return _wrap_dict(data)
            case _:
                return _wrap_scalar_input(data)


def _wrap_single_image(img: Image, unwrap_fn: Any) -> tuple[Any, Any]:
    """Wrap a single Image into a SubjectsBatch."""
    from ..data.batch import SubjectsBatch

    sub = Subject(tio_default_image=img)
    sb = SubjectsBatch.from_subjects([sub])
    return sb, unwrap_fn


def _wrap_scalar_input(data: Any) -> tuple[Any, Any]:
    """Wrap a scalar input (Tensor, ndarray, SimpleITK, NIfTI) into a batch."""
    match data:
        case Image():
            return _wrap_single_image(data, _unwrap_image)
        case Tensor():
            return _wrap_single_image(ScalarImage(data), _unwrap_tensor)
        case np.ndarray():
            tensor = torch.as_tensor(data.copy(), dtype=torch.float32)
            if tensor.ndim == 3:
                tensor = rearrange(tensor, "i j k -> 1 i j k")
            return _wrap_single_image(
                ScalarImage(tensor),
                _unwrap_ndarray,
            )
        case sitk.Image():
            return _wrap_single_image(ScalarImage(data), _unwrap_sitk)
        case nib.Nifti1Image():
            return _wrap_single_image(ScalarImage(data), _unwrap_nifti)
        case _:
            msg = (
                "Expected Subject, Image, Tensor, ndarray, dict,"
                f" SimpleITK Image, NIfTI, ImagesBatch, or SubjectsBatch,"
                f" got {type(data).__name__}"
            )
            raise TypeError(msg)


def _wrap_dict(data: dict) -> tuple[Any, Any]:
    """Wrap a MONAI-style dict into a SubjectsBatch."""
    from ..data.batch import SubjectsBatch

    kwargs: dict[str, Any] = {}
    for k, v in data.items():
        match v:
            case Image():
                kwargs[k] = v
            case Tensor():
                kwargs[k] = ScalarImage(v)
            case _:
                kwargs[k] = v
    sub = Subject(**kwargs)
    keys: list[str] = [str(k) for k in data]
    sb = SubjectsBatch.from_subjects([sub])
    return sb, lambda b: _unwrap_dict(b, keys)


def _collect_init_params(cls: type) -> dict[str, Any]:
    """Collect all __init__ params with defaults from the full MRO.

    Walks from the leaf class up through parent classes, collecting
    named parameters (skipping ``self``, ``*args``, ``**kwargs``).
    Returns an ordered dict of ``{name: default}``.
    """
    params: dict[str, Any] = {}
    for klass in cls.__mro__:
        if klass is object or klass is nn.Module:
            break
        init = klass.__dict__.get("__init__")
        if init is None:
            continue
        sig = inspect.signature(init)
        for name, param in sig.parameters.items():
            if name == "self":
                continue
            if param.kind in (
                inspect.Parameter.VAR_POSITIONAL,
                inspect.Parameter.VAR_KEYWORD,
            ):
                continue
            if name not in params:
                params[name] = param.default
    return params


def _hydra_value(value: Any) -> Any:
    """Convert a value to a plain Python type for Hydra/YAML."""
    if isinstance(value, tuple):
        return list(value)
    if isinstance(value, Tensor):
        return value.tolist()
    if isinstance(value, np.ndarray):
        return value.tolist()
    return value


def _unwrap_subjects_batch(batch: SubjectsBatch) -> SubjectsBatch:
    return batch


def _unwrap_images_batch(batch: SubjectsBatch) -> ImagesBatch:
    return batch.images["tio_default_image"]


def _unwrap_subject(batch: SubjectsBatch) -> Subject:
    return batch.unbatch()[0]


def _unwrap_image(batch: SubjectsBatch) -> Image:
    sub = batch.unbatch()[0]
    return sub.tio_default_image


def _unwrap_tensor(batch: SubjectsBatch) -> Tensor:
    sub = batch.unbatch()[0]
    return sub.tio_default_image.data


def _unwrap_ndarray(batch: SubjectsBatch) -> np.ndarray:
    sub = batch.unbatch()[0]
    return sub.tio_default_image.data.cpu().numpy()


def _unwrap_sitk(batch: SubjectsBatch) -> sitk.Image:
    sub = batch.unbatch()[0]
    image = sub.tio_default_image
    data = image.data
    affine = image.affine
    array = data.cpu().numpy()
    array = array[0] if data.shape[0] == 1 else np.moveaxis(array, 0, -1)
    sitk_image = sitk.GetImageFromArray(array)
    if data.shape[0] > 1:
        sitk_image = sitk.GetImageFromArray(array, isVector=True)
    sitk_image.SetSpacing(affine.spacing)
    sitk_image.SetOrigin(affine.origin)
    sitk_image.SetDirection(rearrange(affine.direction, "i j -> (i j)").tolist())
    return sitk_image


def _unwrap_nifti(batch: SubjectsBatch) -> nib.Nifti1Image:
    sub = batch.unbatch()[0]
    image = sub.tio_default_image
    array = image.data.cpu().numpy()
    array = array[0] if array.shape[0] == 1 else np.moveaxis(array, 0, -1)
    return nib.Nifti1Image(array, image.affine.numpy())


def _unwrap_dict(batch: SubjectsBatch, keys: list[str]) -> dict[str, Any]:
    sub = batch.unbatch()[0]
    result: dict[str, Any] = {}
    for k in keys:
        entry = getattr(sub, k, None)
        if isinstance(entry, Image):
            result[k] = entry.data
        else:
            result[k] = entry
    return result


class SpatialTransform(Transform):
    """Base for transforms that modify spatial geometry.

    Spatial transforms apply to all images (ScalarImage and LabelMap),
    and also transform any Points and BoundingBoxes attached to the
    Subject.
    """


class IntensityTransform(Transform):
    """Base for transforms that modify voxel intensities.

    Intensity transforms apply only to ``ScalarImage`` instances,
    leaving ``LabelMap`` and annotations unchanged.
    """

    def _get_images(self, batch: SubjectsBatch) -> dict[str, ImagesBatch]:
        """Filter to ScalarImage batches only, then apply include/exclude."""
        images = {
            k: v for k, v in batch.images.items() if v._image_class is ScalarImage
        }
        if self.include is not None:
            images = {k: v for k, v in images.items() if k in self.include}
        if self.exclude is not None:
            images = {k: v for k, v in images.items() if k not in self.exclude}
        return images
