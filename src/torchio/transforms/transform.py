"""Transform base classes."""

from __future__ import annotations

import contextlib
import copy as _copy
import inspect
import warnings
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
from ..data.batch import _assign_histories
from ..data.batch import _get_element_histories
from ..data.image import Image
from ..data.image import ScalarImage
from ..data.subject import Subject


@dataclass
class AppliedTransform:
    """Record of a transform application, stored in Subject history.

    Attributes:
        name: Class name of the transform.
        params: Sampled parameters (JSON-serializable).
        include: Original include scope of the applied transform.
        exclude: Original exclude scope of the applied transform.
    """

    name: str
    params: dict[str, Any] = field(default_factory=dict)
    include: list[str] | None = None
    exclude: list[str] | None = None


#: Registry mapping transform class names to classes, for inverse lookup.
_TRANSFORM_REGISTRY: dict[str, type[Transform]] = {}


def _all_elements_gated_out(params: dict[str, Any]) -> bool:
    """Whether per-element gating masked out every batch element.

    Args:
        params: The parameter dict produced by `make_params`, possibly
            carrying a `_keep` mask added by `_tag_batched`.

    Returns:
        `True` only when a `_keep` mask is present and none of its
        elements are kept, i.e. the transform was an exact no-op.
    """
    keep = params.get("_keep")
    return keep is not None and not any(keep)


def _copy_optional_list(value: list[str] | None) -> list[str] | None:
    return None if value is None else list(value)


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
        p: Probability that this transform will be applied. When
            per-instance probability is active (see `per_instance`),
            this is instead the per-element probability and each batch
            element is gated independently.
        copy: Make a deep copy of the input before applying the
            transform. When transforms are composed with
            [`Compose`][torchio.Compose], the outer `Compose`
            copies once and sets `copy=False` on inner transforms
            to avoid redundant copies.
        per_instance: If `True` (default), transforms that support it
            sample independent parameters for each element of a batch
            (and gate each element independently with `p`). If
            `False`, a single parameter set is sampled and applied
            identically to every element, reproducing the legacy
            batch-shared behavior. Single-element inputs (including a
            single [`Subject`][torchio.Subject]) are unaffected by this
            flag.
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
        per_instance: bool = True,
        include: list[str] | None = None,
        exclude: list[str] | None = None,
    ) -> None:
        super().__init__()
        if not 0 <= p <= 1:
            msg = f"Probability must be in [0, 1], got {p}"
            raise ValueError(msg)
        self.p = p
        self.copy = copy
        self.per_instance = per_instance
        self.include = include
        self.exclude = exclude

    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)
        _TRANSFORM_REGISTRY[cls.__name__] = cls

    def _warn_if_noop(self, *, is_noop: bool, hint: str) -> None:
        """Warn that the transform leaves the data unchanged.

        Augmentation transforms whose parameters are sampled from a
        range default to an identity (no-op) when constructed with no
        arguments, so that randomness must be requested explicitly. This
        warns the user when that happens (or whenever the given
        parameters produce a no-op).

        Args:
            is_noop: Whether the configured transform is an identity.
            hint: Example argument to suggest in the warning message.
        """
        if is_noop:
            warnings.warn(
                f"{type(self).__name__} is a no-op with the given parameters"
                " and will not change the data. Pass arguments to apply an"
                f" effect (e.g. {hint}), or a range like (a, b) for random"
                " augmentation.",
                stacklevel=3,
            )

    def __repr__(self) -> str:
        """Show only non-default fields for a compact repr."""
        from .parameter_range import _ParameterRange

        parts = []
        for name, default in _collect_init_params(type(self)).items():
            value = getattr(self, name, default)
            if isinstance(value, _ParameterRange):
                if value._original == default:
                    continue
            elif value == default:
                continue
            parts.append(f"{name}={value!r}")
        return f"{type(self).__name__}({', '.join(parts)})"

    def __add__(self, other: object) -> Transform:
        """Compose two transforms: `t1 + t2` → `Compose([t1, t2])`."""
        if not isinstance(other, Transform):
            return NotImplemented
        from .compose import Compose

        left = self.transforms if isinstance(self, Compose) else [self]
        right = other.transforms if isinstance(other, Compose) else [other]
        return Compose([*left, *right])

    def __or__(self, other: object) -> Transform:
        """Random choice: `t1 | t2` → `OneOf([t1, t2])`."""
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
        self._check_spatial_annotations(data)
        if self.copy:
            data = _copy.deepcopy(data)
        batch, unwrap = self._wrap(data)
        # When per-element gating is active, the transform handles the
        # probability itself (masked-out elements get identity params),
        # so skip the batch-wide coin flip here. Apply iff rand < p, so
        # p=0 is always a no-op and p=1 always applies.
        if not self._per_instance_p_active(batch) and torch.rand(1).item() >= self.p:
            return unwrap(batch)
        params = self.make_params(batch)
        batch = self.apply_transform(batch, params)
        # Record history on the batch, unless every element was gated out by
        # per-element probability: that is an exact no-op, and recording it
        # would let history replay (e.g. an invertible spatial transform)
        # trigger an unnecessary identity resample.
        if not _all_elements_gated_out(params):
            trace = AppliedTransform(
                name=type(self).__name__,
                params=params,
                include=_copy_optional_list(self.include),
                exclude=_copy_optional_list(self.exclude),
            )
            if not hasattr(batch, "applied_transforms"):
                batch.applied_transforms = []
            batch.applied_transforms.append(trace)
        result = unwrap(batch)
        # Propagate history to outputs that can carry it
        if (
            hasattr(batch, "applied_transforms")
            and not isinstance(
                result,
                (ImagesBatch, SubjectsBatch, Tensor, np.ndarray),
            )
            and not isinstance(result, dict)
        ):
            with contextlib.suppress(AttributeError):
                result.applied_transforms = list(batch.applied_transforms)
        return result

    def _check_spatial_annotations(self, data: Any) -> None:
        """Reject spatial transforms that would leave stale annotations."""
        if not isinstance(self, SpatialTransform):
            return
        match data:
            case SubjectsBatch() | ImagesBatch():
                has_annotations = data.has_annotations
            case Subject():
                has_annotations = bool(
                    data.points
                    or data.bounding_boxes
                    or any(
                        image.points or image.bounding_boxes
                        for image in data.images.values()
                    )
                )
            case Image():
                has_annotations = bool(data.points or data.bounding_boxes)
            case _:
                has_annotations = False
        if has_annotations:
            msg = (
                "Spatial transforms do not yet support Points or BoundingBoxes."
                " Remove the annotations before applying the transform, or apply"
                " an annotation-aware spatial operation."
            )
            raise NotImplementedError(msg)

    @property
    def supports_per_instance_params(self) -> bool:
        """Whether this transform can sample parameters per batch element.

        Defaults to `False`. Transforms that implement per-instance
        parameter sampling override this to return `True`. When `False`,
        the transform always uses batch-shared parameters regardless of
        the `per_instance` flag, preserving the legacy behavior.
        """
        return False

    @property
    def supports_per_instance_p(self) -> bool:
        """Whether this transform can gate each batch element independently.

        Defaults to `False`. Shape-preserving transforms that implement
        per-element probability override this to return `True`.
        Shape-changing transforms must leave it `False` because masked
        and unmasked elements would have incompatible shapes.
        """
        return False

    def _per_instance_active(self, batch: SubjectsBatch) -> bool:
        """Whether per-instance parameter sampling applies to *batch*.

        Per-instance sampling only kicks in for genuine batches
        (`batch_size > 1`); single-element inputs always use the legacy
        scalar path.
        """
        return (
            self.per_instance
            and self.supports_per_instance_params
            and batch.batch_size > 1
        )

    def _per_instance_p_active(self, batch: SubjectsBatch) -> bool:
        """Whether per-element probability gating applies to *batch*."""
        return (
            self.per_instance
            and self.supports_per_instance_p
            and batch.batch_size > 1
            and 0.0 < self.p < 1.0
        )

    def _resolve_n(self, batch: SubjectsBatch) -> int | None:
        """Return the number of parameter sets to sample.

        Returns:
            The batch size when per-instance sampling is active,
            otherwise `None` (the legacy single-sample path).
        """
        return batch.batch_size if self._per_instance_active(batch) else None

    def _keep_mask(
        self,
        batch: SubjectsBatch,
        n: int | None,
    ) -> Tensor | None:
        """Sample a per-element keep mask for per-instance probability.

        Args:
            batch: The batch being transformed.
            n: The resolved number of parameter sets (from
                `_resolve_n`).

        Returns:
            A boolean tensor of shape `(n,)` where `True` marks
            elements that receive the transform, or `None` when
            per-element gating is not active (all elements are kept).
        """
        if n is None or not self._per_instance_p_active(batch):
            return None
        return torch.rand(n) < self.p

    @staticmethod
    def _mask_identity(
        value: Tensor | float,
        keep: Tensor | None,
        *,
        identity: float,
    ) -> Tensor | float:
        """Replace masked-out elements of *value* with an identity value.

        Args:
            value: Sampled parameter, either a scalar (legacy path) or a
                `(B,)` tensor (per-instance path).
            keep: Per-element keep mask, or `None` for no masking.
            identity: The value that makes the transform a no-op for an
                element (for example `0.0` for additive or log-space
                parameters).

        Returns:
            The masked parameter.
        """
        if keep is None or not isinstance(value, Tensor):
            return value
        return torch.where(keep, value, torch.full_like(value, identity))

    @staticmethod
    def _serialize_param(value: Tensor | Any) -> Any:
        """Convert a possibly-tensor parameter to a JSON-serializable form."""
        if isinstance(value, Tensor):
            return value.tolist()
        return value

    @staticmethod
    def _is_per_instance_params(params: dict[str, Any]) -> bool:
        """Whether *params* holds per-element (batched) values."""
        return "_batched_keys" in params

    def _tag_batched(
        self,
        params: dict[str, Any],
        batch: SubjectsBatch,
        n: int | None,
        keep: Tensor | None,
        batched_keys: list[str],
    ) -> None:
        """Annotate *params* with per-instance bookkeeping for history.

        Adds the batch size, the names of the per-element keys, and the
        keep mask so that [`SubjectsBatch.unbatch`][torchio.SubjectsBatch.unbatch]
        can split the history per subject.

        Args:
            params: The parameter dict to annotate in place.
            batch: The batch being transformed.
            n: The resolved number of parameter sets.
            keep: The per-element keep mask, or `None`.
            batched_keys: Names of the params that hold one value per
                element.
        """
        if n is None:
            return
        params["_batch_size"] = batch.batch_size
        params["_batched_keys"] = list(batched_keys)
        if keep is not None:
            params["_keep"] = keep.tolist()

    def make_params(self, batch: SubjectsBatch) -> dict[str, Any]:
        """Sample random parameters for this transform.

        Override in subclasses that have random behavior.

        Args:
            batch: A `SubjectsBatch`.

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

        Must be overridden by subclasses. Receives a `SubjectsBatch`
        whose `ImagesBatch` entries contain 5D tensors
        `(B, C, I, J, K)`. Use negative indexing (`-3`, `-2`,
        `-1`) for spatial dims.

        Args:
            batch: A `SubjectsBatch` to transform.
            params: Parameters from `make_params`.

        Returns:
            Transformed `SubjectsBatch`.
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
            A new `Transform` instance that inverts this one.
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

        Returns a dict with `_target_` set to the fully qualified
        class name and only non-default field values included.

        Returns:
            Dict suitable for `hydra.utils.instantiate()`.
        """
        from .parameter_range import _ParameterRange

        cls = type(self)
        target = f"torchio.{cls.__qualname__}"
        cfg: dict[str, Any] = {"_target_": target}

        for name, default in _collect_init_params(cls).items():
            value = getattr(self, name, default)
            if isinstance(value, _ParameterRange):
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
                _assign_histories(sb, _get_element_histories(data))
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
    sub.applied_transforms = list(img.applied_transforms)
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
    named parameters (skipping `self`, `*args`, `**kwargs`).
    Returns an ordered dict of `{name: default}`.
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
    image_batch = batch.images["tio_default_image"]
    _assign_histories(image_batch, _get_element_histories(batch))
    return image_batch


def _unwrap_subject(batch: SubjectsBatch) -> Subject:
    return batch.unbatch()[0]


def _unwrap_image(batch: SubjectsBatch) -> Image:
    sub = batch.unbatch()[0]
    image = sub.tio_default_image
    image.applied_transforms = list(sub.applied_transforms)
    return image


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

    Intensity transforms apply only to `ScalarImage` instances,
    leaving `LabelMap` and annotations unchanged.
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
