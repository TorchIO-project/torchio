"""Transform base classes."""

from __future__ import annotations

import copy as _copy
import inspect
from dataclasses import dataclass
from dataclasses import field
from typing import Any
from typing import TypeVar

import nibabel as nib
import numpy as np
import SimpleITK as sitk
import torch
from einops import rearrange
from torch import Tensor
from torch import nn

from ..data.image import Image
from ..data.image import ScalarImage
from ..data.subject import Subject

#: TypeVar preserving the input type through transforms.
T = TypeVar(
    "T",
    Subject,
    Image,
    Tensor,
    np.ndarray,
    sitk.Image,
    nib.Nifti1Image,
    dict,
)


@dataclass
class AppliedTransform:
    """Record of a transform application, stored in Subject history.

    Attributes:
        name: Class name of the transform.
        params: Sampled parameters (JSON-serializable).
    """

    name: str
    params: dict[str, Any] = field(default_factory=dict)


class Transform(nn.Module):
    """Base class for all TorchIO transforms.

    Transforms accept a ``Subject``, ``Image``, or ``Tensor`` and return
    the same type. Internally, non-Subject inputs are wrapped in a
    temporary Subject and unwrapped on output.

    Args:
        p: Probability of applying the transform.
        copy: Deep-copy the input before transforming. Set to ``False``
            for in-place operation (e.g., inside a ``Compose`` that
            already copied).
        include: Image names to include (``None`` = all).
        exclude: Image names to exclude (``None`` = none).
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

    def forward(
        self,
        data: T,
    ) -> T:
        """Apply the transform.

        Args:
            data: A Subject, Image, 4D Tensor, NumPy array,
                SimpleITK Image, or NiBabel Nifti1Image.

        Returns:
            Transformed data of the same type as input.
        """
        subject, unwrap = self._wrap(data)
        if self.copy:
            subject = _copy.deepcopy(subject)
        if torch.rand(1).item() > self.p:
            return unwrap(subject)
        params = self.make_params(subject)
        subject = self.apply_transform(subject, params)
        subject.applied_transforms.append(
            AppliedTransform(
                name=type(self).__name__,
                params=params,
            ),
        )
        return unwrap(subject)

    def make_params(self, subject: Subject) -> dict[str, Any]:
        """Sample random parameters for this transform.

        Override in subclasses that have random behavior.

        Args:
            subject: The input subject (for shape-dependent sampling).

        Returns:
            Dict of sampled parameters.
        """
        return {}

    def apply_transform(self, subject: Subject, params: dict[str, Any]) -> Subject:
        """Apply the transform with the given parameters.

        Must be overridden by subclasses.

        Args:
            subject: Subject to transform.
            params: Parameters from ``make_params``.

        Returns:
            Transformed subject.
        """
        raise NotImplementedError

    def to_hydra(self) -> dict[str, Any]:
        """Export as a Hydra-compatible config dict.

        Returns a dict with ``_target_`` set to the fully qualified
        class name and only non-default field values included.
        Values are plain Python types (no ParameterRange, no Tensor).

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

    def to_yaml(self) -> str:
        """Export as a YAML string for Hydra.

        Requires PyYAML (part of the standard scientific stack).

        Returns:
            YAML string.
        """
        import yaml

        return yaml.dump(
            self.to_hydra(),
            default_flow_style=False,
            sort_keys=False,
        )

    def _get_images(self, subject: Subject) -> dict[str, Image]:
        """Get images filtered by include/exclude."""
        images = subject.images
        if self.include is not None:
            images = {k: v for k, v in images.items() if k in self.include}
        if self.exclude is not None:
            images = {k: v for k, v in images.items() if k not in self.exclude}
        return images

    @staticmethod
    def _wrap(
        data: T,
    ) -> tuple[Subject, Any]:
        """Wrap non-Subject input into a Subject; return (subject, unwrap_fn)."""
        if isinstance(data, Subject):
            return data, _unwrap_subject
        if isinstance(data, Image):
            sub = Subject(tio_default_image=data)
            return sub, _unwrap_image
        if isinstance(data, Tensor):
            img = ScalarImage.from_tensor(data)
            sub = Subject(tio_default_image=img)
            return sub, _unwrap_tensor
        if isinstance(data, np.ndarray):
            tensor = torch.as_tensor(data.copy(), dtype=torch.float32)
            if tensor.ndim == 3:
                tensor = rearrange(tensor, "i j k -> 1 i j k")
            img = ScalarImage.from_tensor(tensor)
            sub = Subject(tio_default_image=img)
            return sub, _unwrap_ndarray
        if isinstance(data, sitk.Image):
            img = ScalarImage.from_sitk(data)
            sub = Subject(tio_default_image=img)
            return sub, _unwrap_sitk
        if isinstance(data, nib.Nifti1Image):
            img = ScalarImage.from_nifti(data)
            sub = Subject(tio_default_image=img)
            return sub, _unwrap_nifti
        if isinstance(data, dict):
            kwargs: dict[str, Any] = {}
            for k, v in data.items():
                if isinstance(v, Image):
                    kwargs[k] = v
                elif isinstance(v, Tensor):
                    kwargs[k] = ScalarImage.from_tensor(v)
                else:
                    kwargs[k] = v
            sub = Subject(**kwargs)
            keys = list(data.keys())
            return sub, lambda s: _unwrap_dict(s, keys)
        msg = (
            "Expected Subject, Image, Tensor, ndarray, dict,"
            f" SimpleITK Image, or NIfTI, got {type(data).__name__}"
        )
        raise TypeError(msg)


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


def _unwrap_subject(subject: Subject) -> Subject:
    return subject


def _unwrap_image(subject: Subject) -> Image:
    return subject.tio_default_image


def _unwrap_tensor(subject: Subject) -> Tensor:
    return subject.tio_default_image.data


def _unwrap_ndarray(subject: Subject) -> np.ndarray:
    return subject.tio_default_image.data.numpy()


def _unwrap_sitk(subject: Subject) -> sitk.Image:
    image = subject.tio_default_image
    data = image.data
    affine = image.affine
    array = data.numpy()
    array = array[0] if data.shape[0] == 1 else np.moveaxis(array, 0, -1)
    sitk_image = sitk.GetImageFromArray(array)
    if data.shape[0] > 1:
        sitk_image = sitk.GetImageFromArray(array, isVector=True)
    sitk_image.SetSpacing(affine.spacing)
    sitk_image.SetOrigin(affine.origin)
    sitk_image.SetDirection(rearrange(affine.direction, "i j -> (i j)").tolist())
    return sitk_image


def _unwrap_nifti(subject: Subject) -> nib.Nifti1Image:
    image = subject.tio_default_image
    array = image.data.numpy()
    array = array[0] if array.shape[0] == 1 else np.moveaxis(array, 0, -1)
    return nib.Nifti1Image(array, image.affine.numpy())


def _unwrap_dict(subject: Subject, keys: list[str]) -> dict[str, Tensor]:
    result: dict[str, Any] = {}
    for k in keys:
        entry = getattr(subject, k, None)
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

    def _get_images(self, subject: Subject) -> dict[str, Image]:
        """Filter to ScalarImage only, then apply include/exclude."""
        images: dict[str, Image] = {
            k: v for k, v in subject.images.items() if isinstance(v, ScalarImage)
        }
        if self.include is not None:
            images = {k: v for k, v in images.items() if k in self.include}
        if self.exclude is not None:
            images = {k: v for k, v in images.items() if k not in self.exclude}
        return images
