"""MonaiAdapter: wrap MONAI transforms for use in TorchIO pipelines."""

from __future__ import annotations

import warnings
from collections.abc import Callable
from collections.abc import Mapping
from types import ModuleType
from typing import Any

import attrs
import torch

from ..data.image import Image
from ..data.image import ScalarImage
from ..data.subject import Subject
from ..imports import get_monai
from .transform import T
from .transform import Transform


@attrs.define(slots=False, eq=False, kw_only=True, repr=False)
class MonaiAdapter(Transform):
    """Wrap a MONAI transform for use in TorchIO pipelines.

    Both **dictionary transforms** (e.g., ``NormalizeIntensityd``)
    and **array transforms** (e.g., ``NormalizeIntensity``) are
    supported.

    Dictionary transforms operate on the full subject dict — only the
    keys specified in the MONAI transform are modified.

    Array transforms are applied to each ``ScalarImage`` in the
    subject, respecting ``include`` / ``exclude``.

    Args:
        monai_transform: A MONAI transform or any callable.
            Requires MONAI: ``pip install torchio[monai]``.

    Examples:
        >>> from monai.transforms import NormalizeIntensity
        >>> adapter = tio.MonaiAdapter(NormalizeIntensity())
        >>> result = adapter(subject)
    """

    monai_transform: Callable = attrs.field(kw_only=True)

    def __init__(self, monai_transform: Callable, **kwargs: Any) -> None:
        if not callable(monai_transform):
            msg = (
                "monai_transform must be callable, "
                f"got {type(monai_transform).__name__}"
            )
            raise TypeError(msg)
        self.__attrs_init__(monai_transform=monai_transform, **kwargs)

    def forward(
        self,
        data: T,
    ) -> T:
        """Apply without recording history (MONAI transforms are opaque)."""
        subject, unwrap = self._wrap(data)
        if torch.rand(1).item() > self.p:
            return unwrap(subject)
        self.apply(subject, {})
        return unwrap(subject)

    def apply(self, subject: Subject, params: dict[str, Any]) -> Subject:
        monai = get_monai()
        is_dict = isinstance(
            self.monai_transform,
            monai.transforms.MapTransform,
        )
        if is_dict:
            _apply_dict_transform(subject, self.monai_transform, monai)
        else:
            images = self._get_images(subject)
            _apply_array_transform(
                images,
                self.monai_transform,
                monai,
            )
        return subject

    def _get_images(self, subject: Subject) -> dict[str, Image]:
        """Filter to ScalarImage, then apply include/exclude."""
        images = {k: v for k, v in subject.images.items() if isinstance(v, ScalarImage)}
        if self.include is not None:
            images = {k: v for k, v in images.items() if k in self.include}
        if self.exclude is not None:
            images = {k: v for k, v in images.items() if k not in self.exclude}
        return images


# ── Helpers ──────────────────────────────────────────────────────────


def _image_to_meta_tensor(
    image: Image,
    monai: ModuleType,
) -> torch.Tensor:
    affine_tensor = image.affine.data.to(device=image.data.device)
    return monai.data.MetaTensor(image.data, affine=affine_tensor)


def _update_image_from_result(
    image: Image,
    result: torch.Tensor,
    monai: ModuleType,
) -> None:
    meta_tensor_cls = monai.data.MetaTensor
    if isinstance(result, meta_tensor_cls):
        image.set_data(result.as_tensor())
        new_affine = result.affine
        if not torch.equal(new_affine.cpu().to(torch.float64), image.affine.data.cpu()):
            from ..data.affine import Affine

            image._affine = Affine(new_affine)
    else:
        image.set_data(result)


def _apply_array_transform(
    images: dict[str, Image],
    monai_transform: Callable,
    monai: ModuleType,
) -> None:
    if len(images) > 1 and isinstance(
        monai_transform,
        monai.transforms.Randomizable,
    ):
        warnings.warn(
            "Applying a MONAI Randomizable array transform to multiple"
            " images. Each image gets different random parameters —"
            " use the dictionary version (e.g., RandFlipd) to keep"
            " spatial alignment.",
            UserWarning,
            stacklevel=5,
        )
    for image in images.values():
        meta_tensor = _image_to_meta_tensor(image, monai)
        result = monai_transform(meta_tensor)
        if not isinstance(result, torch.Tensor):
            msg = (
                "Expected torch.Tensor from MONAI transform, "
                f"got {type(result).__name__}"
            )
            raise TypeError(msg)
        _update_image_from_result(image, result, monai)


def _apply_dict_transform(
    subject: Subject,
    monai_transform: Callable,
    monai: ModuleType,
) -> None:
    monai_dict: dict[str, Any] = {}
    for name, image in subject.images.items():
        monai_dict[name] = _image_to_meta_tensor(image, monai)
    for key, value in subject.metadata.items():
        monai_dict[key] = value

    result = monai_transform(monai_dict)

    if not isinstance(result, Mapping):
        msg = f"Expected mapping from MONAI dict transform, got {type(result).__name__}"
        raise TypeError(msg)

    for name, image in subject.images.items():
        if name in result and isinstance(result[name], torch.Tensor):
            _update_image_from_result(image, result[name], monai)
