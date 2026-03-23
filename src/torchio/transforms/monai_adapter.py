from __future__ import annotations

import warnings
from collections.abc import Callable
from collections.abc import Mapping
from types import ModuleType

import numpy as np
import torch

from ..data.image import Image
from ..data.image import ScalarImage
from ..data.subject import Subject
from ..external.imports import get_monai
from .transform import Transform


class MonaiAdapter(Transform):
    """Wraps a MONAI transform for use in TorchIO pipelines.

    This adapter allows using
    [MONAI transforms](https://docs.monai.io/en/stable/transforms.html)
    within TorchIO workflows. Both **dictionary transforms** (e.g.,
    ``NormalizeIntensityd``) and **array transforms** (e.g.,
    ``NormalizeIntensity``) are supported.

    The adapter handles conversion between TorchIO's
    [`Subject`][torchio.Subject] (where values are
    [`Image`][torchio.Image] objects) and MONAI's expected format
    (where values are tensors or
    [`MetaTensor`](https://docs.monai.io/en/stable/data.html#metatensor)
    objects). Image tensors are passed as `MetaTensor` instances with the
    affine matrix embedded, so spatial transforms (e.g., cropping, resizing)
    correctly propagate affine changes.

    **Dictionary transforms** (with a ``keys`` attribute) operate on the
    full subject dictionary — only the keys specified in the MONAI transform
    are modified.

    **Array transforms** (without a ``keys`` attribute) are applied to each
    image in the subject individually, respecting the ``include`` and
    ``exclude`` parameters inherited from
    [`Transform`][torchio.transforms.Transform].

    Args:
        monai_transform: A MONAI transform (dictionary or array) or any
            callable.
        **kwargs: See [`Transform`][torchio.transforms.Transform] for
            additional keyword arguments.

    Examples:
        >>> import torch
        >>> import torchio as tio
        >>> from monai.transforms import NormalizeIntensity
        >>> from monai.transforms import NormalizeIntensityd
        >>> from monai.transforms import RandSpatialCropd
        >>> subject = tio.Subject(
        ...     t1=tio.ScalarImage(tensor=torch.randn(1, 64, 64, 64)),
        ...     seg=tio.LabelMap(tensor=torch.ones(1, 64, 64, 64)),
        ... )
        >>> # Array transform — applied to each image
        >>> adapter = tio.MonaiAdapter(NormalizeIntensity())
        >>> transformed = adapter(subject)
        >>> # Dictionary transform — applied to specified keys
        >>> adapter = tio.MonaiAdapter(NormalizeIntensityd(keys=["t1"]))
        >>> transformed = adapter(subject)
        >>> # Spatial dict transform (affine is updated)
        >>> adapter = tio.MonaiAdapter(
        ...     RandSpatialCropd(keys=["t1", "seg"], roi_size=[32, 32, 32]),
        ... )
        >>> transformed = adapter(subject)
        >>> # Inside a Compose pipeline
        >>> pipeline = tio.Compose([
        ...     tio.ToCanonical(),
        ...     tio.MonaiAdapter(NormalizeIntensity()),
        ...     tio.RandomFlip(),
        ... ])
        >>> transformed = pipeline(subject)
    """

    def __init__(
        self,
        monai_transform: Callable,
        **kwargs,
    ):
        super().__init__(**kwargs)
        if not callable(monai_transform):
            message = (
                f'The monai_transform argument must be callable,'
                f' but got {type(monai_transform)}'
            )
            raise TypeError(message)
        self.monai_transform = monai_transform

    def add_transform_to_subject_history(self, subject: Subject) -> None:
        """Skip history recording for MONAI adapter transforms.

        MONAI transform objects are not serializable, so they cannot be
        reconstructed by
        :meth:`Subject.get_applied_transforms()
        <torchio.Subject.get_applied_transforms>`.
        This is consistent with how other non-reproducible transforms
        (e.g., ``Compose``, ``OneOf``) handle history.
        """

    def apply_transform(self, subject: Subject) -> Subject:
        """Apply the wrapped MONAI transform to a subject.

        Args:
            subject: TorchIO subject to transform.

        Returns:
            The transformed subject with updated tensor data and, for spatial
            transforms, updated affine matrices.
        """
        monai = get_monai()
        is_dict_transform = isinstance(
            self.monai_transform,
            monai.transforms.MapTransform,
        )
        if is_dict_transform:
            self._apply_dict_transform(subject, monai)
        else:
            self._apply_array_transform(subject, monai)
        return subject

    def _apply_dict_transform(
        self,
        subject: Subject,
        monai: ModuleType,
    ) -> None:
        monai_dict = _subject_to_monai_dict(subject, monai)
        result = self.monai_transform(monai_dict)
        if not isinstance(result, Mapping):
            message = (
                'Expected the MONAI dict transform to return a single'
                f' mapping, but got {type(result)}. Multi-sample'
                ' transforms (returning a list of dicts) are not'
                ' supported by MonaiAdapter.'
            )
            raise TypeError(message)
        _update_subject_from_monai_dict(subject, result, monai)

    def _apply_array_transform(
        self,
        subject: Subject,
        monai: ModuleType,
    ) -> None:
        images_dict = subject.get_images_dict(
            intensity_only=False,
            include=self.include,
            exclude=self.exclude,
        )
        if len(images_dict) > 1 and isinstance(
            self.monai_transform,
            monai.transforms.Randomizable,
        ):
            warnings.warn(
                'Applying a MONAI Randomizable array transform to a'
                ' subject with multiple images. Each image will receive'
                ' different random parameters, which may break spatial'
                ' alignment. Consider using the dictionary version of'
                ' this transform instead (e.g., RandFlipd instead of'
                ' RandFlip).',
                UserWarning,
                stacklevel=4,
            )
        for image in images_dict.values():
            meta_tensor = _image_to_meta_tensor(image, monai)
            result = self.monai_transform(meta_tensor)
            if not isinstance(result, torch.Tensor):
                message = (
                    'Expected a torch.Tensor from the MONAI transform'
                    f' output, but got {type(result)}'
                )
                raise TypeError(message)
            _update_image_from_result(image, result, monai)


def _to_meta_tensor(
    tensor: torch.Tensor,
    affine: np.ndarray,
    monai: ModuleType,
) -> torch.Tensor:
    """Convert a tensor to a MONAI MetaTensor with affine."""
    MetaTensor = monai.data.MetaTensor
    affine_tensor = torch.as_tensor(affine, dtype=torch.float64, device=tensor.device)
    return MetaTensor(tensor, affine=affine_tensor)


def _image_to_meta_tensor(
    image: Image,
    monai: ModuleType,
) -> torch.Tensor:
    """Convert a TorchIO Image to a MONAI MetaTensor."""
    return _to_meta_tensor(image.data, image.affine, monai)


def _unwrap_tensor(result: torch.Tensor, monai: ModuleType) -> torch.Tensor:
    """Extract a plain tensor from a result, unwrapping MetaTensor if needed."""
    MetaTensor = monai.data.MetaTensor
    if isinstance(result, MetaTensor):
        return result.as_tensor()
    return result


def _extract_affine(
    result: torch.Tensor,
    monai: ModuleType,
) -> np.ndarray | None:
    """Extract the affine from a MetaTensor, or return None."""
    MetaTensor = monai.data.MetaTensor
    if isinstance(result, MetaTensor):
        return result.affine.cpu().numpy()
    return None


def _update_image_from_result(
    image: Image,
    result: torch.Tensor,
    monai: ModuleType,
) -> None:
    """Update a TorchIO Image from a MONAI transform result tensor."""
    image.set_data(_unwrap_tensor(result, monai))
    new_affine = _extract_affine(result, monai)
    if new_affine is not None and not np.array_equal(new_affine, image.affine):
        image.affine = new_affine


def _subject_to_monai_dict(
    subject: Subject,
    monai: ModuleType,
) -> dict[str, object]:
    """Convert a Subject to a MONAI-compatible dictionary.

    Image values are converted to MetaTensor instances.
    Non-image values are passed through unchanged.
    """
    monai_dict: dict[str, object] = {}
    for key, value in subject.items():
        if isinstance(value, Image):
            monai_dict[key] = _image_to_meta_tensor(value, monai)
        else:
            monai_dict[key] = value
    return monai_dict


def _update_subject_from_monai_dict(
    subject: Subject,
    monai_dict: dict[str, object],
    monai: ModuleType,
) -> None:
    """Update a Subject in-place from the MONAI transform output.

    For keys that correspond to Images in the subject, the tensor data
    is updated. If the output is a MetaTensor with an updated affine,
    the Image affine is also updated.
    """
    for key, value in monai_dict.items():
        if key in subject and isinstance(subject[key], Image):
            image = subject[key]
            assert isinstance(image, Image)
            if not isinstance(value, torch.Tensor):
                message = (
                    'Expected a torch.Tensor from the MONAI transform'
                    f' output, but got {type(value)}'
                )
                raise TypeError(message)
            _update_image_from_result(image, value, monai)
        else:
            # New key or non-image key from MONAI output
            if isinstance(value, torch.Tensor):
                tensor = _unwrap_tensor(value, monai)
                if 2 <= tensor.ndim <= 5:
                    new_affine = _extract_affine(value, monai)
                    if new_affine is None:
                        new_affine = np.eye(4)
                    subject[key] = ScalarImage(
                        tensor=tensor,
                        affine=new_affine,
                    )
                else:
                    subject[key] = tensor
            else:
                subject[key] = value
