"""CornucopiaAdapter: wrap Cornucopia transforms for use in TorchIO pipelines."""

from __future__ import annotations

import copy as _copy
from collections.abc import Callable
from typing import Any

import torch

from ..data.batch import SubjectsBatch
from ..data.image import Image
from ..data.image import LabelMap
from ..data.image import ScalarImage
from ..data.subject import Subject
from .transform import Transform


class CornucopiaAdapter(Transform):
    """Wrap a Cornucopia transform for use in TorchIO pipelines.

    `Cornucopia <https://cornucopia.readthedocs.io/>`_ transforms
    operate on ``(C, I, J, K)`` tensors and support passing multiple
    tensors to share spatial parameters (e.g., the same elastic
    deformation is applied to an image and its segmentation).

    The adapter extracts image tensors from the subject, passes them
    to the Cornucopia transform as positional arguments (scalar images
    first, then label maps), and writes the results back.

    Args:
        cornucopia_transform: A Cornucopia transform (any callable
            accepting one or more ``(C, I, J, K)`` tensors).
            Requires ``cornucopia`` to be installed:
            ``pip install cornucopia``.
        **kwargs: See [`Transform`][torchio.Transform].

    Examples:
        >>> import torchio as tio
        >>> import cornucopia as cc  # doctest: +SKIP
        >>> adapter = tio.CornucopiaAdapter(
        ...     cc.ElasticTransform(),
        ... )  # doctest: +SKIP
        >>> result = adapter(subject)  # doctest: +SKIP

    Note:
        ``CornucopiaAdapter`` does **not** record itself in the
        subject's transform history, because Cornucopia transform
        objects are not guaranteed to be serializable.
    """

    def __init__(
        self,
        cornucopia_transform: Callable,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        if not callable(cornucopia_transform):
            msg = (
                "cornucopia_transform must be callable, "
                f"got {type(cornucopia_transform).__name__}"
            )
            raise TypeError(msg)
        self.cornucopia_transform = cornucopia_transform

    def forward(self, data: Any) -> Any:
        """Apply without recording history."""
        batch, unwrap = self._wrap(data)
        if self.copy:
            batch = _copy.deepcopy(batch)
        if torch.rand(1).item() > self.p:
            return unwrap(batch)
        subjects = batch.unbatch()
        for subject in subjects:
            _apply_cornucopia(subject, self.cornucopia_transform, self)
        from ..data.batch import SubjectsBatch

        result = SubjectsBatch.from_subjects(subjects)
        result.applied_transforms = batch.applied_transforms
        return unwrap(result)

    def apply_transform(
        self,
        batch: SubjectsBatch,
        params: dict[str, Any],
    ) -> SubjectsBatch:
        """Not used — CornucopiaAdapter overrides forward directly."""
        return batch

    def add_transform_to_subject_history(self, *args: Any) -> None:
        """No-op: Cornucopia transforms are opaque."""

    @property
    def invertible(self) -> bool:
        """Cornucopia transforms are not invertible through TorchIO."""
        return False


def _apply_cornucopia(
    subject: Subject,
    cornucopia_transform: Callable,
    adapter: CornucopiaAdapter,
) -> None:
    """Apply a Cornucopia transform to a subject's images.

    All images are passed as positional arguments so that spatial
    transforms share the same random parameters.  Scalar images are
    passed first, then label maps.

    Args:
        subject: The subject to transform.
        cornucopia_transform: The Cornucopia callable.
        adapter: The adapter instance (for include/exclude filtering).
    """
    images = _get_filtered_images(subject, adapter)
    if not images:
        return

    names = list(images.keys())
    tensors = [images[n].data for n in names]

    # Cornucopia transforms accept multiple tensors as *args
    # and return the same number of tensors.
    results = cornucopia_transform(*tensors)

    # If only one image, result is a single tensor (not a tuple).
    if len(names) == 1:
        results = (results,)

    for name, result_tensor in zip(names, results, strict=True):
        if isinstance(result_tensor, torch.Tensor):
            images[name].set_data(result_tensor)


def _filter_images(
    images: dict[str, Image],
    include: list[str] | None,
    exclude: list[str] | None,
) -> dict[str, Image]:
    """Apply include/exclude filters to an image dict."""
    if include is not None:
        images = {k: v for k, v in images.items() if k in include}
    if exclude is not None:
        images = {k: v for k, v in images.items() if k not in exclude}
    return images


def _get_filtered_images(
    subject: Subject,
    adapter: CornucopiaAdapter,
) -> dict[str, Image]:
    """Get images from a subject, filtered and ordered.

    Scalar images come first so that Cornucopia transforms that
    treat the first argument specially (e.g., intensity-only noise)
    apply to the right image.

    Args:
        subject: The subject.
        adapter: The adapter (for include/exclude).

    Returns:
        Ordered dict: scalar images first, then label maps.
    """
    filtered = _filter_images(subject.images, adapter.include, adapter.exclude)
    scalars = {k: v for k, v in filtered.items() if isinstance(v, ScalarImage)}
    labels = {k: v for k, v in filtered.items() if isinstance(v, LabelMap)}
    return {**scalars, **labels}
