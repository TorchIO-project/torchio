"""IntensityInversion: flip image intensities (max - x + min) per channel."""

from __future__ import annotations

from typing import Any

from ...data.batch import SubjectsBatch
from ..transform import IntensityTransform


class IntensityInversion(IntensityTransform):
    r"""Invert the intensities of an image: $x \mapsto \max - x + \min$.

    The brightest voxel becomes the darkest and vice versa, while the
    intensity range is preserved.  The minimum and maximum are computed
    **per channel**, so channels with different ranges are each inverted
    within their own range.  Label maps are left untouched.

    This augmentation discourages a model from relying on absolute
    intensity polarity, which is useful when training on one MRI pulse
    sequence but generalizing to others where the contrast may be
    reversed.

    Note:
        The transform is deterministic: there is no parameter to sample.
        To apply it stochastically, use the inherited ``p`` argument,
        e.g. ``IntensityInversion(p=0.5)`` inverts each subject with
        probability 0.5.  The operation is its own inverse.

    Warning:
        CT intensities (Hounsfield units) are physically meaningful and
        standardized, so inverting them is generally **not**
        appropriate.  This transform is intended for MRI-style
        cross-contrast augmentation.

    Args:
        **kwargs: See [`Transform`][torchio.Transform].

    Examples:
        >>> import torchio as tio
        >>> transform = tio.IntensityInversion()
        >>> transform = tio.IntensityInversion(p=0.5)
    """

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)

    def make_params(self, batch: SubjectsBatch) -> dict[str, Any]:
        """Record which images are processed, so the inverse can match them.

        The base transform history only stores the sampled params, not the
        ``include``/``exclude`` filter, so the inverse would otherwise invert
        *every* intensity image. We record the actually-processed names here
        and scope the inverse to them.
        """
        return {"image_names": list(self._get_images(batch).keys())}

    def apply_transform(
        self,
        batch: SubjectsBatch,
        params: dict[str, Any],
    ) -> SubjectsBatch:
        """Invert each intensity image per channel."""
        # Data is (B, C, I, J, K); compute min/max over the spatial dims so
        # each channel of each sample is inverted within its own range.
        spatial_dims = (2, 3, 4)
        for _name, img_batch in self._get_images(batch).items():
            data = img_batch.data
            maxima = data.amax(dim=spatial_dims, keepdim=True)
            minima = data.amin(dim=spatial_dims, keepdim=True)
            img_batch.data = maxima - data + minima
        return batch

    @property
    def invertible(self) -> bool:
        """Whether this transform can be inverted."""
        return True

    def inverse(self, params: dict[str, Any]) -> IntensityInversion:
        """Invert by applying the same transform again (it is self-inverse).

        Scoped to the images recorded in ``make_params`` so that, under
        ``include``/``exclude``, images that were never inverted stay
        untouched.
        """
        return IntensityInversion(include=params["image_names"], copy=False)
