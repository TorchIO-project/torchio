from __future__ import annotations

from collections.abc import Generator

import numpy as np
import torch

from ...constants import LOCATION
from ...data.subject import Subject
from ...types import TypeSpatialShape
from ...types import TypeTripletInt
from ...utils import to_tuple


class PatchSampler:
    r"""Base class for TorchIO samplers.

    Args:
        patch_size: Tuple of integers :math:`(w, h, d)` to generate patches
            of size :math:`w \times h \times d`.
            If a single number :math:`n` is provided, :math:`w = h = d = n`.

    .. warning:: This is an abstract class that should only be instantiated
        using child classes such as :class:`~torchio.data.UniformSampler` and
        :class:`~torchio.data.WeightedSampler`.
    """

    def __init__(self, patch_size: TypeSpatialShape):
        patch_size_array = np.array(to_tuple(patch_size, length=3))
        for n in patch_size_array:
            if n < 1 or not isinstance(n, (int, np.integer)):
                message = (
                    'Patch dimensions must be positive integers,'
                    f' not {patch_size_array}'
                )
                raise ValueError(message)
        self.patch_size = patch_size_array.astype(np.uint16)

    def extract_patch(
        self,
        subject: Subject,
        index_ini: TypeTripletInt,
    ) -> Subject:
        cropped_subject = self.crop(subject, index_ini, self.patch_size)  # type: ignore[arg-type]
        return cropped_subject

    def crop(
        self,
        subject: Subject,
        index_ini: TypeTripletInt,
        patch_size: TypeTripletInt,
    ) -> Subject:
        transform = self._get_crop_transform(subject, index_ini, patch_size)
        cropped_subject = transform(subject)
        index_ini_array = np.asarray(index_ini)
        patch_size_array = np.asarray(patch_size)
        index_fin = index_ini_array + patch_size_array
        location = index_ini_array.tolist() + index_fin.tolist()
        cropped_subject[LOCATION] = torch.as_tensor(location)
        cropped_subject.update_attributes()
        return cropped_subject

    @staticmethod
    def _get_crop_transform(
        subject,
        index_ini: TypeTripletInt,
        patch_size: TypeSpatialShape,
    ):
        from ...transforms.preprocessing.spatial.crop import Crop

        shape = np.array(subject.spatial_shape, dtype=np.uint16)
        index_ini_array = np.array(index_ini, dtype=np.uint16)
        patch_size_array = np.array(patch_size, dtype=np.uint16)
        assert len(index_ini_array) == 3
        assert len(patch_size_array) == 3
        index_fin = index_ini_array + patch_size_array
        crop_ini = index_ini_array.tolist()
        crop_fin = (shape - index_fin).tolist()
        start = ()
        cropping = sum(zip(crop_ini, crop_fin), start)  # type: ignore[arg-type]
        return Crop(cropping)  # type: ignore[arg-type]

    def __call__(
        self,
        subject: Subject,
        num_patches: int | None = None,
    ) -> Generator[Subject]:
        subject.check_consistent_space()
        if np.any(self.patch_size > subject.spatial_shape):
            message = (
                f'Patch size {tuple(self.patch_size)} cannot be'
                f' larger than image size {tuple(subject.spatial_shape)}'
            )
            raise RuntimeError(message)
        kwargs = {} if num_patches is None else {'num_patches': num_patches}
        return self._generate_patches(subject, **kwargs)

    def _generate_patches(
        self,
        subject: Subject,
        num_patches: int | None = None,
    ) -> Generator[Subject]:
        raise NotImplementedError


class RandomSampler(PatchSampler):
    r"""Base class for random samplers.

    Args:
        patch_size: Tuple of integers :math:`(w, h, d)` to generate patches
            of size :math:`w \times h \times d`.
            If a single number :math:`n` is provided, :math:`w = h = d = n`.
    """

    def get_probability_map(self, subject: Subject):
        raise NotImplementedError
