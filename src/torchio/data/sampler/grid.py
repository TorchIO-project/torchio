from __future__ import annotations

from collections.abc import Generator

import numpy as np

from ...data.subject import Subject
from ...types import TypeSpatialShape
from ...types import TypeTripletInt
from ...utils import to_tuple
from .sampler import PatchSampler


class GridSampler(PatchSampler):
    r"""Extract patches across a whole volume.

    Grid samplers are useful to perform inference using all patches from a
    volume. It is often used with a [`GridAggregator`](../inference/#torchio.data.GridAggregator).

    Args:
        subject: Instance of [`Subject`](../../data/subject/#torchio.Subject)
            from which patches will be extracted.
        patch_size: Tuple of integers $(w, h, d)$ to generate patches
            of size $w \times h \times d$.
            If a single number $n$ is provided,
            $w = h = d = n$.
        patch_overlap: Tuple of even integers $(w_o, h_o, d_o)$
            specifying the overlap between patches for dense inference. If a
            single number $n$ is provided, $w_o = h_o = d_o = n$.
        padding_mode: Same as `padding_mode` in
            [`Pad`][torchio.transforms.Pad]. If `None`, the volume will not
            be padded before sampling and patches at the border will not be
            cropped by the aggregator.
            Otherwise, the volume will be padded with
            $\left(\frac{w_o}{2}, \frac{h_o}{2}, \frac{d_o}{2} \right)$
            on each side before sampling. If the sampler is passed to a
            [`GridAggregator`](../inference/#torchio.data.GridAggregator), it will crop the output
            to its original size.

    Examples:
        >>> import torchio as tio
        >>> colin = tio.datasets.Colin27()
        >>> sampler = tio.GridSampler(colin, patch_size=88)
        >>> for i, patch in enumerate(sampler()):
        ...     patch.t1.save(f'patch_{i}.nii.gz')
        ...
        >>> # To figure out the number of patches beforehand:
        >>> sampler = tio.GridSampler(colin, patch_size=88)
        >>> len(sampler)
        8

    Note:
        Adapted from NiftyNet. See [this NiftyNet tutorial
        ](https://niftynet.readthedocs.io/en/dev/window_sizes.html) for more
        information about patch based sampling. Note that
        `patch_overlap` is twice `border` in NiftyNet
        tutorial.
    """

    def __init__(
        self,
        subject: Subject,
        patch_size: TypeSpatialShape,
        patch_overlap: TypeSpatialShape = (0, 0, 0),
        padding_mode: str | float | None = None,
    ):
        super().__init__(patch_size)
        self.patch_overlap = np.array(to_tuple(patch_overlap, length=3))
        self.padding_mode = padding_mode
        self.subject = self._pad(subject)
        self.locations = self._compute_locations(self.subject)

    def __len__(self):
        return len(self.locations)

    def __getitem__(self, index):
        # Assume 3D
        location = self.locations[index]
        index_ini = (
            int(location[0]),
            int(location[1]),
            int(location[2]),
        )
        si, sj, sk = (int(value) for value in self.patch_size.tolist())
        patch_size = si, sj, sk
        cropped_subject = self.crop(self.subject, index_ini, patch_size)
        return cropped_subject

    def __call__(
        self,
        subject: Subject | None = None,
        num_patches: int | None = None,
    ) -> Generator[Subject]:
        subject = self.subject if subject is None else subject
        return super().__call__(subject, num_patches=num_patches)

    def _pad(self, subject: Subject) -> Subject:
        if self.padding_mode is not None:
            from ...transforms import Pad

            border = self.patch_overlap // 2
            padding_values = [int(value) for value in border.repeat(2).tolist()]
            padding = (
                padding_values[0],
                padding_values[1],
                padding_values[2],
                padding_values[3],
                padding_values[4],
                padding_values[5],
            )
            pad = Pad(padding, padding_mode=self.padding_mode)
            transformed = pad(subject)
            assert isinstance(transformed, Subject)
            subject = transformed
        return subject

    def _compute_locations(self, subject: Subject):
        patch_size_values = [int(value) for value in self.patch_size.tolist()]
        patch_overlap_values = [int(value) for value in self.patch_overlap.tolist()]
        patch_size = (
            patch_size_values[0],
            patch_size_values[1],
            patch_size_values[2],
        )
        patch_overlap = (
            patch_overlap_values[0],
            patch_overlap_values[1],
            patch_overlap_values[2],
        )
        self._parse_sizes(subject.spatial_shape, patch_size, patch_overlap)
        return self._get_patches_locations(
            subject.spatial_shape, patch_size, patch_overlap
        )

    def _generate_patches(
        self,
        subject: Subject,
        num_patches: int | None = None,
    ) -> Generator[Subject]:
        if num_patches is not None:
            message = 'GridSampler does not support limiting the number of patches'
            raise ValueError(message)
        subject = self._pad(subject)
        locations = self._compute_locations(subject)
        for location in locations:
            index_ini = (
                int(location[0]),
                int(location[1]),
                int(location[2]),
            )
            yield self.extract_patch(subject, index_ini)

    @staticmethod
    def _parse_sizes(
        image_size: TypeTripletInt,
        patch_size: TypeTripletInt,
        patch_overlap: TypeTripletInt,
    ) -> None:
        image_size_array = np.array(image_size)
        patch_size_array = np.array(patch_size)
        patch_overlap_array = np.array(patch_overlap)
        if np.any(patch_size_array > image_size_array):
            message = (
                f'Patch size {tuple(patch_size_array)} cannot be'
                f' larger than image size {tuple(image_size_array)}'
            )
            raise ValueError(message)
        if np.any(patch_overlap_array >= patch_size_array):
            message = (
                f'Patch overlap {tuple(patch_overlap_array)} must be smaller'
                f' than patch size {tuple(patch_size_array)}'
            )
            raise ValueError(message)
        if np.any(patch_overlap_array % 2):
            message = (
                'Patch overlap must be a tuple of even integers,'
                f' not {tuple(patch_overlap_array)}'
            )
            raise ValueError(message)

    @staticmethod
    def _get_patches_locations(
        image_size: TypeTripletInt,
        patch_size: TypeTripletInt,
        patch_overlap: TypeTripletInt,
    ) -> np.ndarray:
        # Example with image_size 10, patch_size 5, overlap 2:
        # [0 1 2 3 4 5 6 7 8 9]
        # [0 0 0 0 0]
        #       [1 1 1 1 1]
        #           [2 2 2 2 2]
        # Locations:
        # [[0, 5],
        #  [3, 8],
        #  [5, 10]]
        indices = []
        zipped = zip(image_size, patch_size, patch_overlap, strict=True)
        for im_size_dim, patch_size_dim, patch_overlap_dim in zipped:
            end = im_size_dim + 1 - patch_size_dim
            step = patch_size_dim - patch_overlap_dim
            indices_dim = list(range(0, end, step))
            if indices_dim[-1] != im_size_dim - patch_size_dim:
                indices_dim.append(im_size_dim - patch_size_dim)
            indices.append(indices_dim)
        indices_ini = np.array(np.meshgrid(*indices)).reshape(3, -1).T
        indices_ini = np.unique(indices_ini, axis=0)
        indices_fin = indices_ini + np.array(patch_size)
        locations = np.hstack((indices_ini, indices_fin))
        return np.array(sorted(locations.tolist()))
