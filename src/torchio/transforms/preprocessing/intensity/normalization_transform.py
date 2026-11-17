import torch

from ....data.subject import Subject
from ....transforms.transform import TypeMaskingMethod
from ...intensity_transform import IntensityTransform


class NormalizationTransform(IntensityTransform):
    """Base class for intensity preprocessing transforms.

    Args:
        masking_method: Defines the mask used to compute the normalization statistics. It can be one of:

            - `None`: the mask image is all ones, i.e. all values in the image are used.

            - A string: key to a [`torchio.LabelMap`][torchio.LabelMap] in the subject which is used as a mask,
              OR an anatomical label: `'Left'`, `'Right'`, `'Anterior'`, `'Posterior'`,
              `'Inferior'`, `'Superior'` which specifies a side of the mask volume to be ones.

            - A function: the mask image is computed as a function of the intensity image.
              The function must receive and return a [`torch.Tensor`][torch.Tensor]

            - Integer bounds: exclude border voxels from the statistics. An integer excludes that many
              voxels at both ends of each spatial axis. A triple `(i, j, k)` sets the same count at both
              ends of each axis independently. Six integers `(i0, i1, j0, j1, k0, k1)` set the lower and
              upper counts for each axis separately. The bounds select an interior region and leave the
              image shape unchanged. Choose counts that leave enough voxels to compute the statistics.

        **kwargs: See [`Transform`][torchio.transforms.Transform] for additional
            keyword arguments.

    Examples:
        >>> import torchio as tio
        >>> subject = tio.datasets.Colin27()
        >>> subject
        Colin27(Keys: ('t1', 'head', 'brain'); images: 3)
        >>> transform = tio.ZNormalization()  # ZNormalization is a subclass of NormalizationTransform
        >>> transformed = transform(subject)  # use all values to compute mean and std
        >>> transform = tio.ZNormalization(masking_method='brain')
        >>> transformed = transform(subject)  # use only values within the brain
        >>> transform = tio.ZNormalization(masking_method=lambda x: x > x.mean())
        >>> transformed = transform(subject)  # use values above the image mean

        Use an integer to exclude one voxel at both ends of every spatial axis:

        >>> import torch
        >>> image = tio.ScalarImage(tensor=torch.arange(64, dtype=torch.float32).reshape(1, 4, 4, 4))
        >>> normalized = tio.ZNormalization(masking_method=1)(image)
        >>> normalized.shape == image.shape
        True
        >>> interior = normalized.data[:, 1:-1, 1:-1, 1:-1]
        >>> torch.isclose(interior.mean(), torch.tensor(0.0), atol=1e-6).item()
        True
    """

    def __init__(self, masking_method: TypeMaskingMethod = None, **kwargs):
        super().__init__(**kwargs)
        self.masking_method = masking_method

    def apply_transform(self, subject: Subject) -> Subject:
        for image_name, image in self.get_images_dict(subject).items():
            mask = self.get_mask_from_masking_method(
                self.masking_method,
                subject,
                image.data,
            )
            self.apply_normalization(subject, image_name, mask)
        return subject

    def apply_normalization(
        self,
        subject: Subject,
        image_name: str,
        mask: torch.Tensor,
    ) -> None:
        # There must be a nicer way of doing this
        raise NotImplementedError
