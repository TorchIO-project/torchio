import numpy as np
import torch

from ....data.image import Image
from ....data.subject import Subject
from ...spatial_transform import SpatialTransform
from .resample import Resample


class ToReferenceSpace(SpatialTransform):
    """Modify the spatial metadata so it matches a reference space.

    This is useful, for example, to set meaningful spatial metadata of a neural
    network embedding, for visualization or further processing such as
    resampling a segmentation output.

    Example:

    >>> import torchio as tio
    >>> image = tio.datasets.FPG().t1
    >>> embedding_tensor = my_network(image.tensor)  # we lose metadata here
    >>> embedding_image = tio.ToReferenceSpace.from_tensor(embedding_tensor, image)
    """

    def __init__(self, reference: Image, **kwargs):
        super().__init__(**kwargs)
        if not isinstance(reference, Image):
            raise TypeError('The reference must be a TorchIO image')
        self.reference = reference

    def apply_transform(self, subject: Subject) -> Subject:
        images_dict = subject.get_images_dict(
            intensity_only=False,
            include=self.include,
            exclude=self.exclude,
        )
        for image_name, image in images_dict.items():
            new_image = build_image_from_reference(image.data, self.reference)
            subject[image_name] = new_image
        
        # Update attributes to sync dictionary changes with attribute access
        subject.update_attributes()
        return subject

    @staticmethod
    def from_tensor(tensor: torch.Tensor, reference: Image) -> Image:
        """Build a TorchIO image from a tensor and a reference image."""
        return build_image_from_reference(tensor, reference)


def build_image_from_reference(tensor: torch.Tensor, reference: Image) -> Image:
    input_shape = np.array(reference.spatial_shape)
    output_shape = np.array(tensor.shape[-3:])
    downsampling_factor = input_shape / output_shape
    input_spacing = np.array(reference.spacing)
    output_spacing = input_spacing * downsampling_factor
    downsample = Resample(output_spacing, image_interpolation='nearest')
    reference = downsample(reference)
    result = reference.new_like(tensor=tensor, affine=reference.affine)
    return result
