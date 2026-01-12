from ....data.subject import Subject
from ...spatial_transform import SpatialTransform
from .to_orientation import ToOrientation


class Transpose(SpatialTransform):
    """Swap the first and last spatial dimensions of the image.

    The spatial metadata is updated accordingly, so the world coordinates of
    all voxels in the input and output spaces match.

    Example:

    >>> import torchio as tio
    >>> image = tio.datasets.FPG().t1
    >>> image
    ScalarImage(shape: (1, 256, 256, 176); spacing: (1.00, 1.00, 1.00); orientation: PIR+; path: "/home/fernando/.cache/torchio/fpg/t1.nii.gz")
    >>> transpose = tio.Transpose()
    >>> transposed = transpose(image)
    >>> transposed
    ScalarImage(shape: (1, 176, 256, 256); spacing: (1.00, 1.00, 1.00); orientation: RIP+; dtype: torch.IntTensor; memory: 44.0 MiB)
    """

    def apply_transform(self, subject: Subject) -> Subject:
        images_dict = subject.get_images_dict(
            intensity_only=False,
            include=self.include,
            exclude=self.exclude,
        )
        for image_name, image in images_dict.items():
            old_orientation = image.orientation_str
            new_orientation = old_orientation[::-1]
            transform = ToOrientation(new_orientation)
            transposed = transform(image)
            new_image = image.new_like(tensor=transposed.data, affine=transposed.affine)
            subject[image_name] = new_image
        
        # Update attributes to sync dictionary changes with attribute access
        subject.update_attributes()
        return subject

    def is_invertible(self):
        return True

    def inverse(self):
        return self
