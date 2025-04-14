import nibabel as nib
import numpy as np
import torch

from ....data.subject import Subject
from ...spatial_transform import SpatialTransform


class ToOrientation(SpatialTransform):
    """Reorient the data to a specified orientation.

    This transform reorders the voxels and modifies the affine matrix to match
    the specified orientation code.

    Common orientation codes include:

    - "RAS": (neurological convention)
        - The first axis goes from Left to Right (R).
        - The second axis goes from Posterior to Anterior (A).
        - The third axis goes from Inferior to Superior (S).
    - "LAS": (radiological convention)
        - The first axis goes from Right to Left (L).
        - The second axis goes from Posterior to Anterior (A).
        - The third axis goes from Inferior to Superior (S).

    See `NiBabel docs about image orientation`_ for more information.

    Args:
        orientation: A three-letter orientation code.
        **kwargs: See :class:`~torchio.transforms.Transform` for additional
            keyword arguments.

    .. _NiBabel docs about image orientation: https://nipy.org/nibabel/image_orientation.html
    """

    def __init__(self, orientation: str = 'RAS', **kwargs):
        super().__init__(**kwargs)
        if not isinstance(orientation, str) or len(orientation) != 3:
            raise ValueError(
                f'Orientation must be a 3-letter string, got {orientation}'
            )

        valid_codes = set('RLAPIS')
        if not all(c in valid_codes for c in orientation):
            raise ValueError(
                f"Orientation code must be composed of characters from 'RLAPIS', got {orientation}"
            )

        # Check for valid axis directions
        if not (
            ('R' in orientation or 'L' in orientation)
            and ('A' in orientation or 'P' in orientation)
            and ('S' in orientation or 'I' in orientation)
        ):
            raise ValueError(
                f'Orientation code must include one character for each axis direction (RL, AP, SI), got {orientation}'
            )

        self.orientation = orientation

    def apply_transform(self, subject: Subject) -> Subject:
        for image in subject.get_images(intensity_only=False):
            affine = image.affine
            current_orientation = ''.join(nib.aff2axcodes(affine))

            if current_orientation == self.orientation:
                continue

            array = image.numpy()[np.newaxis]  # (1, C, W, H, D)
            # NIfTI images should have channels in 5th dimension
            array = array.transpose(2, 3, 4, 0, 1)  # (W, H, D, 1, C)

            # Create a NIfTI image
            nii = nib.Nifti1Image(array, affine)

            # Directly transform from current orientation to target orientation
            current_ornt = nib.orientations.io_orientation(nii.affine)
            target_ornt = nib.orientations.axcodes2ornt(tuple(self.orientation))
            transform = nib.orientations.ornt_transform(current_ornt, target_ornt)
            reoriented_array = nib.orientations.apply_orientation(
                nii.dataobj, transform
            )

            # Calculate the new affine matrix reflecting the reorientation
            reoriented_affine = nii.affine @ nib.orientations.inv_ornt_aff(
                transform, nii.shape
            )

            # Get the reoriented data
            array = reoriented_array.copy()
            array = array.transpose(3, 4, 0, 1, 2)  # (1, C, W, H, D)

            # Update the image data and affine
            image.set_data(torch.as_tensor(array[0]))
            image.affine = reoriented_affine

        return subject
