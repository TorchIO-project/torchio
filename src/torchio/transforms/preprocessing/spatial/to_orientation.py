import nibabel as nib
import torch
from einops import rearrange
from nibabel import orientations

from ....data.subject import Subject
from ...spatial_transform import SpatialTransform


class ToOrientation(SpatialTransform):
    """Reorient the data to a specified orientation.

    This transform reorders the voxels and modifies the affine matrix to match
    the specified orientation code.
    The image intensity values are not modified, and the sample locations in
    the scanner space are preserved.

    Common orientation codes include:

    - ``'RAS'`` (neurological convention):
        - The first axis goes from Left to Right (R).
        - The second axis goes from Posterior to Anterior (A).
        - The third axis goes from Inferior to Superior (S).
    - ``'LAS'`` (radiological convention):
        - The first axis goes from Right to Left (L).
        - The second axis goes from Posterior to Anterior (A).
        - The third axis goes from Inferior to Superior (S).

    See `NiBabel docs about image orientation`_ for more information.

    Args:
        orientation: A three-letter orientation code. Examples: ``'RAS'``,
            ``'LAS'``, ``'LPS'``, ``'PLS'``, ``'SLP'``. The code must contain
            one character for each axis direction: R or L, A or P, and S or I.
        **kwargs: See :class:`~torchio.transforms.Transform` for additional
            keyword arguments.

    .. _NiBabel docs about image orientation: https://nipy.org/nibabel/image_orientation.html
    """

    def __init__(self, orientation: str = 'RAS', **kwargs):
        super().__init__(**kwargs)
        if not isinstance(orientation, str) or len(orientation) != 3:
            message = f'Orientation must be a 3-letter string, got "{orientation}"'
            raise ValueError(message)

        valid_codes = set('RLAPIS')
        orientation = orientation.upper()
        all_valid = all(axis in valid_codes for axis in orientation)
        if not all_valid:
            message = (
                'Orientation code must be composed of three distinct characters'
                f' in {valid_codes} but got "{orientation}"'
            )
            raise ValueError(message)

        # Check for valid axis directions
        has_sagittal = 'R' in orientation or 'L' in orientation
        has_coronal = 'A' in orientation or 'P' in orientation
        has_axial = 'S' in orientation or 'I' in orientation
        has_all = has_sagittal and has_coronal and has_axial
        if not has_all:
            message = (
                'Orientation code must include one character for each axis direction:'
                f' R or L, A or P, and S or I, but got "{orientation}"'
            )
            raise ValueError(message)

        self.orientation = orientation

    def apply_transform(self, subject: Subject) -> Subject:
        for image in subject.get_images(intensity_only=False):
            current_orientation = ''.join(nib.aff2axcodes(image.affine))

            # If the image is already in the target orientation, skip it
            if current_orientation == self.orientation:
                continue

            # NIfTI images should have channels in 5th dimension
            array = rearrange(image.numpy(), 'C W H D -> W H D 1 C')

            nii = nib.Nifti1Image(array, image.affine)

            # Compute transform from current orientation to target orientation
            current_orientation = orientations.io_orientation(nii.affine)
            target_orientation = orientations.axcodes2ornt(tuple(self.orientation))
            transform = orientations.ornt_transform(
                current_orientation,
                target_orientation,
            )

            # Reorder voxels
            reoriented_array = orientations.apply_orientation(nii.dataobj, transform)
            reoriented_array = rearrange(reoriented_array, 'W H D 1 C -> C W H D')

            # Calculate the new affine matrix reflecting the reorientation
            reoriented_affine = nii.affine @ orientations.inv_ornt_aff(
                transform, nii.shape
            )

            # Update the image data and affine
            image.set_data(torch.from_numpy(reoriented_array.copy()))
            image.affine = reoriented_affine

        return subject
