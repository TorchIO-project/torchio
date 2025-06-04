import copy
import nibabel as nib
import numpy as np

from ....data.subject import Subject
from ....data.image import Image
from .bounds_transform import BoundsTransform
from .bounds_transform import TypeBounds


class Crop(BoundsTransform):
    r"""Crop an image.

    Args:
        cropping: Tuple
            :math:`(w_{ini}, w_{fin}, h_{ini}, h_{fin}, d_{ini}, d_{fin})`
            defining the number of values cropped from the edges of each axis.
            If the initial shape of the image is
            :math:`W \times H \times D`, the final shape will be
            :math:`(- w_{ini} + W - w_{fin}) \times (- h_{ini} + H - h_{fin})
            \times (- d_{ini} + D - d_{fin})`.
            If only three values :math:`(w, h, d)` are provided, then
            :math:`w_{ini} = w_{fin} = w`,
            :math:`h_{ini} = h_{fin} = h` and
            :math:`d_{ini} = d_{fin} = d`.
            If only one value :math:`n` is provided, then
            :math:`w_{ini} = w_{fin} = h_{ini} = h_{fin}
            = d_{ini} = d_{fin} = n`.
        copy: bool, optional
            This transform overwrites the copy argument of the base transform and 
            copies only the cropped patch, instead of the whole image.
            If ``True``, the cropped image will be copied to a new subject.
            If ``False``, the patch will be cropped in place. Default: ``True``.
        **kwargs: See :class:`~torchio.transforms.Transform` for additional
            keyword arguments.

    .. seealso:: If you want to pass the output shape instead, please use
        :class:`~torchio.transforms.CropOrPad` instead.
    """

    def __init__(self, cropping: TypeBounds, copy=True, **kwargs):
        self.copy_patch = copy
        super().__init__(cropping, copy=False, **kwargs)
        self.cropping = cropping
        self.args_names = ['cropping']

    def apply_transform(self, sample: Subject) -> Subject:
        assert self.bounds_parameters is not None
        low = self.bounds_parameters[::2]
        high = self.bounds_parameters[1::2]
        index_ini = low
        index_fin = np.array(sample.spatial_shape) - high
        
        if self.copy_patch:
            # Create a new subject with only the cropped patch
            sample_attributes = {}
            
            # Copy all non-image attributes
            for key, value in sample.items():
                if key not in sample.get_images_dict(intensity_only=False, include=self.include, exclude=self.exclude).keys():
                    sample_attributes[key] = copy.deepcopy(value)
                else:
                    sample_attributes[key] = self.crop_image(value, index_ini, index_fin)
            cropped_sample = type(sample)(**sample_attributes)
            
            # Copy applied transforms history
            cropped_sample.applied_transforms = copy.deepcopy(sample.applied_transforms)
            
            cropped_sample.update_attributes()
            return cropped_sample
        else:
            # Crop in place
            for image in self.get_images(sample):
                self.crop_image(image, index_ini, index_fin)
            return sample
    
    def crop_image(self, image: Image, index_ini: tuple, index_fin: tuple) -> None:
        new_origin = nib.affines.apply_affine(image.affine, index_ini)
        new_affine = image.affine.copy()
        new_affine[:3, 3] = new_origin
        i0, j0, k0 = index_ini
        i1, j1, k1 = index_fin
        
        # Crop the image data
        if self.copy_patch:
            # Create a new image with the cropped data
            cropped_data = image.data[:, i0:i1, j0:j1, k0:k1].clone()
            new_image = type(image)(
                tensor=cropped_data,
                affine=new_affine,
                type=image.type,
                path=image.path,
            )
            return new_image
        else:
            image.set_data(image.data[:, i0:i1, j0:j1, k0:k1].clone())
            image.affine = new_affine
            return image

    def inverse(self):
        from .pad import Pad

        return Pad(self.cropping)
