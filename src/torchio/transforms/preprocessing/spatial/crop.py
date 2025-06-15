from copy import deepcopy

import numpy as np
from nibabel.affines import apply_affine

from ....data.image import Image
from ....data.subject import Subject
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
        copy: If ``True``, each image will be cropped and the patch copied to a new
            subject. If ``False``, each image will be cropped in place. This transform
            overwrites the copy argument of the base transform and copies only the
            cropped patch instead of the whole image. This can provide a significant
            speedup when cropping small patches from large images.
        **kwargs: See :class:`~torchio.transforms.Transform` for additional
            keyword arguments.

    .. seealso:: If you want to pass the output shape instead, please use
        :class:`~torchio.transforms.CropOrPad` instead.
    """

    def __init__(self, cropping: TypeBounds, copy=True, **kwargs):
        self._copy_patch = copy
        # Transform base class deepcopies whole subject by default
        # We want to copy only the cropped patch, so we overwrite the functionality
        super().__init__(cropping, copy=False, **kwargs)
        self.cropping = cropping
        self.args_names = ['cropping']

    def apply_transform(self, subject: Subject) -> Subject:
        assert self.bounds_parameters is not None
        low = self.bounds_parameters[::2]
        high = self.bounds_parameters[1::2]
        index_ini = low
        index_fin = np.array(subject.spatial_shape) - high

        if self._copy_patch:
            # Create a clean new subject to copy the images into
            # We do this __new__ to avoid calling __init__ so we don't have to specify images immediately
            cropped_subject = subject.__class__.__new__(subject.__class__)
            image_keys_to_crop = subject.get_images_dict(
                intensity_only=False,
                include=self.include,
                exclude=self.exclude,
            ).keys()
            keys_to_expose = subject.keys()
            # Copy all attributes we don't want to crop
            # __dict__ returns all attributes, instead of just the images
            for key, value in subject.__dict__.items():
                if key not in image_keys_to_crop:
                    copied_value = deepcopy(value)
                    # Setting __dict__ does not allow key indexing the attribute
                    # so we set it explicitly if we want to expose it
                    if key in keys_to_expose:
                        cropped_subject[key] = copied_value
                    cropped_subject.__dict__[str(key)] = copied_value
                else:
                    # Images are always exposed, so we don't worry about setting __dict__
                    cropped_subject[key] = self._crop_image(
                        value,
                        index_ini,
                        index_fin,
                        copy_patch=self._copy_patch,
                    )

            # Update the __dict__ attribute to include the cropped images
            cropped_subject.update_attributes()
            return cropped_subject
        else:
            # Crop in place
            for image in self.get_images(subject):
                self._crop_image(
                    image,
                    index_ini,
                    index_fin,
                    copy_patch=self._copy_patch,
                )
            return subject

    @staticmethod
    def _crop_image(
        image: Image, index_ini: tuple, index_fin: tuple, *, copy_patch: bool
    ) -> Image:
        new_origin = apply_affine(image.affine, index_ini)
        new_affine = image.affine.copy()
        new_affine[:3, 3] = new_origin
        i0, j0, k0 = index_ini
        i1, j1, k1 = index_fin

        # Crop the image data
        if copy_patch:
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
