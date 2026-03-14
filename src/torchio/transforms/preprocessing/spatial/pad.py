import warnings
from numbers import Number
from typing import Literal
from typing import cast

import numpy as np
import torch
from nibabel.affines import apply_affine

from ....data.image import Image
from ....data.subject import Subject
from .bounds_transform import BoundsTransform
from .bounds_transform import TypeBounds

NumpyPadMode = Literal[
    'empty',
    'edge',
    'wrap',
    'constant',
    'linear_ramp',
    'maximum',
    'mean',
    'median',
    'minimum',
    'reflect',
    'symmetric',
]


class Pad(BoundsTransform):
    r"""Pad an image.

    Args:
        padding: Tuple
            $(w_{ini}, w_{fin}, h_{ini}, h_{fin}, d_{ini}, d_{fin})$
            defining the number of values padded to the edges of each axis.
            If the initial shape of the image is
            $W \times H \times D$, the final shape will be
            $(w_{ini} + W + w_{fin}) \times (h_{ini} + H + h_{fin})
            \times (d_{ini} + D + d_{fin})$.
            If only three values $(w, h, d)$ are provided, then
            $w_{ini} = w_{fin} = w$,
            $h_{ini} = h_{fin} = h$ and
            $d_{ini} = d_{fin} = d$.
            If only one value $n$ is provided, then
            $w_{ini} = w_{fin} = h_{ini} = h_{fin} =
            d_{ini} = d_{fin} = n$.
        padding_mode: See possible modes in [NumPy docs](https://numpy.org/doc/stable/reference/generated/numpy.pad.html). If it is a number,
            the mode will be set to `'constant'`. If it is `'mean'`,
            `'maximum'`, `'median'` or `'minimum'`, the statistic will be
            computed from the whole volume, unlike in NumPy, which computes it
            along the padded axis.
        **kwargs: See [`Transform`][torchio.transforms.Transform] for additional
            keyword arguments.

    See also: If you want to pass the output shape instead, please use
        [`CropOrPad`][torchio.transforms.CropOrPad] instead.

    """

    PADDING_MODES = (
        'empty',
        'edge',
        'wrap',
        'constant',
        'linear_ramp',
        'maximum',
        'mean',
        'median',
        'minimum',
        'reflect',
        'symmetric',
    )

    def __init__(
        self,
        padding: TypeBounds,
        padding_mode: str | float = 0,
        **kwargs,
    ):
        super().__init__(padding, **kwargs)
        self.padding = padding
        self.check_padding_mode(padding_mode)
        self.padding_mode = padding_mode
        self.args_names = ['padding', 'padding_mode']

    @classmethod
    def check_padding_mode(cls, padding_mode):
        is_number = isinstance(padding_mode, Number)
        is_callable = callable(padding_mode)
        if not (padding_mode in cls.PADDING_MODES or is_number or is_callable):
            message = (
                f'Padding mode "{padding_mode}" not valid. Valid options are'
                f' {list(cls.PADDING_MODES)}, a number or a function'
            )
            raise KeyError(message)

    def _check_truncation(self, image: Image, mode: str | float) -> None:
        if mode not in ('mean', 'median'):
            return
        if torch.is_floating_point(image.data):
            return
        message = (
            f'The constant value computed for padding mode "{mode}" might be truncated '
            ' in the output, as the data type of the input image is not float.'
            ' Consider converting the image to a floating point type'
            ' before applying this transform.'
        )
        warnings.warn(message, RuntimeWarning, stacklevel=2)

    def apply_transform(self, subject: Subject) -> Subject:
        assert self.bounds_parameters is not None
        low = self.bounds_parameters[::2]
        images_dict = subject.get_images_dict(
            intensity_only=False,
            include=self.include,
            exclude=self.exclude,
        )
        for image_name, image in images_dict.items():
            self._check_truncation(image, self.padding_mode)
            new_origin = apply_affine(image.affine, -np.array(low))
            new_affine = image.affine.copy()
            new_affine[:3, 3] = new_origin

            mode: NumpyPadMode = 'constant'
            constant: int | float | None = None
            if isinstance(self.padding_mode, Number):
                constant = float(self.padding_mode)
            elif self.padding_mode == 'maximum':
                constant = image.data.max().item()
            elif self.padding_mode == 'mean':
                constant = image.data.float().mean().item()
            elif self.padding_mode == 'median':
                constant = torch.quantile(image.data.float(), 0.5).item()
            elif self.padding_mode == 'minimum':
                constant = image.data.min().item()
            else:
                constant = None
                mode = cast(NumpyPadMode, self.padding_mode)

            pad_params = self.bounds_parameters
            paddings = (0, 0), pad_params[:2], pad_params[2:4], pad_params[4:]
            if constant is not None:
                padded = np.pad(
                    image.data.numpy(),
                    paddings,
                    mode='constant',
                    constant_values=constant,
                )
            else:
                padded = np.pad(image.data.numpy(), paddings, mode=mode)
            new_image = image.new_like(
                tensor=torch.as_tensor(padded),
                affine=new_affine,
            )
            # Replace the image in the subject with the new padded image
            subject[image_name] = new_image
        # Update attributes to sync dictionary changes with attribute access
        subject.update_attributes()
        return subject

    def inverse(self):
        from .crop import Crop

        return Crop(self.padding)
