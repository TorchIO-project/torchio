import warnings
from numbers import Number
from typing import Union

import numpy as np
import torch
from nibabel.affines import apply_affine

from ....data.image import Image
from ....data.subject import Subject
from .bounds_transform import BoundsTransform
from .bounds_transform import TypeBounds


class Pad(BoundsTransform):
    r"""Pad an image.

    Args:
        padding: Tuple
            :math:`(w_{ini}, w_{fin}, h_{ini}, h_{fin}, d_{ini}, d_{fin})`
            defining the number of values padded to the edges of each axis.
            If the initial shape of the image is
            :math:`W \times H \times D`, the final shape will be
            :math:`(w_{ini} + W + w_{fin}) \times (h_{ini} + H + h_{fin})
            \times (d_{ini} + D + d_{fin})`.
            If only three values :math:`(w, h, d)` are provided, then
            :math:`w_{ini} = w_{fin} = w`,
            :math:`h_{ini} = h_{fin} = h` and
            :math:`d_{ini} = d_{fin} = d`.
            If only one value :math:`n` is provided, then
            :math:`w_{ini} = w_{fin} = h_{ini} = h_{fin} =
            d_{ini} = d_{fin} = n`.
        padding_mode: See possible modes in `NumPy docs`_. If it is a number,
            the mode will be set to ``'constant'``. If it is ``'mean'``,
            ``'maximum'``, ``'median'`` or ``'minimum'``, the statistic will be
            computed from the whole volume, unlike in NumPy, which computes it
            along the padded axis.
        **kwargs: See :class:`~torchio.transforms.Transform` for additional
            keyword arguments.

    .. seealso:: If you want to pass the output shape instead, please use
        :class:`~torchio.transforms.CropOrPad` instead.

    .. _NumPy docs: https://numpy.org/doc/stable/reference/generated/numpy.pad.html
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
        padding_mode: Union[str, float] = 0,
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

    def _check_truncation(self, image: Image, mode: Union[str, float]) -> None:
        if mode not in ('mean', 'median'):
            return
        if torch.is_floating_point(image.data):
            return
        message = (
            f'The constant value computed for padding mode "{mode}" might '
            ' be truncated in the output, as the input image is not'
            'floating point. Consider converting the image to a floating'
            ' point type before applying this transform.'
        )
        warnings.warn(message, RuntimeWarning, stacklevel=2)

    def apply_transform(self, subject: Subject) -> Subject:
        assert self.bounds_parameters is not None
        low = self.bounds_parameters[::2]
        for image in self.get_images(subject):
            self._check_truncation(image, self.padding_mode)
            new_origin = apply_affine(image.affine, -np.array(low))
            new_affine = image.affine.copy()
            new_affine[:3, 3] = new_origin

            mode: str | float = 'constant'
            constant: torch.Tensor | float | None = None
            kwargs: dict[str, str | float | torch.Tensor] = {}
            if isinstance(self.padding_mode, Number):
                constant = self.padding_mode  # type: ignore[assignment]
            elif self.padding_mode == 'maximum':
                constant = image.data.max()
            elif self.padding_mode == 'mean':
                constant = image.data.float().mean()
            elif self.padding_mode == 'median':
                constant = torch.quantile(image.data.float(), 0.5)
            elif self.padding_mode == 'minimum':
                constant = image.data.min()
            else:
                constant = None
                mode = self.padding_mode

            if constant is not None:
                kwargs['constant_values'] = constant
            kwargs['mode'] = mode

            pad_params = self.bounds_parameters
            paddings = (0, 0), pad_params[:2], pad_params[2:4], pad_params[4:]
            padded = np.pad(image.data, paddings, **kwargs)  # type: ignore[call-overload]
            image.set_data(torch.as_tensor(padded))
            image.affine = new_affine
        return subject

    def inverse(self):
        from .crop import Crop

        return Crop(self.padding)
