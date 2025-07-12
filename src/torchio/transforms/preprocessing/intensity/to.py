from __future__ import annotations

from typing import Any

import torch

from ....data.image import ScalarImage
from ....data.subject import Subject
from ...intensity_transform import IntensityTransform


class To(IntensityTransform):
    """Convert the image tensor data type and/or device.

    This transform is a thin wrapper around :func:`torch.Tensor.to`.

    Args:
        target: First argument to :func:`torch.Tensor.to`.
        to_kwargs: Additional keyword arguments to pass to :func:`torch.Tensor.to`.

    Example:
        >>> import torchio as tio
        >>> ct = tio.datasets.Slicer('CTChest').CT_chest
        >>> clamp = tio.Clamp(out_min=-1000, out_max=1000)
        >>> ct_clamped = clamp(ct)
        >>> rescale = tio.RescaleIntensity(in_min_max=(-1000, 1000), out_min_max=(0, 255))
        >>> ct_rescaled = rescale(ct_clamped)
        >>> to_uint8 = tio.To(torch.uint8)
        >>> ct_uint8 = to_uint8(ct_rescaled)
    """

    def __init__(
        self,
        target: str | torch.dtype | torch.device,
        to_kwargs: dict[str, Any] | None = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.target = target
        if to_kwargs is None:
            to_kwargs = {}
        self.to_kwargs = to_kwargs
        self.args_names = ['target', 'to_kwargs']

    def apply_transform(self, subject: Subject) -> Subject:
        for image in self.get_images(subject):
            assert isinstance(image, ScalarImage)
            image.set_data(image.data.to(self.target, **self.to_kwargs))
        return subject
