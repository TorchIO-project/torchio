from __future__ import annotations

import numpy as np

from ....data.subject import Subject
from ....types import TypeTripletInt
from ....utils import to_tuple
from ...spatial_transform import SpatialTransform
from .crop_or_pad import CropOrPad


class EnsureShapeMultiple(SpatialTransform):
    """Ensure that all values in the image shape are divisible by $n$.

    Some convolutional neural network architectures need that the size of the
    input across all spatial dimensions is a power of $2$.

    For example, the canonical 3D U-Net from
    [Çiçek et al. ](https://link.springer.com/chapter/10.1007/978-3-319-46723-8_49)
    includes three downsampling (pooling) and upsampling operations:

    ![3D U-Net](https://www.researchgate.net/profile/Olaf-Ronneberger/publication/304226155/figure/fig1/AS:375619658502144@1466566113191/The-3D-u-net-architecture-Blue-boxes-represent-feature-maps-The-number-of-channels-is.png)

    Pooling operations in PyTorch round down the output size:

        >>> import torch
        >>> x = torch.rand(3, 10, 20, 31)
        >>> x_down = torch.nn.functional.max_pool3d(x, 2)
        >>> x_down.shape
        torch.Size([3, 5, 10, 15])

    If we upsample this tensor, the original shape is lost:

        >>> x_down_up = torch.nn.functional.interpolate(x_down, scale_factor=2)
        >>> x_down_up.shape
        torch.Size([3, 10, 20, 30])
        >>> x.shape
        torch.Size([3, 10, 20, 31])

    If we try to concatenate `x_down` and `x_down_up` (to create skip
    connections), we will get an error. It is therefore good practice to ensure
    that the size of our images is such that concatenations will be safe.

    Note:
        In these examples, it's assumed that all convolutions in the
        U-Net use padding so that the output size is the same as the input
        size.

    The image above shows $3$ downsampling operations, so the input size
    along all dimensions should be a multiple of $2^3 = 8$.

    Example (assuming `pip install unet` has been run before):

        >>> import torchio as tio
        >>> import unet
        >>> net = unet.UNet3D(padding=1)
        >>> t1 = tio.datasets.Colin27().t1
        >>> tensor_bad = t1.data.unsqueeze(0)
        >>> tensor_bad.shape
        torch.Size([1, 1, 181, 217, 181])
        >>> net(tensor_bad).shape
        Traceback (most recent call last):
          File "<stdin>", line 1, in <module>
          File "/home/fernando/miniconda3/envs/resseg/lib/python3.7/site-packages/torch/nn/modules/module.py", line 727, in _call_impl
            result = self.forward(*input, **kwargs)
          File "/home/fernando/miniconda3/envs/resseg/lib/python3.7/site-packages/unet/unet.py", line 122, in forward
            x = self.decoder(skip_connections, encoding)
          File "/home/fernando/miniconda3/envs/resseg/lib/python3.7/site-packages/torch/nn/modules/module.py", line 727, in _call_impl
            result = self.forward(*input, **kwargs)
          File "/home/fernando/miniconda3/envs/resseg/lib/python3.7/site-packages/unet/decoding.py", line 61, in forward
            x = decoding_block(skip_connection, x)
          File "/home/fernando/miniconda3/envs/resseg/lib/python3.7/site-packages/torch/nn/modules/module.py", line 727, in _call_impl
            result = self.forward(*input, **kwargs)
          File "/home/fernando/miniconda3/envs/resseg/lib/python3.7/site-packages/unet/decoding.py", line 131, in forward
            x = torch.cat((skip_connection, x), dim=CHANNELS_DIMENSION)
        RuntimeError: Sizes of tensors must match except in dimension 1. Got 45 and 44 in dimension 2 (The offending index is 1)
        >>> num_poolings = 3
        >>> fix_shape_unet = tio.EnsureShapeMultiple(2**num_poolings)
        >>> t1_fixed = fix_shape_unet(t1)
        >>> tensor_ok = t1_fixed.data.unsqueeze(0)
        >>> tensor_ok.shape
        torch.Size([1, 1, 184, 224, 184])  # as expected

    Args:
        target_multiple: Tuple $(n_w, n_h, n_d)$, so that the size of the
            output along axis $i$ is a multiple of $n_i$. If a
            single value $n$ is provided, then
            $n_w = n_h = n_d = n$.
        method: Either `'crop'` or `'pad'`.
        **kwargs: See [`Transform`][torchio.transforms.Transform] for additional
            keyword arguments.

    Examples:
        >>> import torchio as tio
        >>> image = tio.datasets.Colin27().t1
        >>> image.shape
        (1, 181, 217, 181)
        >>> transform = tio.EnsureShapeMultiple(8, method='pad')
        >>> transformed = transform(image)
        >>> transformed.shape
        (1, 184, 224, 184)
        >>> transform = tio.EnsureShapeMultiple(8, method='crop')
        >>> transformed = transform(image)
        >>> transformed.shape
        (1, 176, 216, 176)
        >>> image_2d = image.data[..., :1]
        >>> image_2d.shape
        torch.Size([1, 181, 217, 1])
        >>> transformed = transform(image_2d)
        >>> transformed.shape
        torch.Size([1, 176, 216, 1])
    """

    def __init__(
        self,
        target_multiple: int | TypeTripletInt,
        *,
        method: str = 'pad',
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.target_multiple = np.array(to_tuple(target_multiple, 3))
        if method not in ('crop', 'pad'):
            raise ValueError('Method must be "crop" or "pad"')
        self.method = method

    def apply_transform(self, subject: Subject) -> Subject:
        source_shape = np.array(subject.spatial_shape, np.uint16)
        if self.method == 'crop':
            integer_ratio = np.floor(source_shape / self.target_multiple)
        else:
            integer_ratio = np.ceil(source_shape / self.target_multiple)
        target_shape = integer_ratio * self.target_multiple
        target_shape = np.maximum(target_shape, 1)
        target_shape_values = [
            int(value) for value in target_shape.astype(int).tolist()
        ]
        transform = CropOrPad(
            (
                target_shape_values[0],
                target_shape_values[1],
                target_shape_values[2],
            ),
            copy=self.copy,
            include=self.include,
            exclude=self.exclude,
            keep=self.keep,
            parse_input=self.parse_input,
            label_keys=self.label_keys,
        )
        transformed = transform(subject)
        assert isinstance(transformed, Subject)
        return transformed
