from collections.abc import Sequence

import numpy as np
import torch

from ....data.subject import Subject
from ....types import TypeRangeFloat
from ....utils import to_tuple
from ...intensity_transform import IntensityTransform
from .. import RandomTransform


class RandomGamma(RandomTransform, IntensityTransform):
    r"""Randomly change contrast of an image by raising its values to the power
    $\gamma$.

    Args:
        log_gamma: Tuple $(a, b)$ to compute the exponent
            $\gamma = e ^ \beta$,
            where $\beta \sim \mathcal{U}(a, b)$.
            If a single value $d$ is provided, then
            $\beta \sim \mathcal{U}(-d, d)$.
            Negative and positive values for this argument perform gamma
            compression and expansion, respectively.
            See the [Gamma correction](https://en.wikipedia.org/wiki/Gamma_correction) Wikipedia entry for more information.
        **kwargs: See [`Transform`][torchio.transforms.Transform] for additional
            keyword arguments.


    Note:
        Fractional exponentiation of negative values is generally not
        well-defined for non-complex numbers.
        If negative values are found in the input image $I$,
        the applied transform is $\text{sign}(I) |I|^\gamma$,
        instead of the usual $I^\gamma$. The
        [`RescaleIntensity`][torchio.transforms.RescaleIntensity]
        transform may be used to ensure that all values are positive. This is
        generally not problematic, but it is recommended to visualize results
        on images with negative values. More information can be found on
        [this StackExchange question](https://math.stackexchange.com/questions/317528/how-do-you-compute-negative-numbers-to-fractional-powers).



    Examples:
        >>> import torchio as tio
        >>> subject = tio.datasets.FPG()
        >>> transform = tio.RandomGamma(log_gamma=(-0.3, 0.3))  # gamma between 0.74 and 1.34
        >>> transformed = transform(subject)

    """

    def __init__(self, log_gamma: TypeRangeFloat = (-0.3, 0.3), **kwargs):
        super().__init__(**kwargs)
        self.log_gamma_range = self._parse_range(log_gamma, 'log_gamma')

    def apply_transform(self, subject: Subject) -> Subject:
        images_dict = self.get_images_dict(subject)
        if not images_dict:
            return subject

        gamma_by_name: dict[str, float | Sequence[float]] = {}
        for name, image in images_dict.items():
            gammas = [self.get_params(self.log_gamma_range) for _ in image.data]
            gamma_by_name[name] = gammas
        transform = Gamma(gamma=gamma_by_name, **self.get_base_args())
        transformed = transform(subject)
        assert isinstance(transformed, Subject)
        return transformed

    def get_params(self, log_gamma_range: tuple[float, float]) -> float:
        gamma = np.exp(self.sample_uniform(*log_gamma_range))
        return gamma


class Gamma(IntensityTransform):
    r"""Change contrast of an image by raising its values to the power
    $\gamma$.

    Args:
        gamma: Exponent to which values in the image will be raised.
            Negative and positive values for this argument perform gamma
            compression and expansion, respectively.
            See the [Gamma correction](https://en.wikipedia.org/wiki/Gamma_correction) Wikipedia entry for more information.
        **kwargs: See [`Transform`][torchio.transforms.Transform] for additional
            keyword arguments.


    Note:
        Fractional exponentiation of negative values is generally not
        well-defined for non-complex numbers.
        If negative values are found in the input image $I$,
        the applied transform is $\text{sign}(I) |I|^\gamma$,
        instead of the usual $I^\gamma$. The
        [`RescaleIntensity`][torchio.transforms.preprocessing.intensity.rescale.RescaleIntensity]
        transform may be used to ensure that all values are positive. This is
        generally not problematic, but it is recommended to visualize results
        on image with negative values. More information can be found on
        [this StackExchange question](https://math.stackexchange.com/questions/317528/how-do-you-compute-negative-numbers-to-fractional-powers).


    Examples:
        >>> import torchio as tio
        >>> subject = tio.datasets.FPG()
        >>> transform = tio.Gamma(0.8)
        >>> transformed = transform(subject)
    """

    def __init__(
        self,
        gamma: float | Sequence[float] | dict[str, float | Sequence[float]],
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.gamma = gamma
        self.args_names = ['gamma']
        self.invert_transform = False

    def apply_transform(self, subject: Subject) -> Subject:
        for name, image in self.get_images_dict(subject).items():
            gamma = self.get_parameter(self.gamma, name)
            gammas = to_tuple(gamma, length=len(image.data))
            transformed_tensors = []
            image.set_data(image.data.float())
            for gamma, tensor in zip(gammas, image.data, strict=True):
                if self.invert_transform:
                    correction = power(tensor, 1 - gamma)
                    transformed_tensor = tensor * correction
                else:
                    transformed_tensor = power(tensor, gamma)
                transformed_tensors.append(transformed_tensor)
            image.set_data(torch.stack(transformed_tensors))
        return subject


def power(tensor, gamma):
    if tensor.min() < 0:
        output = tensor.sign() * tensor.abs() ** gamma
    else:
        output = tensor**gamma
    return output
