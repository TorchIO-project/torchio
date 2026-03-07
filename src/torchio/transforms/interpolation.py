import enum

import SimpleITK as sitk


class Interpolation(enum.Enum):
    """Interpolation techniques available in ITK.

    For a full quantitative comparison of interpolation methods, you can read
    [Meijering et al. 1999, Quantitative Comparison of Sinc-Approximating Kernels for
    Medical Image Interpolation ](https://link.springer.com/chapter/10.1007/10704282_23).

    Examples:
        >>> import torchio as tio
        >>> transform = tio.RandomAffine(image_interpolation='bspline')
    """

    NEAREST = 'sitkNearestNeighbor'
    """Nearest neighbor interpolation."""

    LINEAR = 'sitkLinear'
    """Linear interpolation."""

    BSPLINE = 'sitkBSpline'
    """B-Spline of order 3 (cubic) interpolation."""

    CUBIC = 'sitkBSpline'
    """Same as `BSPLINE`."""

    GAUSSIAN = 'sitkGaussian'
    """Gaussian interpolation. Sigma is set to 0.8 input pixels and alpha is 4."""

    LABEL_GAUSSIAN = 'sitkLabelGaussian'
    """Smoothly interpolate multi-label images. Sigma is set to 1 input pixel and alpha is 1."""

    HAMMING = 'sitkHammingWindowedSinc'
    """Hamming windowed sinc kernel."""

    COSINE = 'sitkCosineWindowedSinc'
    """Cosine windowed sinc kernel."""

    WELCH = 'sitkWelchWindowedSinc'
    """Welch windowed sinc kernel."""

    LANCZOS = 'sitkLanczosWindowedSinc'
    """Lanczos windowed sinc kernel."""

    BLACKMAN = 'sitkBlackmanWindowedSinc'
    """Blackman windowed sinc kernel."""


def get_sitk_interpolator(interpolation: str) -> int:
    if not isinstance(interpolation, str):
        message = (
            f'Interpolation must be a string, not "{interpolation}"'
            f' of type {type(interpolation)}'
        )
        raise ValueError(message)
    string = getattr(Interpolation, interpolation.upper()).value
    return getattr(sitk, string)
