import numpy as np

from ..data import ScalarImage
from ..data.subject import Subject


class ZonePlate(Subject):
    """Synthetic data generated from a zone plate.

    The zone plate is a circular diffraction grating that produces concentric
    rings of light and dark bands. This dataset is useful for testing image
    processing algorithms, particularly those related to frequency analysis and
    interpolation.

    See equation 10.63 in `Practical Handbook on Image Processing for
    Scientific Applications <https://www.routledge.com/Practical-Handbook-on-Image-Processing-for-Scientific-and-Technical-Applications/Jahne/p/book/9780849319006?srsltid=AfmBOoptrtzILIlMx9FYqvx6UrGbevfD66x2k242iprFdn_CfyOWXjjH>`_
    by Bernd Jähne.

    Args:
        size: The size of the generated image along all dimensions.
    """

    def __init__(self, size: int = 501):
        if size < 3:
            raise ValueError('Size must be at least 3.')
        self.size = size
        image = self._generate_image(size)
        super().__init__(image=image)

    @staticmethod
    def _generate_image(size: int) -> ScalarImage:
        if size % 2 == 1:
            fin = (size - 1) // 2
            ini = -fin
        else:
            fin = size // 2
            ini = -fin + 1
        x = np.arange(ini, fin)
        y = np.arange(ini, fin)
        z = np.arange(ini, fin)
        X, Y, Z = np.meshgrid(x, y, z)
        r = np.sqrt(X**2 + Y**2 + Z**2)
        km = 0.8 * np.pi
        rm = ini
        w = rm / 10
        term1 = np.sin((km * r**2) / (2 * rm))
        term2 = 0.5 * np.tanh((rm - r) / w) + 0.5
        g = term1 * term2
        affine = np.eye(4)
        origin = np.array([ini, ini, ini])
        affine[:3, 3] = origin
        return ScalarImage(tensor=g[np.newaxis], affine=affine)
