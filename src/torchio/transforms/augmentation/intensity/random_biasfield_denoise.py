import torch
from ...transform import Transform
from torchio.data.subject import Subject


class RandomBiasFieldDenoise(Transform):
    """
    Simple placeholder transform that simulates denoising after bias field
    correction by blending voxel intensities toward the mean value.

    Parameters:
        noise_reduction_factor (float): Strength of denoising (0-1).
    """

    def __init__(self, noise_reduction_factor: float = 0.1, **kwargs):
        super().__init__(**kwargs)
        self.noise_reduction_factor = noise_reduction_factor

    def apply_transform(self, subject: Subject) -> Subject:
        for _, image in subject.get_images_dict(intensity_only=True).items():
            tensor = image.data.float()

            # Basic denoising by shifting toward mean intensity
            mean_val = tensor.mean()
            tensor = (tensor * (1 - self.noise_reduction_factor)) + (
                mean_val * self.noise_reduction_factor
            )

            image.set_data(tensor)

        return subject

    def __repr__(self):
        return f'{self.__class__.__name__}(noise_reduction_factor={self.noise_reduction_factor})'
