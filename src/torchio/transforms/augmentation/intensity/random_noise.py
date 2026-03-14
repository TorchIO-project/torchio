import torch

from ....data.subject import Subject
from ...intensity_transform import IntensityTransform
from .. import RandomTransform


class RandomNoise(RandomTransform, IntensityTransform):
    r"""Add Gaussian noise with random parameters.

    Add noise sampled from a normal distribution with random parameters.

    Args:
        mean: Mean $\mu$ of the Gaussian distribution
            from which the noise is sampled.
            If two values $(a, b)$ are provided,
            then $\mu \sim \mathcal{U}(a, b)$.
            If only one value $d$ is provided,
            $\mu \sim \mathcal{U}(-d, d)$.
        std: Standard deviation $\sigma$ of the Gaussian distribution
            from which the noise is sampled.
            If two values $(a, b)$ are provided,
            then $\sigma \sim \mathcal{U}(a, b)$.
            If only one value $d$ is provided,
            $\sigma \sim \mathcal{U}(0, d)$.
        **kwargs: See [`Transform`][torchio.transforms.Transform] for additional
            keyword arguments.
    """

    def __init__(
        self,
        mean: float | tuple[float, float] = 0,
        std: float | tuple[float, float] = (0, 0.25),
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.mean_range = self._parse_range(mean, 'mean')
        self.std_range = self._parse_range(std, 'std', min_constraint=0)

    def apply_transform(self, subject: Subject) -> Subject:
        images_dict = self.get_images_dict(subject)
        if not images_dict:
            return subject

        means_by_name: dict[str, float] = {}
        stds_by_name: dict[str, float] = {}
        seeds_by_name: dict[str, int] = {}
        for image_name in images_dict:
            mean, std, seed = self.get_params(self.mean_range, self.std_range)
            means_by_name[image_name] = mean
            stds_by_name[image_name] = std
            seeds_by_name[image_name] = seed
        transform = Noise(
            mean=means_by_name,
            std=stds_by_name,
            seed=seeds_by_name,
            **self._get_base_args(),
        )
        transformed = transform(subject)
        assert isinstance(transformed, Subject)
        return transformed

    def get_params(
        self,
        mean_range: tuple[float, float],
        std_range: tuple[float, float],
    ) -> tuple[float, float, int]:
        mean = self.sample_uniform(*mean_range)
        std = self.sample_uniform(*std_range)
        seed = self._get_random_seed()
        return mean, std, seed


class Noise(IntensityTransform):
    r"""Add Gaussian noise.

    Add noise sampled from a normal distribution.

    Args:
        mean: Mean $\mu$ of the Gaussian distribution
            from which the noise is sampled.
        std: Standard deviation $\sigma$ of the Gaussian distribution
            from which the noise is sampled.
        seed: Seed for the random number generator.
        **kwargs: See [`Transform`][torchio.transforms.Transform] for additional
            keyword arguments.
    """

    def __init__(
        self,
        mean: float | dict[str, float],
        std: float | dict[str, float],
        seed: int | dict[str, int],
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.noise_mean: float | dict[str, float] = mean
        self.noise_std: float | dict[str, float] = std
        self.seed: int | dict[str, int] = seed
        self.invert_transform = False
        self.args_names = ['mean', 'std', 'seed']

    def apply_transform(self, subject: Subject) -> Subject:
        for name, image in self.get_images_dict(subject).items():
            mean = self.get_parameter(self.noise_mean, name)
            std = self.get_parameter(self.noise_std, name)
            seed = self.get_parameter(self.seed, name)
            with self._use_seed(seed):
                noise = get_noise(image.data, mean, std)
            if self.invert_transform:
                noise *= -1
            image.set_data(image.data + noise)
        return subject

    def _get_named_arguments(self) -> dict[str, object]:
        return {
            'mean': self.noise_mean,
            'std': self.noise_std,
            'seed': self.seed,
        }


def get_noise(tensor: torch.Tensor, mean: float, std: float) -> torch.Tensor:
    return torch.randn(*tensor.shape) * std + mean
