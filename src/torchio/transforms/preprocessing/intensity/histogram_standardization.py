from __future__ import annotations

from collections.abc import Iterable
from collections.abc import Sequence
from pathlib import Path
from typing import Callable
from typing import Union

import numpy as np
import torch
from tqdm.auto import tqdm

from ....data.io import read_image
from ....data.subject import Subject
from ....types import TypePath
from .normalization_transform import NormalizationTransform
from .normalization_transform import TypeMaskingMethod

DEFAULT_CUTOFF = 0.01, 0.99
STANDARD_RANGE = 0, 100
TypeLandmarks = Union[TypePath, dict[str, Union[TypePath, np.ndarray]]]


class HistogramStandardization(NormalizationTransform):
    """Perform histogram standardization of intensity values.

    Implementation of `New variants of a method of MRI scale
    standardization <https://ieeexplore.ieee.org/document/836373>`_.

    See example in :func:`torchio.transforms.HistogramStandardization.train`.

    Args:
        landmarks: Dictionary (or path to a PyTorch file with ``.pt`` or ``.pth``
            extension in which a dictionary has been saved) whose keys are
            image names in the subject and values are NumPy arrays or paths to
            NumPy arrays defining the landmarks after training with
            :meth:`torchio.transforms.HistogramStandardization.train`.
        masking_method: See
            :class:`~torchio.transforms.preprocessing.intensity.NormalizationTransform`.
        **kwargs: See :class:`~torchio.transforms.Transform` for additional
            keyword arguments.

    Example:
        >>> import torch
        >>> import torchio as tio
        >>> landmarks = {
        ...     't1': 't1_landmarks.npy',
        ...     't2': 't2_landmarks.npy',
        ... }
        >>> transform = tio.HistogramStandardization(landmarks)
        >>> torch.save(landmarks, 'path_to_landmarks.pth')
        >>> transform = tio.HistogramStandardization('path_to_landmarks.pth')
    """

    def __init__(
        self,
        landmarks: TypeLandmarks,
        masking_method: TypeMaskingMethod = None,
        **kwargs,
    ):
        super().__init__(masking_method=masking_method, **kwargs)
        self.landmarks = landmarks
        self.landmarks_dict = self._parse_landmarks(landmarks)
        self.args_names = ['landmarks', 'masking_method']

    @staticmethod
    def _parse_landmarks(landmarks: TypeLandmarks) -> dict[str, np.ndarray]:
        if isinstance(landmarks, (str, Path)):
            path = Path(landmarks)
            if path.suffix not in ('.pt', '.pth'):
                message = (
                    'The landmarks file must have extension .pt or .pth,'
                    f' not "{path.suffix}"'
                )
                raise ValueError(message)
            landmarks_dict = torch.load(path)
        else:
            landmarks_dict = landmarks
        for key, value in landmarks_dict.items():
            if isinstance(value, (str, Path)):
                landmarks_dict[key] = np.load(value)
        return landmarks_dict

    def apply_normalization(
        self,
        subject: Subject,
        image_name: str,
        mask: torch.Tensor,
    ) -> None:
        if image_name not in self.landmarks_dict:
            keys = tuple(self.landmarks_dict.keys())
            message = (
                f'Image name "{image_name}" should be a key in the'
                f' landmarks dictionary, whose keys are {keys}'
            )
            raise KeyError(message)
        image = subject[image_name]
        landmarks = self.landmarks_dict[image_name]
        normalized = _normalize(image.data, landmarks, mask=mask.numpy())
        image.set_data(normalized)

    @classmethod
    def train(
        cls,
        images_paths: Sequence[TypePath],
        cutoff: tuple[float, float] | None = None,
        mask_path: Sequence[TypePath] | TypePath | None = None,
        masking_function: Callable | None = None,
        output_path: TypePath | None = None,
        *,
        progress: bool = True,
    ) -> np.ndarray:
        """Extract average histogram landmarks from images used for training.

        Args:
            images_paths: List of image paths used to train.
            cutoff: Optional minimum and maximum quantile values,
                respectively, that are used to select a range of intensity of
                interest. Equivalent to :math:`pc_1` and :math:`pc_2` in
                `Nyúl and Udupa's paper <https://pubmed.ncbi.nlm.nih.gov/10571928/>`_.
            mask_path: Path (or list of paths) to a binary image that will be
                used to select the voxels use to compute the stats during
                histogram training. If ``None``, all voxels in the image will
                be used.
            masking_function: Function used to extract voxels used for
                histogram training.
            output_path: Optional file path with extension ``.txt`` or
                ``.npy``, where the landmarks will be saved.

        Example:

            >>> import torch
            >>> import numpy as np
            >>> from pathlib import Path
            >>> from torchio.transforms import HistogramStandardization
            >>>
            >>> t1_paths = ['subject_a_t1.nii', 'subject_b_t1.nii.gz']
            >>> t2_paths = ['subject_a_t2.nii', 'subject_b_t2.nii.gz']
            >>>
            >>> t1_landmarks_path = Path('t1_landmarks.npy')
            >>> t2_landmarks_path = Path('t2_landmarks.npy')
            >>>
            >>> t1_landmarks = (
            ...     t1_landmarks_path
            ...     if t1_landmarks_path.is_file()
            ...     else HistogramStandardization.train(t1_paths)
            ... )
            >>> np.save(t1_landmarks_path, t1_landmarks)
            >>>
            >>> t2_landmarks = (
            ...     t2_landmarks_path
            ...     if t2_landmarks_path.is_file()
            ...     else HistogramStandardization.train(t2_paths)
            ... )
            >>> np.save(t2_landmarks_path, t2_landmarks)
            >>>
            >>> landmarks_dict = {
            ...     't1': t1_landmarks,
            ...     't2': t2_landmarks,
            ... }
            >>>
            >>> transform = HistogramStandardization(landmarks_dict)
        """
        is_masks_list = isinstance(mask_path, Sequence)
        if is_masks_list and len(mask_path) != len(images_paths):  # type: ignore[arg-type]
            message = (
                f'Different number of images ({len(images_paths)})'  # type: ignore[arg-type]
                f' and mask ({len(mask_path)}) paths found'  # type: ignore[arg-type]
            )
            raise ValueError(message)
        quantiles_cutoff = DEFAULT_CUTOFF if cutoff is None else cutoff
        percentiles_cutoff = 100 * np.array(quantiles_cutoff)
        percentiles_database = []
        a, b = percentiles_cutoff  # for mypy
        percentiles = _get_percentiles((a, b))
        iterable: Iterable[TypePath]
        iterable = tqdm(images_paths) if progress else images_paths  # type: ignore[assignment]
        for i, image_file_path in enumerate(iterable):
            tensor, _ = read_image(image_file_path)
            if masking_function is not None:
                mask = masking_function(tensor)
            else:
                if mask_path is None:
                    mask = np.ones_like(tensor, dtype=bool)
                else:
                    if is_masks_list:
                        assert isinstance(mask_path, Sequence)
                        path = mask_path[i]
                    else:
                        path = mask_path  # type: ignore[assignment]
                    mask, _ = read_image(path)
                    mask = mask.numpy() > 0
            array = tensor.numpy()
            percentile_values = np.percentile(array[mask], percentiles)
            percentiles_database.append(percentile_values)
        percentiles_database_array = np.vstack(percentiles_database)
        mapping = _get_average_mapping(percentiles_database_array)

        if output_path is not None:
            output_path = Path(output_path).expanduser()
            extension = output_path.suffix
            if extension == '.txt':
                modality = 'image'
                text = f'{modality} {" ".join(map(str, mapping))}'
                output_path.write_text(text)
            elif extension == '.npy':
                np.save(output_path, mapping)
        return mapping


def _standardize_cutoff(cutoff: Sequence[float]) -> np.ndarray:
    """Standardize the cutoff values given in the configuration.

    Computes percentile landmark normalization by default.
    """
    cutoff_array = np.asarray(cutoff)
    cutoff_array[0] = max(0, cutoff_array[0])
    cutoff_array[1] = min(1, cutoff_array[1])
    cutoff_array[0] = np.min([cutoff_array[0], 0.09])
    cutoff_array[1] = np.max([cutoff_array[1], 0.91])
    return cutoff_array


def _get_average_mapping(percentiles_database: np.ndarray) -> np.ndarray:
    """Map the landmarks of the database to the chosen range.

    Args:
        percentiles_database: Percentiles database over which to perform the
            averaging.
    """
    # Assuming percentiles_database.shape == (num_data_points, num_percentiles)
    pc1 = percentiles_database[:, 0]
    pc2 = percentiles_database[:, -1]
    s1, s2 = STANDARD_RANGE
    slopes = (s2 - s1) / (pc2 - pc1)
    slopes = np.nan_to_num(slopes)
    intercepts = np.mean(s1 - slopes * pc1)
    num_images = len(percentiles_database)
    final_map = slopes.dot(percentiles_database) / num_images + intercepts
    return final_map


def _get_percentiles(percentiles_cutoff: tuple[float, float]) -> np.ndarray:
    quartiles = np.arange(25, 100, 25).tolist()
    deciles = np.arange(10, 100, 10).tolist()
    all_percentiles = list(percentiles_cutoff) + quartiles + deciles
    percentiles = sorted(set(all_percentiles))
    return np.array(percentiles)


def _normalize(
    tensor: torch.Tensor,
    landmarks: np.ndarray,
    mask: np.ndarray | None,
    cutoff: tuple[float, float] | None = None,
    epsilon: float = 1e-5,
) -> torch.Tensor:
    cutoff_ = DEFAULT_CUTOFF if cutoff is None else cutoff
    array = tensor.numpy()
    mapping = landmarks

    data = array
    shape = data.shape
    data = data.reshape(-1).astype(np.float32)

    if mask is None:
        mask = np.ones_like(data, bool)
    mask = mask.reshape(-1)

    range_to_use = [0, 1, 2, 4, 5, 6, 7, 8, 10, 11, 12]

    quantiles_cutoff = _standardize_cutoff(cutoff_)
    percentiles_cutoff = 100 * np.array(quantiles_cutoff)
    a, b = percentiles_cutoff  # for mypy
    percentiles = _get_percentiles((a, b))
    percentile_values = np.percentile(data[mask], percentiles)

    # Apply linear histogram standardization
    range_mapping = mapping[range_to_use]
    range_perc = percentile_values[range_to_use]
    diff_mapping = np.diff(range_mapping)
    diff_perc = np.diff(range_perc)

    # Handling the case where two landmarks are the same
    # for a given input image. This usually happens when
    # image background is not removed from the image.
    diff_perc[diff_perc < epsilon] = np.inf

    affine_map = np.zeros([2, len(range_to_use) - 1])

    # Compute slopes of the linear models
    affine_map[0] = diff_mapping / diff_perc

    # Compute intercepts of the linear models
    affine_map[1] = range_mapping[:-1] - affine_map[0] * range_perc[:-1]

    bin_id = np.digitize(data, range_perc[1:-1], right=False)
    lin_img = affine_map[0, bin_id]
    aff_img = affine_map[1, bin_id]
    new_img = lin_img * data + aff_img
    new_img = new_img.reshape(shape)
    new_img = new_img.astype(np.float32)
    new_img = torch.as_tensor(new_img)
    return new_img


# train_histogram kept for backward compatibility
train = train_histogram = HistogramStandardization.train
