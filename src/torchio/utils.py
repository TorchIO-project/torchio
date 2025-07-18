from __future__ import annotations

import ast
import gzip
import os
import shutil
import sys
import tempfile
from collections.abc import Iterable
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import SimpleITK as sitk
import torch
from nibabel.nifti1 import Nifti1Image
from torch.utils.data import DataLoader
from torch.utils.data._utils.collate import default_collate
from tqdm.auto import trange

from . import constants
from .types import TypeNumber
from .types import TypePath


def to_tuple(
    value: Any,
    length: int = 1,
) -> tuple[TypeNumber, ...]:
    """Convert variable to tuple of length n.

    Example:
        >>> from torchio.utils import to_tuple
        >>> to_tuple(1, length=1)
        (1,)
        >>> to_tuple(1, length=3)
        (1, 1, 1)

    If value is an iterable, n is ignored and tuple(value) is returned

    Example:
        >>> to_tuple((1,), length=1)
        (1,)
        >>> to_tuple((1, 2), length=1)
        (1, 2)
        >>> to_tuple([1, 2], length=3)
        (1, 2)
    """
    try:
        iter(value)
        value = tuple(value)
    except TypeError:
        value = length * (value,)
    return value


def get_stem(
    path: TypePath | Sequence[TypePath],
) -> str | list[str]:
    """Get stem of path or paths.

    Example:
        >>> from torchio.utils import get_stem
        >>> get_stem('/home/user/my_image.nii.gz')
        'my_image'
    """

    def _get_stem(path_string: TypePath) -> str:
        return Path(path_string).name.split('.')[0]

    if isinstance(path, (str, os.PathLike)):
        return _get_stem(path)
    else:  # path is actually a sequence of paths
        return [_get_stem(p) for p in path]


def create_dummy_dataset(
    num_images: int,
    size_range: tuple[int, int],
    directory: TypePath | None = None,
    suffix: str = '.nii.gz',
    force: bool = False,
    verbose: bool = False,
):
    from .data import LabelMap
    from .data import ScalarImage
    from .data import Subject

    output_dir = tempfile.gettempdir() if directory is None else directory
    output_dir = Path(output_dir)
    images_dir = output_dir / 'dummy_images'
    labels_dir = output_dir / 'dummy_labels'

    if force:
        shutil.rmtree(images_dir)
        shutil.rmtree(labels_dir)

    subjects: list[Subject] = []
    if images_dir.is_dir():
        for i in trange(num_images):
            image_path = images_dir / f'image_{i}{suffix}'
            label_path = labels_dir / f'label_{i}{suffix}'
            subject = Subject(
                one_modality=ScalarImage(image_path),
                segmentation=LabelMap(label_path),
            )
            subjects.append(subject)
    else:
        images_dir.mkdir(exist_ok=True, parents=True)
        labels_dir.mkdir(exist_ok=True, parents=True)
        iterable: Iterable[int]
        if verbose:
            print('Creating dummy dataset...')  # noqa: T201
            iterable = trange(num_images)
        else:
            iterable = range(num_images)
        for i in iterable:
            shape = np.random.randint(*size_range, size=3)
            affine = np.eye(4)
            image = np.random.rand(*shape)
            label = np.ones_like(image)
            label[image < 0.33] = 0
            label[image > 0.66] = 2
            image *= 255

            image_path = images_dir / f'image_{i}{suffix}'
            nii = Nifti1Image(image.astype(np.uint8), affine)
            nii.to_filename(str(image_path))

            label_path = labels_dir / f'label_{i}{suffix}'
            nii = Nifti1Image(label.astype(np.uint8), affine)
            nii.to_filename(str(label_path))

            subject = Subject(
                one_modality=ScalarImage(image_path),
                segmentation=LabelMap(label_path),
            )
            subjects.append(subject)
    return subjects


def apply_transform_to_file(
    input_path: TypePath,
    transform,  # : Transform seems to create a circular import
    output_path: TypePath,
    class_: str = 'ScalarImage',
    verbose: bool = False,
):
    from . import data

    image = getattr(data, class_)(input_path)
    subject = data.Subject(image=image)
    transformed = transform(subject)
    transformed.image.save(output_path)
    if verbose and transformed.history:
        print('Applied transform:', transformed.history[0])  # noqa: T201


def guess_type(string: str) -> Any:
    # Adapted from
    # https://www.reddit.com/r/learnpython/comments/4599hl/module_to_guess_type_from_a_string/czw3f5s
    string = string.replace(' ', '')
    result_type: Any
    try:
        value = ast.literal_eval(string)
    except ValueError:
        result_type = str
    else:
        result_type = type(value)
    if result_type in (list, tuple):
        string = string[1:-1]  # remove brackets
        split = string.split(',')
        list_result = [guess_type(n) for n in split]
        value = tuple(list_result) if result_type is tuple else list_result
        return value
    try:
        value = result_type(string)
    except TypeError:
        value = None
    return value


def get_torchio_cache_dir() -> Path:
    return Path('~/.cache/torchio').expanduser()


def compress(
    input_path: TypePath,
    output_path: TypePath | None = None,
) -> Path:
    if output_path is None:
        output_path = Path(input_path).with_suffix('.nii.gz')
    with open(input_path, 'rb') as f_in:
        with gzip.open(output_path, 'wb') as f_out:
            shutil.copyfileobj(f_in, f_out)
    return Path(output_path)


def check_sequence(sequence: Sequence, name: str) -> None:
    try:
        iter(sequence)
    except TypeError as err:
        message = f'"{name}" must be a sequence, not {type(name)}'
        raise TypeError(message) from err


def get_major_sitk_version() -> int:
    # This attribute was added in version 2
    # https://github.com/SimpleITK/SimpleITK/pull/1171
    version = getattr(sitk, '__version__', None)
    major_version = 1 if version is None else 2
    return major_version


def history_collate(batch: Sequence, collate_transforms=True) -> dict:
    attr = constants.HISTORY if collate_transforms else 'applied_transforms'
    # Adapted from
    # https://github.com/romainVala/torchQC/blob/master/segmentation/collate_functions.py
    from .data import Subject

    first_element = batch[0]
    if isinstance(first_element, Subject):
        dictionary = {
            key: default_collate([d[key] for d in batch]) for key in first_element
        }
        if hasattr(first_element, attr):
            dictionary.update({attr: [getattr(d, attr) for d in batch]})
    else:
        dictionary = {}
    return dictionary


def get_subclasses(target_class: type) -> list[type]:
    subclasses = target_class.__subclasses__()
    subclasses += sum((get_subclasses(cls) for cls in subclasses), [])
    return subclasses


def get_first_item(data_loader: DataLoader):
    return next(iter(data_loader))


def get_batch_images_and_size(batch: dict) -> tuple[list[str], int]:
    """Get number of images and images names in a batch.

    Args:
        batch: Dictionary generated by a :class:`tio.SubjectsLoader`
            extracting data from a :class:`torchio.SubjectsDataset`.

    Raises:
        RuntimeError: If the batch does not seem to contain any dictionaries
            that seem to represent a :class:`torchio.Image`.
    """
    names = []
    for key, value in batch.items():
        if isinstance(value, dict) and constants.DATA in value:
            size = len(value[constants.DATA])
            names.append(key)
    if not names:
        raise RuntimeError('The batch does not seem to contain any images')
    return names, size


def get_subjects_from_batch(batch: dict) -> list:
    """Get list of subjects from collated batch.

    Args:
        batch: Dictionary generated by a :class:`tio.SubjectsLoader`
            extracting data from a :class:`torchio.SubjectsDataset`.
    """
    from .data import LabelMap
    from .data import ScalarImage
    from .data import Subject

    subjects = []
    image_names, batch_size = get_batch_images_and_size(batch)

    for i in range(batch_size):
        subject_dict = {}

        for key, value in batch.items():
            if key in image_names:
                image_name = key
                image_dict = value
                data = image_dict[constants.DATA][i]
                affine = image_dict[constants.AFFINE][i]
                path = Path(image_dict[constants.PATH][i])
                is_label = image_dict[constants.TYPE][i] == constants.LABEL
                klass = LabelMap if is_label else ScalarImage
                image = klass(tensor=data, affine=affine, filename=path.name)
                subject_dict[image_name] = image
            else:
                instance_value = value[i]
                subject_dict[key] = instance_value

        subject = Subject(subject_dict)

        if constants.HISTORY in batch:
            applied_transforms = batch[constants.HISTORY][i]
            for transform in applied_transforms:
                transform.add_transform_to_subject_history(subject)

        subjects.append(subject)
    return subjects


def add_images_from_batch(
    subjects: list,
    tensor: torch.Tensor,
    class_=None,
    name='prediction',
) -> None:
    """Add images to subjects in a list, typically from a network prediction.

    The spatial metadata (affine matrices) will be extracted from one of the
    images of each subject.

    Args:
        subjects: List of instances of :class:`torchio.Subject` to which images
            will be added.
        tensor: PyTorch tensor of shape :math:`(B, C, W, H, D)`, where
            :math:`B` is the batch size.
        class_: Class used to instantiate the images,
            e.g., :class:`torchio.LabelMap`.
            If ``None``, :class:`torchio.ScalarImage` will be used.
        name: Name of the images added to the subjects.
    """
    if class_ is None:
        from . import ScalarImage

        class_ = ScalarImage
    for subject, data in zip(subjects, tensor):
        one_image = subject.get_first_image()
        kwargs = {'tensor': data, 'affine': one_image.affine}
        if 'filename' in one_image:
            kwargs['filename'] = one_image['filename']
        image = class_(**kwargs)
        subject.add_image(image, name)


def guess_external_viewer() -> Path | None:
    """Guess the path to an executable that could be used to visualize images.

    Currently, it looks for 1) ITK-SNAP and 2) 3D Slicer. Implemented
    for macOS and Windows.
    """
    if 'SITK_SHOW_COMMAND' in os.environ:
        return Path(os.environ['SITK_SHOW_COMMAND'])
    platform = sys.platform
    itk = 'ITK-SNAP'
    slicer = 'Slicer'
    if platform == 'darwin':
        app_path = '/Applications/{}.app/Contents/MacOS/{}'
        itk_snap_path = Path(app_path.format(2 * (itk,)))
        if itk_snap_path.is_file():
            return itk_snap_path
        slicer_path = Path(app_path.format(2 * (slicer,)))
        if slicer_path.is_file():
            return slicer_path
    elif platform == 'win32':
        program_files_dir = Path(os.environ['ProgramW6432'])
        itk_snap_dirs = list(program_files_dir.glob('ITK-SNAP*'))
        if itk_snap_dirs:
            itk_snap_dir = itk_snap_dirs[-1]
            itk_snap_path = itk_snap_dir / 'bin/itk-snap.exe'
            if itk_snap_path.is_file():
                return itk_snap_path
        slicer_dirs = list(program_files_dir.glob('Slicer*'))
        if slicer_dirs:
            slicer_dir = slicer_dirs[-1]
            slicer_path = slicer_dir / 'slicer.exe'
            if slicer_path.is_file():
                return slicer_path
    elif 'linux' in platform:
        itk_snap_which = shutil.which('itksnap')
        if itk_snap_which is not None:
            return Path(itk_snap_which)
        slicer_which = shutil.which('Slicer')
        if slicer_which is not None:
            return Path(slicer_which)
    return None  # for mypy


def parse_spatial_shape(shape):
    result = to_tuple(shape, length=3)
    for n in result:
        if n < 1 or n % 1:
            message = (
                'All elements in a spatial shape must be positive integers,'
                f' but the following shape was passed: {shape}'
            )
            raise ValueError(message)
    if len(result) != 3:
        message = (
            'Spatial shapes must have 3 elements, but the following shape'
            f' was passed: {shape}'
        )
        raise ValueError(message)
    return result


def normalize_path(path: TypePath):
    return Path(path).expanduser().resolve()


def is_iterable(object: Any) -> bool:
    try:
        iter(object)
        return True
    except TypeError:
        return False
