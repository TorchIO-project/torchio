import ast
import multiprocessing
from importlib.util import find_spec
from pathlib import Path
from typing import Literal
from typing import Optional
from typing import Union

import numpy as np
import pandas as pd
import SimpleITK as sitk
from huggingface_hub import hf_hub_download
from huggingface_hub.errors import GatedRepoError
from tqdm.contrib.concurrent import thread_map

from ..data.dataset import SubjectsDataset
from ..data.image import ScalarImage
from ..data.subject import Subject
from ..types import TypePath

TypeSplit = Union[
    Literal['train'],
    Literal['valid'],
    Literal['validation'],
]


# TODO: add docstring
class CtRate(SubjectsDataset):
    _REPO_ID = 'ibrahimhamamci/CT-RATE'

    def __init__(
        self,
        root: TypePath,
        split: TypeSplit = 'train',
        *,
        token: Optional[str] = None,
        download: bool = False,
        num_subjects: Optional[int] = None,
        **kwargs,
    ):
        self._root_dir = Path(root)
        self._token = token
        self._download = download
        self._num_subjects = num_subjects

        self._split = self._parse_split(split)
        self._metadata = self._get_metadata()
        subjects_list = self._get_subjects_list()
        super().__init__(subjects_list, **kwargs)

    @staticmethod
    def _parse_split(split: str) -> str:
        if split in ['valid', 'validation']:
            return 'valid'
        if split not in ['train', 'valid']:
            raise ValueError(f"Invalid split '{split}'. Use 'train' or 'valid'")
        return split

    def _get_metadata(self) -> pd.DataFrame:
        subfolder = Path('dataset/metadata')
        filename = f'{self._split}_metadata.csv'
        metadata_path = Path(self._root_dir, subfolder, filename)
        if not metadata_path.exists():
            self._download_file(metadata_path)
        metadata = pd.read_csv(metadata_path)
        if self._num_subjects is not None:
            metadata = metadata.head(self._num_subjects)
        return metadata

    def _download_file(self, path: Path) -> None:
        if self._download:
            _check_huggingface_hub()
            relative_path = path.relative_to(self._root_dir)
            try:
                hf_hub_download(
                    repo_id=self._REPO_ID,
                    repo_type='dataset',
                    token=self._token,
                    subfolder=str(relative_path.parent),
                    filename=relative_path.name,
                    local_dir=self._root_dir,
                )
            except GatedRepoError as e:
                message = (
                    f'The dataset "{self._REPO_ID}" is gated. Visit'
                    f' https://huggingface.co/datasets/{self._REPO_ID}, accept the'
                    ' terms and conditions, and log in or create and pass a token to'
                    ' the `token` argument'
                )
                raise RuntimeError(message) from e
        else:
            raise FileNotFoundError(
                f'File "{path}" not found.'
                " Set 'download=True' to download the dataset"
            )

    def _get_subjects_list(self) -> list[Subject]:
        subjects = thread_map(
            self._get_subject,
            self._metadata.iterrows(),
            max_workers=multiprocessing.cpu_count(),
            total=len(self._metadata),
        )
        return subjects

    def _get_subject(self, index_and_row: tuple[int, pd.Series]) -> Subject:
        _, row = index_and_row
        subject_dict = row.to_dict()
        filename = subject_dict.pop('VolumeName')
        image_path = self._root_dir / self._get_image_path(filename)
        if not image_path.exists():
            self._download_file(image_path)
            self._fix_image(image_path, subject_dict)
        image = ScalarImage(image_path)
        subject_dict['image'] = image
        return Subject(**subject_dict)

    @staticmethod
    def _get_image_path(filename: str) -> Path:
        parts = filename.split('_')
        base_dir = 'dataset'
        split_dir = parts[0]
        level1 = f'{parts[0]}_{parts[1]}'
        level2 = f'{level1}_{parts[2]}'
        return Path(base_dir, split_dir, level1, level2, filename)

    @staticmethod
    def _fix_image(path: Path, metadata: dict[str, str]) -> None:
        # Adapted from https://huggingface.co/datasets/ibrahimhamamci/CT-RATE/blob/main/download_scripts/fix_metadata.py
        image = sitk.ReadImage(str(path))

        spacing_x, spacing_y = map(float, ast.literal_eval(metadata['XYSpacing']))
        spacing_z = metadata['ZSpacing']
        image.SetSpacing((spacing_x, spacing_y, spacing_z))

        image.SetOrigin(ast.literal_eval(metadata['ImagePositionPatient']))

        orientation = ast.literal_eval(metadata['ImageOrientationPatient'])
        row_cosine, col_cosine = orientation[:3], orientation[3:6]
        z_cosine = np.cross(row_cosine, col_cosine).tolist()
        image.SetDirection(row_cosine + col_cosine + z_cosine)

        RescaleIntercept = metadata['RescaleIntercept']
        RescaleSlope = metadata['RescaleSlope']
        adjusted_hu = image * RescaleSlope + RescaleIntercept
        cast_int16 = sitk.Cast(adjusted_hu, sitk.sitkInt16)

        sitk.WriteImage(cast_int16, str(path))


def _check_huggingface_hub() -> None:
    if find_spec('huggingface_hub') is None:
        message = (
            'The `huggingface_hub` package is required to download the dataset.'
            ' Install TorchIO with the `huggingface` extra:'
            ' `pip install torchio[huggingface]`.'
        )
        raise ImportError(message)
