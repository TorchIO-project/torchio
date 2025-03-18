from __future__ import annotations

import ast
import multiprocessing
from importlib.util import find_spec
from pathlib import Path
from typing import TYPE_CHECKING
from typing import Literal
from typing import Optional
from typing import Union

import numpy as np
import SimpleITK as sitk
from tqdm.contrib.concurrent import thread_map

from ..data.dataset import SubjectsDataset
from ..data.image import ScalarImage
from ..data.subject import Subject
from ..external.imports import get_huggingface_hub
from ..external.imports import get_pandas
from ..types import TypePath

if TYPE_CHECKING:
    import pandas as pd


TypeSplit = Union[
    Literal['train'],
    Literal['valid'],
    Literal['validation'],
]


# TODO: add docstring
class CtRate(SubjectsDataset):
    _REPO_ID = 'ibrahimhamamci/CT-RATE'
    _FILENAME_KEY = 'VolumeName'

    def __init__(
        self,
        root: TypePath,
        split: TypeSplit = 'train',
        *,
        token: Optional[str] = None,
        download: bool = False,
        num_subjects: Optional[int] = None,
        image_key: str = 'image',
        report_key: str = 'report',
        **kwargs,
    ):
        self._root_dir = Path(root)
        self._token = token
        self._download = download
        self._num_subjects = num_subjects
        self._image_key = image_key
        self._report_key = report_key

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

    def _get_csv(
        self,
        dirname: str,
        filename: str,
        num_subjects: Optional[int] = None,
    ) -> pd.DataFrame:
        subfolder = Path(f'dataset/{dirname}')
        path = Path(self._root_dir, subfolder, filename)
        if not path.exists():
            self._download_file_if_needed(path)
        pd = get_pandas()
        table = pd.read_csv(path)
        if num_subjects is not None:
            table = table.head(num_subjects)
        return table

    def _get_csv_prefix(self) -> str:
        if self._split == 'valid':
            prefix = 'validation'
        else:
            prefix = self._split
        return prefix

    def _get_metadata(self) -> pd.DataFrame:
        dirname = 'metadata'
        prefix = self._get_csv_prefix()
        filename = f'{prefix}_metadata.csv'
        metadata = self._get_csv(dirname, filename, self._num_subjects)

        index_columns = [
            'subject_id',
            'scan_id',
            'reconstruction_id',
        ]
        pattern = r'\w+_(\d+)_(\w+)_(\d+)\.nii\.gz'
        metadata[index_columns] = metadata[self._FILENAME_KEY].str.extract(pattern)

        # Add reports to metadata, keeping only the reports for the images in the
        # metadata table
        pd = get_pandas()
        metadata = pd.merge(
            metadata,
            self._get_reports(),
            on=self._FILENAME_KEY,
            how='left',
        )

        metadata.set_index(index_columns, inplace=True)
        return metadata

    def _get_reports(self) -> pd.DataFrame:
        dirname = 'radiology_text_reports'
        prefix = self._get_csv_prefix()
        filename = f'{prefix}_reports.csv'
        reports = self._get_csv(dirname, filename)
        return reports

    def _download_file_if_needed(self, path: Path) -> None:
        if self._download:
            self._download_file(path)
        else:
            raise FileNotFoundError(
                f'File "{path}" not found.'
                " Set 'download=True' to download the dataset"
            )

    def _download_file(self, path: Path) -> None:
        _check_huggingface_hub()
        relative_path = path.relative_to(self._root_dir)
        huggingface_hub = get_huggingface_hub()
        try:
            huggingface_hub.hf_hub_download(
                repo_id=self._REPO_ID,
                repo_type='dataset',
                token=self._token,
                subfolder=str(relative_path.parent),
                filename=relative_path.name,
                local_dir=self._root_dir,
            )
        except huggingface_hub.errors.GatedRepoError as e:
            message = (
                f'The dataset "{self._REPO_ID}" is gated. Visit'
                f' https://huggingface.co/datasets/{self._REPO_ID}, accept the'
                ' terms and conditions, and log in or create and pass a token to'
                ' the `token` argument'
            )
            raise RuntimeError(message) from e

    def _get_subjects_list(self) -> list[Subject]:
        subjects = thread_map(
            self._get_subject,
            self._metadata.iterrows(),
            max_workers=multiprocessing.cpu_count(),
            total=len(self._metadata),
        )
        return subjects

    def _get_subject(self, index_and_row: tuple[str, pd.Series]) -> Subject:
        _, row = index_and_row
        subject_dict = row.to_dict()
        filename = subject_dict[self._FILENAME_KEY]
        image_path = self._root_dir / self._get_image_path(filename)
        if not image_path.exists():
            self._download_file_if_needed(image_path)
            self._fix_image(image_path, subject_dict)
        image = ScalarImage(image_path)
        subject_dict[self._image_key] = image
        report_dict = self._extract_report_dict(subject_dict)
        subject_dict[self._report_key] = report_dict
        return Subject(**subject_dict)

    def _extract_report_dict(self, subject_dict: dict[str, str]) -> dict[str, str]:
        report_keys = [
            'ClinicalInformation_EN',
            'Findings_EN',
            'Impressions_EN',
            'Technique_EN',
        ]
        report_dict = {}
        for key in report_keys:
            report_dict[key] = subject_dict.pop(key)
        return report_dict

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
