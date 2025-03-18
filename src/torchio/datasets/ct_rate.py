from __future__ import annotations

import ast
import multiprocessing
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
    ABNORMALITIES = [
        'Medical material',
        'Arterial wall calcification',
        'Cardiomegaly',
        'Pericardial effusion',
        'Coronary artery wall calcification',
        'Hiatal hernia',
        'Lymphadenopathy',
        'Emphysema',
        'Atelectasis',
        'Lung nodule',
        'Lung opacity',
        'Pulmonary fibrotic sequela',
        'Pleural effusion',
        'Mosaic attenuation pattern',
        'Peribronchial thickening',
        'Consolidation',
        'Bronchiectasis',
        'Interlobular septal thickening',
    ]

    def __init__(
        self,
        root: TypePath,
        split: TypeSplit = 'train',
        *,
        token: Optional[str] = None,
        download: bool = False,
        num_subjects: Optional[int] = None,
        report_key: str = 'report',
        **kwargs,
    ):
        self._root_dir = Path(root)
        self._token = token
        self._download = download
        self._num_subjects = num_subjects
        self._report_key = report_key

        self._split = self._parse_split(split)
        self.metadata = self._get_metadata()
        subjects_list = self._get_subjects_list(self.metadata)
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

    def _get_csv_prefix(self, expand_validation: bool = True) -> str:
        if expand_validation and self._split == 'valid':
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

        # Add reports and abnormality labels to metadata, keeping only the rows for the
        # images in the metadata table
        metadata = self._merge(metadata, self._get_reports())
        metadata = self._merge(metadata, self._get_labels())

        metadata.set_index(index_columns, inplace=True)
        return metadata

    def _merge(self, base_df, new_df):
        pd = get_pandas()
        return pd.merge(
            base_df,
            new_df,
            on=self._FILENAME_KEY,
            how='left',
        )

    def _get_reports(self) -> pd.DataFrame:
        dirname = 'radiology_text_reports'
        prefix = self._get_csv_prefix()
        filename = f'{prefix}_reports.csv'
        return self._get_csv(dirname, filename)

    def _get_labels(self) -> pd.DataFrame:
        dirname = 'multi_abnormality_labels'
        prefix = self._get_csv_prefix(expand_validation=False)
        filename = f'{prefix}_predicted_labels.csv'
        return self._get_csv(dirname, filename)

    def _download_file_if_needed(self, path: Path) -> None:
        if self._download:
            self._download_file(path)
        else:
            raise FileNotFoundError(
                f'File "{path}" not found.'
                " Set 'download=True' to download the dataset"
            )

    def _download_file(self, path: Path) -> None:
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

    def _get_subjects_list(self, metadata: pd.DataFrame) -> list[Subject]:
        df_no_index = metadata.reset_index()
        num_subjects = df_no_index['subject_id'].nunique()
        iterable = df_no_index.groupby('subject_id')
        subjects = thread_map(
            self._get_subject,
            iterable,
            max_workers=multiprocessing.cpu_count(),
            total=num_subjects,
        )
        return subjects

    def _get_subject(
        self,
        subject_id_and_metadata: tuple[str, pd.DataFrame],
    ) -> Subject:
        subject_id, subject_df = subject_id_and_metadata
        subject_dict: dict[str, Union[str, ScalarImage]] = {'subject_id': subject_id}
        for _, image_row in subject_df.iterrows():
            image = self._instantiate_image(image_row)
            scan_id = image_row['scan_id']
            reconstruction_id = image_row['reconstruction_id']
            image_key = f'scan_{scan_id}_reconstruction_{reconstruction_id}'
            subject_dict[image_key] = image
        return Subject(**subject_dict)

    def _instantiate_image(self, image_row: pd.Series) -> ScalarImage:
        image_dict = image_row.to_dict()
        filename = image_dict[self._FILENAME_KEY]
        image_path = self._root_dir / self._get_image_path(filename)
        if not image_path.exists():
            self._download_file_if_needed(image_path)
            self._fix_image(image_path, image_dict)
        report_dict = self._extract_report_dict(image_dict)
        image_dict[self._report_key] = report_dict
        image = ScalarImage(image_path, **image_dict)
        return image

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
