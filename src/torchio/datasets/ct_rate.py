from __future__ import annotations

import ast
import enum
import multiprocessing
import shutil
from pathlib import Path
from typing import TYPE_CHECKING
from typing import Literal
from typing import Union

import numpy as np
import SimpleITK as sitk
from tqdm.auto import tqdm
from tqdm.contrib.concurrent import process_map
from tqdm.contrib.concurrent import thread_map

from ..data.dataset import SubjectsDataset
from ..data.image import ScalarImage
from ..data.subject import Subject
from ..external.imports import get_pandas
from ..types import TypePath

if TYPE_CHECKING:
    import pandas as pd


TypeSplit = Union[
    Literal['train'],
    Literal['valid'],
    Literal['validation'],
]

TypeParallelism = Literal['thread', 'process', None]


class MetadataIndexColumn(str, enum.Enum):
    SUBJECT_ID = 'subject_id'
    SCAN_ID = 'scan_id'
    RECONSTRUCTION_ID = 'reconstruction_id'


class CtRate(SubjectsDataset):
    """CT-RATE dataset.

    This class provides access to
    `CT-RATE <https://huggingface.co/datasets/ibrahimhamamci/CT-RATE>`_,
    which contains chest CT scans with associated radiology reports and
    abnormality labels.

    The dataset must have been downloaded previously.

    Args:
        root: Root directory where the dataset has been downloaded.
        split: Dataset split to use, either ``'train'`` or ``'validation'``.
        token: Hugging Face token for accessing gated repositories. Alternatively,
            login using `huggingface-cli login` to cache the token.
        num_subjects: Optional limit on the number of subjects to load (useful for
            testing). If ``None``, all subjects in the split are loaded.
        report_key: Key to use for storing radiology reports in the Subject metadata.
        sizes: List of image sizes (in pixels) to include. Default: [512, 768, 1024].
        **kwargs: Additional arguments for SubjectsDataset.

    Examples:
        >>> dataset = CtRate('/path/to/data', split='train')
    """

    _REPO_ID = 'ibrahimhamamci/CT-RATE'
    _FILENAME_KEY = 'VolumeName'
    _SIZES = [512, 768, 1024]
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
        token: str | None = None,
        num_subjects: int | None = None,
        report_key: str = 'report',
        sizes: list[int] | None = None,
        **kwargs,
    ):
        self._root_dir = Path(root)
        self._token = token
        self._num_subjects = num_subjects
        self._report_key = report_key
        self._sizes = self._SIZES if sizes is None else sizes

        self._split = self._parse_split(split)
        self.metadata = self._get_metadata()
        subjects_list = self._get_subjects_list(self.metadata)
        super().__init__(subjects_list, **kwargs)

    @staticmethod
    def _parse_split(split: str) -> str:
        """Normalize the split name.

        Converts 'validation' to 'valid' and validates that the split name
        is one of the allowed values.

        Args:
            split: The split name to parse ('train', 'valid', or 'validation').

        Returns:
            str: Normalized split name ('train' or 'valid').

        Raises:
            ValueError: If the split name is not one of the allowed values.
        """
        if split in ['valid', 'validation']:
            return 'valid'
        if split not in ['train', 'valid']:
            raise ValueError(f"Invalid split '{split}'. Use 'train' or 'valid'")
        return split

    def _get_csv(
        self,
        dirname: str,
        filename: str,
    ) -> pd.DataFrame:
        """Load a CSV file from the specified directory within the dataset.

        Args:
            dirname: Directory name within 'dataset/' where the CSV is located.
            filename: Name of the CSV file to load.
        """
        subfolder = Path(f'dataset/{dirname}')
        path = Path(self._root_dir, subfolder, filename)
        pd = get_pandas()
        table = pd.read_csv(path)
        return table

    def _get_csv_prefix(self, expand_validation: bool = True) -> str:
        """Get the prefix for CSV filenames based on the current split.

        Returns the appropriate prefix for CSV filenames based on the current split.
        For the validation split, can either return 'valid' or 'validation' depending
        on the expand_validation parameter.

        Args:
            expand_validation: If ``True`` and split is ``'valid'``, return
                ``'validation'``. Otherwise, return the split name as is.
        """
        if expand_validation and self._split == 'valid':
            prefix = 'validation'
        else:
            prefix = self._split
        return prefix

    def _get_metadata(self) -> pd.DataFrame:
        """Load and process the dataset metadata.

        Loads metadata from the appropriate CSV file, filters images by size,
        extracts subject, scan, and reconstruction IDs from filenames, and
        merges in reports and abnormality labels.
        """
        dirname = 'metadata'
        prefix = self._get_csv_prefix()
        filename = f'{prefix}_metadata.csv'
        metadata = self._get_csv(dirname, filename)

        # Exclude images with size not in self._sizes
        rows_int = metadata['Rows'].astype(int)
        metadata = metadata[rows_int.isin(self._sizes)]

        index_columns = [
            MetadataIndexColumn.SUBJECT_ID.value,
            MetadataIndexColumn.SCAN_ID.value,
            MetadataIndexColumn.RECONSTRUCTION_ID.value,
        ]
        pattern = r'\w+_(\d+)_(\w+)_(\d+)\.nii\.gz'
        metadata[index_columns] = metadata[self._FILENAME_KEY].str.extract(pattern)

        if self._num_subjects is not None:
            metadata = self._keep_n_subjects(metadata, self._num_subjects)

        # Add reports and abnormality labels to metadata, keeping only the rows for the
        # images in the metadata table
        metadata = self._merge(metadata, self._get_reports())
        metadata = self._merge(metadata, self._get_labels())

        metadata.set_index(index_columns, inplace=True)
        return metadata

    def _merge(self, base_df: pd.DataFrame, new_df: pd.DataFrame) -> pd.DataFrame:
        """Merge a new dataframe into the base dataframe using the filename as the key.

        This method performs a left join between ``base_df`` and ``new_df`` using the
        volume filename as the join key, ensuring that all records from ``base_df`` are
        preserved while matching data from ``new_df`` is added.

        Args:
            base_df: The primary dataframe to merge into.
            new_df: The dataframe containing additional data to be merged.

        Returns:
            pd.DataFrame: The merged dataframe with all rows from base_df and
            matching columns from new_df.
        """
        pd = get_pandas()
        return pd.merge(
            base_df,
            new_df,
            on=self._FILENAME_KEY,
            how='left',
        )

    def _keep_n_subjects(self, metadata: pd.DataFrame, n: int) -> pd.DataFrame:
        """Limit the metadata to the first ``n`` subjects.

        Args:
            metadata: The complete metadata dataframe.
            n: Maximum number of subjects to keep.
        """
        unique_subjects = metadata['subject_id'].unique()
        selected_subjects = unique_subjects[:n]
        return metadata[metadata['subject_id'].isin(selected_subjects)]

    def _get_reports(self) -> pd.DataFrame:
        """Load the radiology reports associated with the CT scans.

        Retrieves the CSV file containing radiology reports for the current split
        (train or validation).
        """
        dirname = 'radiology_text_reports'
        prefix = self._get_csv_prefix()
        filename = f'{prefix}_reports.csv'
        return self._get_csv(dirname, filename)

    def _get_labels(self) -> pd.DataFrame:
        """Load the abnormality labels for the CT scans.

        Retrieves the CSV file containing predicted abnormality labels for the
        current split.
        """
        dirname = 'multi_abnormality_labels'
        prefix = self._get_csv_prefix(expand_validation=False)
        filename = f'{prefix}_predicted_labels.csv'
        return self._get_csv(dirname, filename)

    def _get_subjects_list(self, metadata: pd.DataFrame) -> list[Subject]:
        """Create a list of Subject instances from the metadata.

        Processes the metadata to create Subject objects, each containing one or more
        CT images. Processing is performed in parallel.

        Note:
            This method uses parallelization to improve performance when creating
            multiple Subject instances.
        """
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
        """Create a Subject instance for a specific subject.

        Processes all images belonging to a single subject and creates a Subject
        object containing those images.

        Args:
            subject_id_and_metadata: A tuple containing the subject ID (string) and a
                DataFrame containing metadata for all images associated to that subject.
        """
        subject_id, subject_df = subject_id_and_metadata
        subject_dict: dict[str, str | ScalarImage] = {'subject_id': subject_id}
        for _, image_row in subject_df.iterrows():
            image = self._instantiate_image(image_row)
            scan_id = image_row['scan_id']
            reconstruction_id = image_row['reconstruction_id']
            image_key = f'scan_{scan_id}_reconstruction_{reconstruction_id}'
            subject_dict[image_key] = image
        return Subject(**subject_dict)  # type: ignore[arg-type]

    def _instantiate_image(self, image_row: pd.Series) -> ScalarImage:
        """Create a ScalarImage object for a specific image.

        Processes a row from the metadata DataFrame to create a ScalarImage object,

        Args:
            image_row: A pandas Series representing a row from the metadata DataFrame,
                containing information about a single image.
        """
        image_dict = image_row.to_dict()
        filename = image_dict[self._FILENAME_KEY]
        image_path = self._root_dir / self._get_image_path(filename)
        report_dict = self._extract_report_dict(image_dict)
        image_dict[self._report_key] = report_dict
        image = ScalarImage(image_path, **image_dict)
        return image

    def _extract_report_dict(self, subject_dict: dict[str, str]) -> dict[str, str]:
        """Extract radiology report information from the subject dictionary.

        Extracts the English radiology report components (clinical information,
        findings, impressions, and technique) from the subject dictionary and
        removes these keys from the original dictionary.

        Args:
            subject_dict: Image metadata including report fields.

        Note:
            This method modifies the input subject_dict by removing the report keys.
        """
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
        """Construct the relative path to an image file within the dataset structure.

        Parses the filename to determine the hierarchical directory structure
        where the image is stored in the CT-RATE dataset.

        Args:
            filename: The name of the image file (e.g., 'train_2_a_1.nii.gz').

        Returns:
            Path: The relative path to the image file within the dataset directory.

        Example:
            >>> path = CtRate._get_image_path('train_2_a_1.nii.gz')
            # Returns Path('dataset/train/train_2/train_2_a/train_2_a_1.nii.gz')
        """
        parts = filename.split('_')
        base_dir = 'dataset'
        split_dir = parts[0]
        level1 = f'{parts[0]}_{parts[1]}'
        level2 = f'{level1}_{parts[2]}'
        return Path(base_dir, split_dir, level1, level2, filename)

    @staticmethod
    def _fix_image(image: ScalarImage, out_path: Path, *, force: bool = False) -> None:
        """Fix the spatial metadata of a CT-RATE image file.

        The original NIfTI files in the CT-RATE dataset have incorrect spatial
        metadata. This method reads the image, fixes the spacing, origin, and
        orientation based on the metadata provided in the CSV, and applies the correct
        rescaling to convert to Hounsfield units.

        Args:
            in_path: The path to the image file to fix.
            out_path: The path where the fixed image will be saved.

        Note:
            This method overwrites the original file with the fixed version.
            The fixed image is stored as INT16 with proper HU values.
        """
        # Adapted from https://huggingface.co/datasets/ibrahimhamamci/CT-RATE/blob/main/download_scripts/fix_metadata.py
        if not force and out_path.exists():
            return
        spacing_x, spacing_y = map(float, ast.literal_eval(image['XYSpacing']))
        spacing_z = image['ZSpacing']
        image_sitk = sitk.ReadImage(str(image.path))
        image_sitk.SetSpacing((spacing_x, spacing_y, spacing_z))

        image_sitk.SetOrigin(ast.literal_eval(image['ImagePositionPatient']))

        orientation = ast.literal_eval(image['ImageOrientationPatient'])
        row_cosine, col_cosine = orientation[:3], orientation[3:6]
        z_cosine = np.cross(row_cosine, col_cosine).tolist()
        image_sitk.SetDirection(row_cosine + col_cosine + z_cosine)

        RescaleIntercept = image['RescaleIntercept']
        RescaleSlope = image['RescaleSlope']
        adjusted_hu = image_sitk * RescaleSlope + RescaleIntercept
        cast_int16 = sitk.Cast(adjusted_hu, sitk.sitkInt16)

        out_path.parent.mkdir(parents=True, exist_ok=True)
        sitk.WriteImage(cast_int16, str(out_path))
        return cast_int16

    def _copy_not_images(self, out_dir: Path) -> None:
        """Copy all files from the root directory except the images."""
        for path in self._root_dir.iterdir():
            if path.name == 'dataset':
                for subdirectory in path.iterdir():
                    if subdirectory.name in ['train', 'valid']:
                        continue
                    print(
                        f'Copying {subdirectory} to {out_dir / subdirectory.relative_to(self._root_dir)}'
                    )
                    shutil.copytree(
                        subdirectory,
                        out_dir / subdirectory.relative_to(self._root_dir),
                        dirs_exist_ok=True,
                    )
            elif path.name.startswith('.'):
                continue
            elif path.is_dir():
                print(f'Copying {path} to {out_dir / path.name}')
                shutil.copytree(
                    path,
                    out_dir / path.name,
                    dirs_exist_ok=True,
                )
            else:
                print(f'Copying {path} to {out_dir / path.name}')
                shutil.copy(path, out_dir / path.name)

    def fix_metadata(
        self,
        out_dir: str | Path,
        parallelism: TypeParallelism = None,
    ) -> CtRate:
        """Fix the metadata of all images in the dataset.

        Reads each image, applies the correct spatial metadata, and saves the fixed
        image to the specified output directory.

        Args:
            out_dir: The directory where the fixed images will be saved.
        """
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        # self._copy_not_images(out_dir)
        images = []
        out_paths = []
        for subject in self.dry_iter():
            for image in subject.get_images():
                out_path = out_dir / image.path.relative_to(self._root_dir)
                images.append(image)
                out_paths.append(out_path)
        if parallelism == 'thread':
            thread_map(
                self._fix_image,
                images,
                out_paths,
                max_workers=multiprocessing.cpu_count(),
                desc='Fixing metadata',
            )
        elif parallelism == 'process':
            process_map(
                self._fix_image,
                images,
                out_paths,
                max_workers=multiprocessing.cpu_count(),
                desc='Fixing metadata',
            )
        else:
            zipped = zip(images, out_paths)
            with tqdm(total=len(images), desc='Fixing metadata') as pbar:
                for image, out_path in zipped:
                    pbar.set_description(f'Fixing {image.path.name}')
                    self._fix_image(image, out_path)
                    pbar.update(1)
        new_dataset = CtRate(
            out_dir,
            split=self._split,
            token=self._token,
            num_subjects=self._num_subjects,
            report_key=self._report_key,
            sizes=self._sizes,
        )
        return new_dataset
