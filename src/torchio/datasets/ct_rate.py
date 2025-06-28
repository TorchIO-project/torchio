from __future__ import annotations

import enum
import multiprocessing
from pathlib import Path
from typing import TYPE_CHECKING
from typing import Literal
from typing import Union

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

    This class helps loading the `CT-RATE dataset
    <https://huggingface.co/datasets/ibrahimhamamci/CT-RATE>`_,
    which contains chest CT scans with associated radiology reports and
    abnormality labels.

    The dataset must have been downloaded previously.

    Args:
        root: Root directory where the dataset has been downloaded.
        split: Dataset split to use, either ``'train'`` or ``'validation'``.
        num_subjects: Optional limit on the number of subjects to load (useful for
            debugging). If ``None``, all subjects in the split are loaded.
        report_key: Key to use for storing radiology reports in the Subject metadata.
        sizes: List of image sizes (in-plane, in voxels) to include.
        load_fixed: If ``True``, load the files with fixed spatial metadata
            added in `this pull request
            <https://huggingface.co/datasets/ibrahimhamamci/CT-RATE/discussions/85>`_.
            Otherwise, load the original files with incorrect spatial metadata.
        **kwargs: Additional arguments for SubjectsDataset.

    Examples:
        >>> from torchio.datasets import CtRate
        >>> dataset = CtRate('/path/to/CT-RATE', sizes=[512])
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
    REPORT_KEYS = [
        'ClinicalInformation_EN',
        'Findings_EN',
        'Impressions_EN',
        'Technique_EN',
    ]

    def __init__(
        self,
        root: TypePath,
        split: TypeSplit = 'train',
        *,
        num_subjects: int | None = None,
        report_key: str = 'report',
        sizes: list[int] | None = None,
        load_fixed: bool = True,
        **kwargs,
    ):
        self._root_dir = Path(root)
        self._num_subjects = num_subjects
        self._report_key = report_key
        self._sizes = self._SIZES if sizes is None else sizes

        self._split = self._parse_split(split)
        self.metadata = self._get_metadata()
        self._load_fixed = load_fixed
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
        relative_image_path = self._get_image_path(
            filename,
            load_fixed=self._load_fixed,
        )
        image_path = self._root_dir / relative_image_path
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
        report_dict = {}
        for key in self.REPORT_KEYS:
            report_dict[key] = subject_dict.pop(key)
        return report_dict

    @staticmethod
    def _get_image_path(filename: str, load_fixed: bool) -> Path:
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
        if load_fixed:
            split_dir = f'{split_dir}_fixed'
        level1 = f'{parts[0]}_{parts[1]}'
        level2 = f'{level1}_{parts[2]}'
        return Path(base_dir, split_dir, level1, level2, filename)
