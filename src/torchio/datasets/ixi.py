"""IXI dataset: ~600 brain MRIs from healthy subjects.

The `Information eXtraction from Images (IXI)
<https://brain-development.org/ixi-dataset/>`_ dataset contains
nearly 600 MR images from normal, healthy subjects.

This data is made available under the Creative Commons CC BY-SA 3.0
license. If you use it, please acknowledge the source.
"""

from __future__ import annotations

import shutil
from collections.abc import Sequence
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any

from ..data.image import LabelMap
from ..data.image import ScalarImage
from ..data.subject import Subject
from ..download import download_and_extract_archive
from ..types import TypePath


def ixi(
    root: TypePath,
    *,
    download: bool = False,
    modalities: Sequence[str] = ("T1", "T2"),
) -> list[Subject]:
    """Download and load the full IXI dataset.

    Args:
        root: Root directory for the dataset.
        download: If ``True``, download the data into ``root``.
        modalities: Modalities to include. Must be a subset of
            ``('T1', 'T2', 'PD', 'MRA', 'DTI')``.

    Returns:
        List of subjects, one per scan with all requested modalities.

    Warning:
        The dataset is several GB. Downloading may take a while.
    """
    root = Path(root)
    md5s = _IXI_MD5
    for m in modalities:
        if m not in md5s:
            msg = f'Modality "{m}" must be one of {tuple(md5s.keys())}'
            raise ValueError(msg)
    if download:
        _download_ixi(root, modalities, md5s)
    if not all((root / m).is_dir() for m in modalities):
        msg = "Dataset not found. Use download=True to download it"
        raise RuntimeError(msg)
    return _load_ixi_subjects(root, modalities)


def ixi_tiny(
    root: TypePath,
    *,
    download: bool = False,
) -> list[Subject]:
    r"""Download and load IXITiny (566 $T_1$ images + segmentations).

    All images have shape $83 \times 44 \times 55$. Useful as a
    medical image MNIST for quick experiments.

    Args:
        root: Root directory for the dataset.
        download: If ``True``, download the data into ``root``.

    Returns:
        List of subjects with ``image`` and ``label`` keys.
    """
    root = Path(root)
    if download:
        _download_ixi_tiny(root)
    if not root.is_dir():
        msg = "Dataset not found. Use download=True to download it"
        raise RuntimeError(msg)
    return _load_ixi_tiny_subjects(root)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

_IXI_BASE_URL = (
    "http://biomedic.doc.ic.ac.uk/brain-development/downloads/IXI/IXI-{modality}.tar"
)
_IXI_MD5: dict[str, str] = {
    "T1": "34901a0593b41dd19c1a1f746eac2d58",
    "T2": "e3140d78730ecdd32ba92da48c0a9aaa",
    "PD": "88ecd9d1fa33cb4a2278183b42ffd749",
    "MRA": "29be7d2fee3998f978a55a9bdaf3407e",
    "DTI": "636573825b1c8b9e8c78f1877df3ee66",
}

_IXI_TINY_URL = "https://www.dropbox.com/s/ogxjwjxdv5mieah/ixi_tiny.zip?dl=1"
_IXI_TINY_MD5 = "bfb60f4074283d78622760230bfa1f98"


def _download_ixi(
    root: Path,
    modalities: Sequence[str],
    md5s: dict[str, str],
) -> None:
    for modality in modalities:
        modality_dir = root / modality
        if modality_dir.is_dir():
            continue
        modality_dir.mkdir(exist_ok=True, parents=True)
        url = _IXI_BASE_URL.format(modality=modality)
        with NamedTemporaryFile(suffix=".tar", delete=False) as f:
            download_and_extract_archive(
                url,
                download_root=modality_dir,
                filename=f.name,
                md5=md5s[modality],
            )


def _load_ixi_subjects(
    root: Path,
    modalities: Sequence[str],
) -> list[Subject]:
    first = modalities[0]
    paths = sorted((root / first).glob("*.nii.gz"))
    subjects: list[Subject] = []
    for filepath in paths:
        sid = _subject_id(filepath)
        images: dict[str, Any] = {"subject_id": sid}
        images[first] = ScalarImage(filepath)
        skip = False
        for m in modalities[1:]:
            matches = sorted((root / m).glob(f"{sid}-{m}.nii.gz"))
            if matches:
                images[m] = ScalarImage(matches[0])
            else:
                skip = True
                break
        if not skip:
            subjects.append(Subject(**images))
    return subjects


def _download_ixi_tiny(root: Path) -> None:
    if root.is_dir():
        return
    with NamedTemporaryFile(suffix=".zip", delete=False) as f:
        download_and_extract_archive(
            _IXI_TINY_URL,
            download_root=root,
            filename=f.name,
            md5=_IXI_TINY_MD5,
        )
    ixi_tiny_dir = root / "ixi_tiny"
    (ixi_tiny_dir / "image").rename(root / "image")
    (ixi_tiny_dir / "label").rename(root / "label")
    shutil.rmtree(ixi_tiny_dir)


def _load_ixi_tiny_subjects(root: Path) -> list[Subject]:
    image_paths = sorted((root / "image").glob("*.nii.gz"))
    label_paths = sorted((root / "label").glob("*.nii.gz"))
    if not (image_paths and label_paths):
        msg = f"Images not found. Remove {root} and try again"
        raise FileNotFoundError(msg)
    subjects: list[Subject] = []
    for img_path, lbl_path in zip(image_paths, label_paths, strict=True):
        subjects.append(
            Subject(
                image=ScalarImage(img_path),
                label=LabelMap(lbl_path),
                subject_id=_subject_id(img_path),
            ),
        )
    return subjects


def _subject_id(path: Path) -> str:
    return "-".join(path.name.split("-")[:-1])
