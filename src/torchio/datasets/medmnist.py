"""3D MedMNIST v2 datasets.

Datasets from `MedMNIST v2: A Large-Scale Lightweight Benchmark for
2D and 3D Biomedical Image Classification
<https://arxiv.org/abs/2110.14795>`_.

Check the `MedMNIST website <https://medmnist.com/>`_ for details
and licensing.
"""

from __future__ import annotations

import numpy as np
import torch

from ..data.image import ScalarImage
from ..data.subject import Subject
from ..download import download_url
from ..download import get_torchio_cache_dir


def _load_medmnist(
    class_name: str,
    split: str,
) -> list[Subject]:
    """Shared loader for all MedMNIST 3D datasets."""
    valid = ("train", "training", "val", "validation", "test", "testing")
    if split not in valid:
        msg = f"Split must be one of {valid}, got {split!r}"
        raise ValueError(msg)
    match split:
        case "training":
            split = "train"
        case "validation":
            split = "val"
        case "testing":
            split = "test"

    filename = f"{class_name}.npz"
    url = f"https://zenodo.org/record/5208230/files/{filename}?download=1"
    download_root = get_torchio_cache_dir() / "MedMNIST"
    download_url(url, download_root, filename=filename)
    path = download_root / filename
    npz = np.load(path)
    images = npz[f"{split}_images"]
    labels = npz[f"{split}_labels"]
    subjects: list[Subject] = []
    for image, label in zip(images, labels, strict=True):
        tensor = torch.from_numpy(image[np.newaxis].copy()).float()
        scalar = ScalarImage.from_tensor(  # ty: ignore[unresolved-attribute]
            tensor,
        )
        subjects.append(
            Subject(
                image=scalar,
                labels=torch.from_numpy(label.copy()),
            ),
        )
    return subjects


def organ_mnist_3d(split: str = "train") -> list[Subject]:
    """3D organ segmentation dataset.

    Args:
        split: ``'train'``, ``'val'``, or ``'test'``.
    """
    return _load_medmnist("organmnist3d", split)


def nodule_mnist_3d(split: str = "train") -> list[Subject]:
    """3D lung nodule dataset.

    Args:
        split: ``'train'``, ``'val'``, or ``'test'``.
    """
    return _load_medmnist("nodulemnist3d", split)


def adrenal_mnist_3d(split: str = "train") -> list[Subject]:
    """3D adrenal gland dataset.

    Args:
        split: ``'train'``, ``'val'``, or ``'test'``.
    """
    return _load_medmnist("adrenalmnist3d", split)


def fracture_mnist_3d(split: str = "train") -> list[Subject]:
    """3D bone fracture dataset.

    Args:
        split: ``'train'``, ``'val'``, or ``'test'``.
    """
    return _load_medmnist("fracturemnist3d", split)


def vessel_mnist_3d(split: str = "train") -> list[Subject]:
    """3D vessel dataset.

    Args:
        split: ``'train'``, ``'val'``, or ``'test'``.
    """
    return _load_medmnist("vesselmnist3d", split)


def synapse_mnist_3d(split: str = "train") -> list[Subject]:
    """3D synapse dataset.

    Args:
        split: ``'train'``, ``'val'``, or ``'test'``.
    """
    return _load_medmnist("synapsemnist3d", split)
