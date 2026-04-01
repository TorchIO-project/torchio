"""DataLoader wrappers for Subject and Image collation."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from torch.utils.data import DataLoader
from torch.utils.data import Dataset

from .data.batch import ImagesBatch
from .data.batch import SubjectsBatch


def collate_subjects(batch: Sequence[Any]) -> SubjectsBatch:
    """Collate a list of Subjects into a SubjectsBatch.

    Args:
        batch: Sequence of ``Subject`` instances.

    Returns:
        A ``SubjectsBatch`` with stacked 5D tensors.
    """
    return SubjectsBatch.from_subjects(list(batch))


def collate_images(batch: Sequence[Any]) -> ImagesBatch:
    """Collate a list of Images into an ImagesBatch.

    Args:
        batch: Sequence of ``Image`` instances.

    Returns:
        An ``ImagesBatch`` with a stacked 5D tensor.
    """
    return ImagesBatch.from_images(list(batch))


class SubjectsLoader(DataLoader):
    """DataLoader that returns ``SubjectsBatch`` instances.

    A thin wrapper around ``torch.utils.data.DataLoader`` that
    collates ``Subject`` instances into ``SubjectsBatch``.

    Args:
        dataset: A dataset that returns ``Subject`` instances.
        **kwargs: Passed to ``DataLoader.__init__``.

    Examples:
        >>> loader = tio.SubjectsLoader(dataset, batch_size=4)
        >>> batch = next(iter(loader))
        >>> batch.t1.data.shape
        torch.Size([4, 1, 256, 256, 176])
    """

    def __init__(self, dataset: Dataset, **kwargs: Any) -> None:
        if "collate_fn" in kwargs:
            msg = (
                "SubjectsLoader sets collate_fn automatically; "
                "pass a plain DataLoader if you need a custom collate_fn"
            )
            raise ValueError(msg)
        super().__init__(dataset, collate_fn=collate_subjects, **kwargs)


class ImagesLoader(DataLoader):
    """DataLoader that returns ``ImagesBatch`` instances.

    A thin wrapper around ``torch.utils.data.DataLoader`` that
    collates ``Image`` instances into ``ImagesBatch``.

    Args:
        dataset: A dataset that returns ``Image`` instances.
        **kwargs: Passed to ``DataLoader.__init__``.

    Examples:
        >>> loader = tio.ImagesLoader(dataset, batch_size=4)
        >>> batch = next(iter(loader))
        >>> batch.data.shape
        torch.Size([4, 1, 256, 256, 176])
    """

    def __init__(self, dataset: Dataset, **kwargs: Any) -> None:
        if "collate_fn" in kwargs:
            msg = (
                "ImagesLoader sets collate_fn automatically; "
                "pass a plain DataLoader if you need a custom collate_fn"
            )
            raise ValueError(msg)
        super().__init__(dataset, collate_fn=collate_images, **kwargs)


# Aliases for radiology users — see Subject/Study note in subject.py.
StudiesLoader = SubjectsLoader
collate_studies = collate_subjects
