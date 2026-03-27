"""DataLoader wrappers for Subject and Image collation via tensordict."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import torch
from tensordict import TensorDict
from torch.utils.data import DataLoader
from torch.utils.data import Dataset


def collate_subjects(batch: Sequence[Any]) -> TensorDict:
    """Collate a list of Subjects (or TensorDicts) into a batched TensorDict.

    Each element is converted via ``Subject.to_tensordict()`` if it is not
    already a ``TensorDict``, then the list is stacked with
    ``torch.stack``.

    Args:
        batch: Sequence of ``Subject`` or ``TensorDict`` instances.

    Returns:
        Stacked ``TensorDict`` with ``batch_size=[N]``.
    """
    from .data.subject import Subject

    tds = []
    for item in batch:
        if isinstance(item, Subject):
            tds.append(item.to_tensordict())
        elif isinstance(item, TensorDict):
            tds.append(item)
        else:
            msg = (
                f"collate_subjects expects Subject or TensorDict, "
                f"got {type(item).__name__}"
            )
            raise TypeError(msg)
    return torch.stack(tds)


def collate_images(batch: Sequence[Any]) -> TensorDict:
    """Collate a list of Images (or TensorDicts) into a batched TensorDict.

    Each element is converted via ``Image.to_tensordict()`` if it is not
    already a ``TensorDict``, then the list is stacked with
    ``torch.stack``.

    Args:
        batch: Sequence of ``Image`` or ``TensorDict`` instances.

    Returns:
        Stacked ``TensorDict`` with ``batch_size=[N]``.
    """
    from .data.image import Image

    tds = []
    for item in batch:
        if isinstance(item, Image):
            tds.append(item.to_tensordict())
        elif isinstance(item, TensorDict):
            tds.append(item)
        else:
            msg = (
                f"collate_images expects Image or TensorDict, got {type(item).__name__}"
            )
            raise TypeError(msg)
    return torch.stack(tds)


class SubjectsLoader(DataLoader):
    """DataLoader that automatically collates Subjects via tensordict.

    A thin wrapper around ``torch.utils.data.DataLoader`` that sets
    ``collate_fn=collate_subjects``.  All standard ``DataLoader``
    keyword arguments are forwarded.

    Args:
        dataset: A dataset that returns ``Subject`` instances.
        **kwargs: Passed to ``DataLoader.__init__``.

    Examples:
        >>> loader = tio.SubjectsLoader(dataset, batch_size=4)
        >>> batch = next(iter(loader))
        >>> batch["t1", "data"].shape
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
    """DataLoader that automatically collates Images via tensordict.

    A thin wrapper around ``torch.utils.data.DataLoader`` that sets
    ``collate_fn=collate_images``.  All standard ``DataLoader``
    keyword arguments are forwarded.

    Args:
        dataset: A dataset that returns ``Image`` instances.
        **kwargs: Passed to ``DataLoader.__init__``.

    Examples:
        >>> loader = tio.ImagesLoader(dataset, batch_size=4)
        >>> batch = next(iter(loader))
        >>> batch["data"].shape
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
