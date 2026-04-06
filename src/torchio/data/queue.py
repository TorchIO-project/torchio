"""Patch queue for efficient patch-based training."""

from __future__ import annotations

import random as _random
from collections import deque
from collections.abc import Iterator
from collections.abc import Sequence
from collections.abc import Sized
from concurrent.futures import Future
from concurrent.futures import ThreadPoolExecutor
from itertools import islice
from typing import Any

import humanize
from torch.utils.data import IterableDataset
from torch.utils.data import Sampler

from .sampler import PatchSampler
from .subject import Subject


class Queue(IterableDataset):
    """Buffer of patches for stochastic patch-based training.

    Loads and preprocesses subjects in background threads, extracts
    random patches via a sampler, and yields them one at a time.
    Designed for use with ``SubjectsLoader`` or ``DataLoader``.

    Args:
        subjects: Sequence of subjects to sample patches from.
        patch_sampler: A sampler (e.g.,
            [`UniformSampler`][torchio.data.UniformSampler]) used to
            extract patches from loaded subjects. The sampler must
            accept a subject and ``num_patches`` in its ``__call__``.
        max_length: Maximum number of patches held in the buffer.
            Larger values increase diversity but use more RAM.
        patches_per_volume: Maximum patches to extract from each
            subject. The sampler may yield fewer if valid positions
            are exhausted.
        num_workers: Number of background threads for loading and
            preprocessing subjects. Set to 0 for synchronous loading.
        shuffle_subjects: Shuffle the subject order at the start of
            each epoch.
        shuffle_patches: Shuffle the buffer after each refill.
        transform: Optional transform applied to each subject after
            loading and before patch extraction.
        subject_sampler: A ``torch.utils.data.Sampler`` (e.g.,
            ``DistributedSampler``) that yields subject indices.
            When provided, ``shuffle_subjects`` must be ``False``.

    Examples:
        >>> queue = tio.Queue(
        ...     subjects,
        ...     patch_sampler=tio.UniformSampler(subject, patch_size=64),
        ...     max_length=300,
        ...     patches_per_volume=10,
        ...     num_workers=4,
        ... )
        >>> loader = SubjectsLoader(queue, batch_size=16)
        >>> for batch in loader:
        ...     outputs = model(batch.t1.data)
    """

    def __init__(
        self,
        subjects: Sequence[Subject],
        patch_sampler: PatchSampler,
        max_length: int = 300,
        patches_per_volume: int = 10,
        num_workers: int = 0,
        shuffle_subjects: bool = True,
        shuffle_patches: bool = True,
        transform: Any | None = None,
        subject_sampler: Sampler | None = None,
    ) -> None:
        if subject_sampler is not None and shuffle_subjects:
            msg = (
                "shuffle_subjects must be False when subject_sampler"
                " is provided (the sampler controls the order)"
            )
            raise ValueError(msg)
        self.subjects = subjects
        self.patch_sampler = patch_sampler
        self.max_length = max_length
        self.patches_per_volume = patches_per_volume
        self.num_workers = num_workers
        self.shuffle_subjects = shuffle_subjects
        self.shuffle_patches = shuffle_patches
        self.transform = transform
        self.subject_sampler = subject_sampler

    def __iter__(self) -> Iterator[Subject]:
        """Yield patches, loading subjects in the background."""
        buffer: list[Subject] = []
        subject_iter = self._make_subject_iter()

        if self.num_workers > 0:
            yield from self._iter_threaded(subject_iter, buffer)
        else:
            yield from self._iter_sync(subject_iter, buffer)

    def _iter_sync(
        self,
        subject_iter: Iterator[Subject],
        buffer: list[Subject],
    ) -> Iterator[Subject]:
        for raw in subject_iter:
            prepared = self._prepare(raw)
            buffer.extend(self._sample_patches(prepared))
            yield from self._drain_if_full(buffer)
        yield from self._flush(buffer)

    def _iter_threaded(
        self,
        subject_iter: Iterator[Subject],
        buffer: list[Subject],
    ) -> Iterator[Subject]:
        with ThreadPoolExecutor(max_workers=self.num_workers) as pool:
            futures: deque[Future[Subject]] = deque()

            for raw in subject_iter:
                futures.append(pool.submit(self._prepare, raw))
                yield from self._collect_ready(futures, buffer)
                yield from self._drain_if_full(buffer)

            # Drain remaining futures
            for future in futures:
                prepared = future.result()
                buffer.extend(self._sample_patches(prepared))

        yield from self._flush(buffer)

    def _collect_ready(
        self,
        futures: deque[Future[Subject]],
        buffer: list[Subject],
    ) -> Iterator[Subject]:
        """Move patches from completed futures into the buffer."""
        while futures and futures[0].done():
            prepared = futures.popleft().result()
            buffer.extend(self._sample_patches(prepared))
        return iter(())  # nothing to yield yet

    def _drain_if_full(self, buffer: list[Subject]) -> Iterator[Subject]:
        """Yield all patches from buffer if it reached max_length."""
        if len(buffer) >= self.max_length:
            yield from self._flush(buffer)

    def _flush(self, buffer: list[Subject]) -> Iterator[Subject]:
        """Shuffle (if enabled) and yield all patches from buffer."""
        if self.shuffle_patches:
            _random.shuffle(buffer)
        while buffer:
            yield buffer.pop()

    def _prepare(self, subject: Subject) -> Subject:
        """Load images and apply transform (may run in a thread)."""
        subject.load()
        if self.transform is not None:
            subject = self.transform(subject)
        return subject

    def _sample_patches(self, subject: Subject) -> list[Subject]:
        """Extract up to patches_per_volume patches."""
        gen = iter(self.patch_sampler(subject))
        return list(islice(gen, self.patches_per_volume))

    def _make_subject_iter(self) -> Iterator[Subject]:
        """Build the subject iterator for one epoch."""
        if self.subject_sampler is not None:
            indices = list(self.subject_sampler)
            return (self.subjects[i] for i in indices)
        subjects = list(self.subjects)
        if self.shuffle_subjects:
            _random.shuffle(subjects)
        return iter(subjects)

    @property
    def num_subjects(self) -> int:
        """Number of subjects per epoch."""
        sampler = self.subject_sampler
        if sampler is not None:
            if not isinstance(sampler, Sized):
                msg = "subject_sampler must have a __len__ method"
                raise TypeError(msg)
            return len(sampler)
        return len(self.subjects)

    @property
    def patches_per_epoch(self) -> int:
        """Total patches yielded per epoch (upper bound)."""
        return self.num_subjects * self.patches_per_volume

    @property
    def max_memory(self) -> int:
        """Estimated max RAM for the patch buffer in bytes."""
        sample = self.subjects[0]
        channels = sum(img.num_channels for img in sample.images.values())
        voxels = 1
        for s in self.patch_sampler.patch_size:
            voxels *= s
        return 4 * channels * voxels * self.max_length

    @property
    def max_memory_pretty(self) -> str:
        """Human-readable max memory estimate."""
        return humanize.naturalsize(self.max_memory, binary=True)
