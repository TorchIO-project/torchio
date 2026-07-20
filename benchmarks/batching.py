"""Benchmark core TorchIO batching operations."""

from __future__ import annotations

import copy
import time
from collections.abc import Callable

import torch
from rich.console import Console
from rich.table import Table

import torchio as tio

_BATCH_SIZE = 8
_SPATIAL_SHAPE = 64, 64, 64
_ITERATIONS = 10


def _make_subjects(
    batch_size: int,
    spatial_shape: tuple[int, int, int],
    device: torch.device,
) -> list[tio.Subject]:
    """Build representative image and metadata subjects."""
    data = torch.rand(1, *spatial_shape, device=device)
    return [
        tio.Subject(
            image=tio.ScalarImage(data.clone()),
            age=40 + index,
        )
        for index in range(batch_size)
    ]


def _measure(operation: Callable[[], object], iterations: int) -> float:
    """Return average operation time in milliseconds."""
    for _ in range(2):
        operation()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(iterations):
        operation()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    return (time.perf_counter() - start) * 1000 / iterations


def _benchmark_device(device: torch.device) -> dict[str, float]:
    """Benchmark batching operations on one device."""
    subjects = _make_subjects(_BATCH_SIZE, _SPATIAL_SHAPE, device)
    batch = tio.SubjectsBatch.from_subjects(subjects)
    transform = tio.Gamma(log_gamma=(0.2, 0.8))
    uniform_transformed = tio.Gamma(log_gamma=0.3, per_instance=False)(batch)
    metadata_subjects = [tio.Subject(age=index) for index in range(_BATCH_SIZE)]

    return {
        "construct": _measure(
            lambda: tio.SubjectsBatch.from_subjects(subjects),
            _ITERATIONS,
        ),
        "unbatch": _measure(batch.unbatch, _ITERATIONS),
        "vectorized transform": _measure(
            lambda: transform(copy.deepcopy(batch)),
            _ITERATIONS,
        ),
        "uniform inverse": _measure(
            uniform_transformed.apply_inverse_transform,
            _ITERATIONS,
        ),
        "metadata-only": _measure(
            lambda: tio.SubjectsBatch.from_subjects(metadata_subjects),
            _ITERATIONS,
        ),
    }


def main() -> None:
    """Print CPU and optional CUDA batching benchmarks."""
    devices = [torch.device("cpu")]
    if torch.cuda.is_available():
        devices.append(torch.device("cuda"))

    table = Table(title="TorchIO batching benchmarks")
    table.add_column("Operation")
    for device in devices:
        table.add_column(f"{device.type} (ms)", justify="right")

    results = {device.type: _benchmark_device(device) for device in devices}
    for operation in results["cpu"]:
        table.add_row(
            operation,
            *(f"{results[device.type][operation]:.2f}" for device in devices),
        )
    Console().print(table)


if __name__ == "__main__":
    main()
