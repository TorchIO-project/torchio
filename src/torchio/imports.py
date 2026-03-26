"""Helpers for optional dependency imports."""

from __future__ import annotations

from importlib import import_module
from importlib.util import find_spec
from types import ModuleType


def _check_module(*, module: str, extra: str, package: str | None = None) -> None:
    if find_spec(module) is None:
        name = module if package is None else package
        msg = (
            f"The `{name}` package is required for this."
            f" Install TorchIO with the `{extra}` extra:"
            f" `pip install torchio[{extra}]`."
        )
        raise ImportError(msg)


def _check_and_import(module: str, extra: str, **kwargs: object) -> ModuleType:
    _check_module(module=module, extra=extra, **kwargs)
    return import_module(module)


def get_niizarr() -> ModuleType:
    return _check_and_import(
        module="niizarr",
        extra="zarr",
        package="nifti-zarr",
    )
