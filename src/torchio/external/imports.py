"""Helpers for optional dependency imports."""

from __future__ import annotations

from importlib import import_module
from importlib.util import find_spec
from types import ModuleType
from typing import Any


def _check_module(*, module: str, extra: str, package: str | None = None) -> None:
    if find_spec(module) is None:
        name = module if package is None else package
        msg = (
            f"The `{name}` package is required for this."
            f" Install TorchIO with the `{extra}` extra:"
            f" `pip install torchio[{extra}]`."
        )
        raise ImportError(msg)


def _check_and_import(module: str, extra: str, **kwargs: Any) -> ModuleType:
    _check_module(module=module, extra=extra, **kwargs)
    return import_module(module)


def get_niizarr() -> ModuleType:
    return _check_and_import(
        module="niizarr",
        extra="zarr",
        package="nifti-zarr",
    )


def get_matplotlib() -> ModuleType:
    return _check_and_import(module="matplotlib", extra="plot")


def get_matplotlib_pyplot() -> ModuleType:
    _check_module(module="matplotlib", extra="plot")
    return import_module("matplotlib.pyplot")


def get_colorcet() -> ModuleType | None:
    """Return colorcet if installed, else None (fallback to tab10)."""
    if find_spec("colorcet") is None:
        return None
    return import_module("colorcet")


def get_pillow() -> ModuleType:
    return _check_and_import(module="PIL", extra="plot", package="Pillow")


def get_ffmpeg() -> ModuleType:
    return _check_and_import(module="ffmpeg", extra="video", package="ffmpeg-python")


def get_monai() -> ModuleType:
    return _check_and_import(module="monai", extra="monai")
