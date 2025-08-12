from __future__ import annotations

from importlib import import_module
from importlib.util import find_spec
from shutil import which
from types import ModuleType


def _check_module(*, module: str, extra: str, package: str | None = None) -> None:
    if find_spec(module) is None:
        name = module if package is None else package
        message = (
            f'The `{name}` package is required for this.'
            f' Install TorchIO with the `{extra}` extra:'
            f' `pip install torchio[{extra}]`.'
        )
        raise ImportError(message)


def _check_and_import(module: str, extra: str, **kwargs) -> ModuleType:
    _check_module(module=module, extra=extra, **kwargs)
    return import_module(module)


def get_pandas() -> ModuleType:
    return _check_and_import(module='pandas', extra='csv')


def get_colorcet() -> ModuleType:
    return _check_and_import(module='colorcet', extra='plot')


def get_ffmpeg() -> ModuleType:
    ffmpeg = _check_and_import(module='ffmpeg', extra='video', package='ffmpeg-python')
    _check_executable('ffmpeg')
    return ffmpeg


def get_sklearn() -> ModuleType:
    return _check_and_import(module='sklearn', extra='sklearn', package='scikit-learn')


def _check_executable(executable: str) -> None:
    if which(executable) is None:
        message = (
            f'The `{executable}` executable is required for this. Install it from your'
            ' package manager or download it from the official website.'
        )
        raise FileNotFoundError(message)
