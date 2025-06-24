from importlib import import_module
from importlib.util import find_spec
from types import ModuleType


def _check_package(*, package: str, extra: str) -> None:
    if find_spec(package) is None:
        message = (
            f'The `{package}` package is required for this.'
            f' Install TorchIO with the `{extra}` extra:'
            f' `pip install torchio[{extra}]`.'
        )
        raise ImportError(message)


def _check_and_import(package: str, extra: str) -> ModuleType:
    _check_package(package=package, extra=extra)
    return import_module(package)


def get_pandas() -> ModuleType:
    return _check_and_import(package='pandas', extra='csv')
