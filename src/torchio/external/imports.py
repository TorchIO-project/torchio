from importlib.util import find_spec
from types import ModuleType


def get_pandas() -> ModuleType:
    _check_package('pandas', 'pandas')
    import pandas

    return pandas


def get_huggingface_hub() -> ModuleType:
    _check_package('huggingface_hub', 'huggingface')
    import huggingface_hub

    return huggingface_hub


def _check_package(package: str, extra: str) -> None:
    if find_spec(package) is None:
        message = (
            f'The `{package}` package is required for this.'
            f' Install TorchIO with the `{extra}` extra:'
            f' `pip install torchio[{extra}]`.'
        )
        raise ImportError(message)
