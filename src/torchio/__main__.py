from __future__ import annotations

import importlib
import sys
from pathlib import Path
from typing import Annotated
from typing import Any

import torch
import typer

from torchio import transforms
from torchio.transforms.transform import Transform
from torchio.utils import guess_type

app = typer.Typer()


@app.command(context_settings={'allow_extra_args': True})
def main(
    input_path: Annotated[
        Path,
        typer.Argument(
            ...,
            exists=True,
            file_okay=True,
            dir_okay=True,
            readable=True,
            help=(
                'Path to the input image. If a text file (.txt) is given, process all'
                ' paths specified in the file. If a directory is given, process all'
                ' files in the directory.'
            ),
        ),
    ],
    context: typer.Context,
    output_path: Annotated[
        Path,
        typer.Option(
            '--output',
            file_okay=True,
            dir_okay=True,
            writable=True,
            help=(
                'Path to the input image. If a text file (.txt) is given, process all'
                ' paths specified in the file. If a directory is given, process all'
                ' files in the directory.'
            ),
        ),
    ],
    # output_path,
    # overwrite,
    flatten: Annotated[
        bool,
        typer.Option(),
    ] = None,
    recursive: Annotated[
        bool,
        typer.Option(
            '--recursive/--no-recursive',
            '-r/-R',
        ),
    ] = False,
    transform: Annotated[
        str,
        typer.Option(),
    ] = None,
    source: Annotated[
        str,
        typer.Option(),
    ] = None,  # path to yaml, path to Python, github/repo:ref
    # flatten,  # if true, ignore the rel paths of the inputs to the input dir and write all files to output dir
):
    input_paths, output_paths = _get_paths(
        input_path,
        output_path,
        flatten=flatten,
        recursive=recursive,
    )
    if transform is None:
        transform = lambda x: x
    else:
        transform = _instantiate_transform(transform, source, context)
    # _process_images(input_paths, output_paths, transform, print_info=info)


def _get_input_paths(
    input_path: Path,
    *,
    recursive: bool,
) -> tuple[list[Path], bool]:
    is_dir = input_path.is_dir()
    if recursive and not is_dir():
        raise ValueError

    if is_dir:
        if recursive:
            paths = sorted(input_path.rglob('*'))
        else:
            paths = sorted(input_path.iterdir())
    elif input_path.suffix == '.txt':
        paths = input_path.read_text().splitlines()
    else:
        paths = [input_path]
    return paths, is_dir


def _get_paths(
    input_path: Path,
    output_path: Path | None,
    *,
    flatten: bool,
    recursive: bool,
) -> tuple[list[Path], list[Path] | None]:
    input_paths, input_is_dir = _get_input_paths(input_path, recursive=recursive)
    if input_is_dir:
        input_dir = input_path

    if output_path is None:
        return input_paths, None

    if output_path.is_dir():  # what if it's None?
        output_dir = output_path
        for input_path in input_paths:
            if flatten:
                output_path = output_dir / input_path.name
            else:
                if not input_is_dir:
                    raise
                output_path = output_dir / input_path.relative_to(input_dir)
    else:
        output_paths = [output_path]
    return input_paths, output_paths


def _instantiate_transform(
    transform_name: str | None,
    source: str | None,
    context: typer.Context,
) -> Transform:
    if source is None:
        transform = _instantiate_native_transform(transform_name, context)
    elif _is_yaml(source):
        transform = _instantiate_from_config(source)
    elif _is_python(source):
        _instantiate_from_module(source, transform_name)
    else:
        transform = torch.hub.load(source, transform_name, trust_repo=True)
    return transform


def _instantiate_native_transform(
    transform_name: str, context: typer.Context
) -> Transform:
    try:
        transform_class = getattr(transforms, transform_name)
    except AttributeError as error:
        message = f'Transform "{transform_name}" not found in torchio'
        raise ValueError(message) from error
    params_dict = _get_kwargs_from_context(context)
    transform = transform_class(**params_dict)
    return transform


def _instantiate_from_config(config_path: str) -> Transform:
    try:
        from hydra.utils import instantiate
    except ImportError as e:
        msg = (
            'hydra is needed to instantiate a transform as was not found. Install'
            ' the hydra extra with: pip install torchio[hydra]'
        )
        raise ImportError(msg) from e
    return instantiate(config_path)


def _instantiate_from_module(module_path: str, transform_name: str):
    module_path = Path(module_path)
    parent_path = str(module_path.parent)
    sys.path.append(parent_path)
    module = importlib.import_module(module_path.stem)
    transform = getattr(module, transform_name)
    sys.path.pop()
    return transform


def _is_yaml(path: str) -> bool:
    return Path(path).suffix in ('.yaml', '.yml')


def _is_python(path: str) -> bool:
    return Path(path).suffix == '.py'


def _get_kwargs_from_context(context: typer.Context) -> dict[str, Any]:
    kwargs = {}
    for arg in context.args:
        key, value = arg.split('=')
        kwargs[key] = guess_type(value)
    return kwargs


if __name__ == '__main__':
    app()
