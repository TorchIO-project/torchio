"""TorchIO command-line interface."""

from __future__ import annotations

import ast
import shutil
import sys
from dataclasses import dataclass
from dataclasses import field
from pathlib import Path
from typing import Annotated
from typing import Union

import tyro

import torchio as tio
from torchio.download import get_torchio_cache_dir

# ---------------------------------------------------------------------------
# Subcommands
# ---------------------------------------------------------------------------


@dataclass
class Plot:
    """Plot 3 orthogonal slices of an image."""

    path: Annotated[Path, tyro.conf.Positional]
    """Path to the image file."""

    channel: int = 0
    """Channel index to display."""

    output: Path | None = None
    """Save the figure to a file instead of displaying."""

    indices: tuple[int, int, int] | None = None
    """Slice indices (i, j, k). Defaults to mid-slices."""

    def run(self) -> None:
        image = tio.ScalarImage(self.path)
        show = self.output is None
        image.plot(
            channel=self.channel,
            indices=self.indices,
            output_path=self.output,
            show=show,
        )


@dataclass
class Animate:
    """Create an animated GIF or MP4 sweeping through slices.

    The output format is inferred from the file extension:
    ``.gif`` produces an animated GIF, ``.mp4`` produces a video.

    Examples::

        torchio animate brain.nii.gz brain.gif
        torchio animate brain.nii.gz brain.mp4 --seconds 10 --direction S
    """

    path: Annotated[Path, tyro.conf.Positional]
    """Path to the input image."""

    output: Annotated[Path, tyro.conf.Positional]
    """Output path (.gif or .mp4)."""

    seconds: float = 5.0
    """Duration of the animation in seconds."""

    direction: str = "I"
    """Anatomical sweep direction (I, S, A, P, R, or L)."""

    def run(self) -> None:
        image = tio.ScalarImage(self.path)
        suffix = self.output.suffix.lower()
        if suffix == ".gif":
            image.to_gif(
                self.output,
                seconds=self.seconds,
                direction=self.direction,
            )
        elif suffix == ".mp4":
            image.to_video(
                self.output,
                seconds=self.seconds,
                direction=self.direction,
            )
        else:
            msg = f"Unsupported output format {self.output.suffix!r}. Use .gif or .mp4."
            print(msg, file=sys.stderr)
            sys.exit(1)
        print(f"Created {self.output}")


@dataclass
class Info:
    """Print image metadata to stdout."""

    path: Annotated[Path, tyro.conf.Positional]
    """Path to the image file."""

    def run(self) -> None:
        image = tio.ScalarImage(self.path)
        print(repr(image))


@dataclass
class Convert:
    """Convert an image between formats.

    Supports all SimpleITK formats plus NIfTI-Zarr (.nii.zarr).
    The output format is inferred from the file extension.
    """

    input: Annotated[Path, tyro.conf.Positional]
    """Path to the input image."""

    output: Annotated[Path, tyro.conf.Positional]
    """Path for the output image."""

    def run(self) -> None:
        image = tio.ScalarImage(self.input)
        output_str = str(self.output)
        if output_str.endswith(".nii.zarr"):
            image.to_nifti_zarr(self.output)
        else:
            image.save(self.output)
        print(f"Converted {self.input} -> {self.output}")


@dataclass
class Transform:
    """Apply a transform to an image.

    Extra arguments are passed as key=value pairs to the transform.

    Examples::

        torchio transform brain.nii.gz noisy.nii.gz Noise std=0.1
        torchio transform brain.nii.gz cropped.nii.gz CropOrPad target_shape=128
    """

    input: Annotated[Path, tyro.conf.Positional]
    """Path to the input image."""

    output: Annotated[Path, tyro.conf.Positional]
    """Path for the output image."""

    name: Annotated[str, tyro.conf.Positional]
    """Transform class name (e.g., Noise, Flip, CropOrPad)."""

    device: str = "cpu"
    """Device to run the transform on (e.g., "cpu", "cuda", "cuda:0" or "mps")."""

    args: Annotated[list[str], tyro.conf.Positional] = field(
        default_factory=list,
    )
    """Extra arguments as key=value pairs (e.g., std=0.1)."""

    def run(self) -> None:
        transform_cls = _get_transform_class(self.name)
        kwargs = _parse_kwargs(self.args)
        transform = transform_cls(**kwargs)
        image = tio.ScalarImage(self.input).to(self.device)
        result = transform(image)
        result.save(self.output)


@dataclass
class Dir:
    """Print the cache directory path."""

    def run(self) -> None:
        print(get_torchio_cache_dir())


@dataclass
class Clean:
    """Clear cached data."""

    dataset: str | None = None
    """Dataset name to clear (e.g., 'colin27', 'fpg'). Clears all if omitted."""

    def run(self) -> None:
        cache_dir = get_torchio_cache_dir()
        if self.dataset is not None:
            target = cache_dir / self.dataset
            if not target.exists():
                print(f"No cached data for {self.dataset!r}")
                return
            shutil.rmtree(target)
            print(f"Cleared cache for {self.dataset!r}")
        else:
            if not cache_dir.exists():
                print("Cache is already empty")
                return
            shutil.rmtree(cache_dir)
            print(f"Cleared all cached data from {cache_dir}")


@dataclass
class Cache:
    """Manage the TorchIO data cache."""

    command: tyro.conf.OmitSubcommandPrefixes[Dir | Clean]

    def run(self) -> None:
        self.command.run()


Command = Union[Plot, Animate, Info, Convert, Transform, Cache]  # noqa: UP007 (tyro needs Union)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _get_transform_class(name: str) -> type:
    """Look up a transform class by name."""
    from torchio.transforms.transform import _TRANSFORM_REGISTRY

    if name not in _TRANSFORM_REGISTRY:
        available = sorted(_TRANSFORM_REGISTRY.keys())
        print(f"Unknown transform {name!r}.", file=sys.stderr)
        print(f"Available: {', '.join(available)}", file=sys.stderr)
        sys.exit(1)
    return _TRANSFORM_REGISTRY[name]


def _parse_kwargs(args: list[str]) -> dict[str, object]:
    """Parse key=value pairs into a dict, inferring Python types."""
    kwargs: dict[str, object] = {}
    for arg in args:
        if "=" not in arg:
            print(f"Invalid argument {arg!r}, expected key=value", file=sys.stderr)
            sys.exit(1)
        key, value_str = arg.split("=", 1)
        kwargs[key] = _parse_value(value_str)
    return kwargs


def _parse_value(value_str: str) -> object:
    """Parse a string value into a Python literal (int, float, bool, tuple, etc.)."""
    try:
        return ast.literal_eval(value_str)
    except (ValueError, SyntaxError):
        return value_str


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    """TorchIO CLI: tools for medical image processing."""
    cmd = tyro.cli(Command)  # ty: ignore[no-matching-overload]
    cmd.run()


if __name__ == "__main__":
    main()
