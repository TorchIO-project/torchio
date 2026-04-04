"""Visualization utilities for TorchIO images.

Requires the ``[plot]`` extras: ``pip install torchio[plot]``.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING
from typing import Any
from typing import Literal
from typing import cast
from typing import overload

import numpy as np

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.colors import Colormap
    from matplotlib.figure import Figure

    from .data.image import Image
    from .types import TypePath

# Opposite anatomical direction for each orientation code.
_OPPOSITE: dict[str, str] = {
    "R": "L",
    "L": "R",
    "A": "P",
    "P": "A",
    "S": "I",
    "I": "S",
}

# Map orientation code to the anatomical view obtained by slicing that axis.
_VIEW_NAME: dict[str, str] = {
    "R": "Sagittal",
    "L": "Sagittal",
    "A": "Coronal",
    "P": "Coronal",
    "S": "Axial",
    "I": "Axial",
}


def _get_mpl():
    """Lazy-import matplotlib, raising a helpful error if missing."""
    try:
        import matplotlib
        import matplotlib.pyplot as plt
    except ImportError:
        msg = (
            "matplotlib is required for plotting. "
            "Install it with: pip install torchio[plot]"
        )
        raise ImportError(msg) from None
    return matplotlib, plt


def _get_categorical_cmap(
    slices_2d: list[np.ndarray],
    cmap_name: str = "glasbey_category10",
) -> Any:
    """Build a categorical ListedColormap for label maps.

    Uses colorcet if available, otherwise falls back to matplotlib's
    ``tab10`` colors.
    """
    from itertools import cycle

    mpl, _ = _get_mpl()
    num_classes = max(int(s.max()) for s in slices_2d)
    colors: list[tuple[float, ...]] = [
        (0.0, 0.0, 0.0),  # black for background
        (1.0, 1.0, 1.0),  # white for class 1
    ]
    if num_classes > 1:
        from .external.imports import get_colorcet

        cc = get_colorcet()
        if cc is not None:
            cc_cmap = getattr(cc.cm, cmap_name)
            color_cycle = cycle(cc_cmap.colors)
        else:
            tab10 = mpl.colormaps["tab10"]
            color_cycle = cycle(tab10.colors)
        colors.extend(next(color_cycle) for _ in range(num_classes - 1))
    boundaries = np.arange(-0.5, num_classes + 1.5, 1)
    colormap = mpl.colors.ListedColormap(colors)
    norm = mpl.colors.BoundaryNorm(boundaries, ncolors=colormap.N)
    return colormap, norm


@overload
def plot_image(
    image: Image,
    *,
    show: Literal[False],
    channel: int = ...,
    indices: tuple[int | None, int | None, int | None] | None = ...,
    axes: Sequence[Axes] | None = ...,
    cmap: str | Colormap | None = ...,
    percentiles: tuple[float, float] = ...,
    figsize: tuple[float, float] | None = ...,
    title: str | None = ...,
    output_path: TypePath | None = ...,
    savefig_kwargs: dict[str, Any] | None = ...,
    **imshow_kwargs: Any,
) -> Figure: ...


@overload
def plot_image(
    image: Image,
    *,
    show: Literal[True] = ...,
    channel: int = ...,
    indices: tuple[int | None, int | None, int | None] | None = ...,
    axes: Sequence[Axes] | None = ...,
    cmap: str | Colormap | None = ...,
    percentiles: tuple[float, float] = ...,
    figsize: tuple[float, float] | None = ...,
    title: str | None = ...,
    output_path: TypePath | None = ...,
    savefig_kwargs: dict[str, Any] | None = ...,
    **imshow_kwargs: Any,
) -> None: ...


def plot_image(
    image: Image,
    *,
    channel: int = 0,
    indices: tuple[int | None, int | None, int | None] | None = None,
    axes: Sequence[Axes] | None = None,
    cmap: str | Colormap | None = None,
    percentiles: tuple[float, float] = (0.5, 99.5),
    figsize: tuple[float, float] | None = None,
    title: str | None = None,
    output_path: TypePath | None = None,
    show: bool = True,
    savefig_kwargs: dict[str, Any] | None = None,
    **imshow_kwargs: Any,
) -> Figure | None:
    """Plot 3 orthogonal slices of a 3D image.

    Each subplot shows a slice through one spatial axis. Orientation
    labels are derived from the image's affine — no reorientation is
    applied. The horizontal axis is flipped so that, e.g., the left
    hemisphere appears on the right side of the screen (radiological
    convention). Uses lazy ``Image.__getitem__`` so only the 3
    requested planes are read from disk.

    Args:
        image: The image to plot.
        channel: Which channel to display.
        indices: Slice index for each spatial axis. ``None`` entries
            default to the mid-slice. Pass ``None`` for all mid-slices.
        axes: Pre-created sequence of 3 matplotlib ``Axes``. If
            ``None``, a new figure with 3 subplots is created.
        cmap: Colormap. Defaults to ``'gray'`` for intensity images.
        percentiles: Intensity percentile range for display windowing.
            Ignored for label maps.
        figsize: Figure size in inches ``(width, height)``.
        title: Figure super-title.
        output_path: Save figure to this path.
        show: Call ``plt.show()`` after plotting.
        savefig_kwargs: Extra keyword arguments for ``fig.savefig()``.
        **imshow_kwargs: Forwarded to ``ax.imshow()``.

    Returns:
        The matplotlib ``Figure``, or ``None`` when ``show=True``
        (the figure is displayed and closed to prevent duplicate
        rendering in notebooks).
    """
    from .data.image import LabelMap

    _, plt = _get_mpl()

    # Read spatial metadata from headers (no data load)
    spatial_shape = image.spatial_shape
    spacing = image.spacing
    orientation = image.orientation

    # Resolve indices — None → mid-slice
    if indices is None:
        indices = (None, None, None)
    resolved = tuple(
        s // 2 if idx is None else idx
        for idx, s in zip(indices, spatial_shape, strict=True)
    )

    # Extract 2D slices via lazy Image.__getitem__
    slices_2d: list[np.ndarray] = []
    for axis in range(3):
        sl: list[slice | int] = [slice(None), slice(None), slice(None), slice(None)]
        sl[0] = channel
        sl[axis + 1] = resolved[axis]
        plane = image[tuple(sl)]
        slices_2d.append(plane.data.squeeze().cpu().numpy())

    # Set up imshow defaults
    is_label = isinstance(image, LabelMap)
    if cmap is None:
        if is_label:
            cmap, norm = _get_categorical_cmap(slices_2d)
            imshow_kwargs.setdefault("norm", norm)
        else:
            cmap = "gray"
    imshow_kwargs.setdefault("cmap", cmap)
    imshow_kwargs["origin"] = "lower"
    if is_label:
        imshow_kwargs.setdefault("interpolation", "none")
    else:
        imshow_kwargs.setdefault("interpolation", "bilinear")

    # Percentile windowing (skip for label maps)
    if not is_label:
        all_values = np.concatenate([s.ravel() for s in slices_2d])
        vmin, vmax = np.percentile(all_values, percentiles)
        imshow_kwargs.setdefault("vmin", vmin)
        imshow_kwargs.setdefault("vmax", vmax)

    # Create figure if needed
    fig: Figure
    if axes is None:
        fig, ax_array = plt.subplots(1, 3, figsize=figsize)
        plot_axes: Sequence[Axes] = list(ax_array.flat)
    else:
        if len(axes) < 3:
            msg = f"Expected 3 axes, got {len(axes)}"
            raise ValueError(msg)
        plot_axes = axes
        fig = cast("Figure", plot_axes[0].get_figure())

    for axis in range(3):
        ax = plot_axes[axis]
        data_2d = slices_2d[axis]

        # The two remaining axes (row=first remaining, col=second remaining)
        remaining = [d for d in range(3) if d != axis]
        row_dim, col_dim = remaining[0], remaining[1]

        # Aspect ratio from spacing
        aspect = spacing[row_dim] / spacing[col_dim]

        ax.imshow(data_2d.T, aspect=aspect, **imshow_kwargs)

        # Flip horizontal axis (radiological convention)
        ax.invert_xaxis()

        # Orientation labels
        # x-axis is flipped: positive direction is on the left
        row_code = orientation[row_dim]
        col_code = orientation[col_dim]
        ax.set_xlabel(f"{row_code} ← {_OPPOSITE[row_code]}")
        ax.set_ylabel(f"{_OPPOSITE[col_code]} → {col_code}")

        # View name from the sliced axis
        view = _VIEW_NAME.get(orientation[axis], f"Axis {axis}")
        ax.set_title(f"{view} [{resolved[axis]}]")

    if title is not None:
        fig.suptitle(title)
    fig.tight_layout()

    if output_path is not None:
        fig.savefig(output_path, **(savefig_kwargs or {}))
    if show:
        plt.show()
        plt.close(fig)
        return None

    return fig
