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

# Each view is defined by:
#   (name, slice_pair, x_pair, y_pair, x_positive_on_left, y_positive_on_top)
# "pair" means the L/R, A/P, or S/I axis pair.
# x_positive_on_left / y_positive_on_top: the code that should appear on the
# left side (x) or top (y) of the display.
_VIEWS: list[tuple[str, str, str, str, str, str]] = [
    ("Sagittal", "LR", "AP", "SI", "A", "S"),
    ("Coronal", "AP", "LR", "SI", "R", "S"),
    ("Axial", "SI", "LR", "AP", "R", "A"),
]

# Intersection line colors (from 3D Slicer, via v1).
# Each color identifies the slice position being shown.
_COLOR_SAGITTAL = "#42A5F5"  # blue
_COLOR_CORONAL = "#8FE561"  # green
_COLOR_AXIAL = "#FF8372"  # red

# Map view name to its intersection color.
_VIEW_COLOR: dict[str, str] = {
    "Sagittal": _COLOR_SAGITTAL,
    "Coronal": _COLOR_CORONAL,
    "Axial": _COLOR_AXIAL,
}
_CODE_TO_PAIR: dict[str, str] = {
    "R": "LR",
    "L": "LR",
    "A": "AP",
    "P": "AP",
    "S": "SI",
    "I": "SI",
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


def _find_axis(orientation: tuple[str, str, str], pair: str) -> int:
    """Find which tensor axis (0, 1, 2) corresponds to an anatomical pair."""
    for i, code in enumerate(orientation):
        if _CODE_TO_PAIR[code] == pair:
            return i
    msg = f"No axis found for pair {pair!r} in orientation {orientation}"
    raise ValueError(msg)


@overload
def plot_image(
    image: Image,
    *,
    show: Literal[False],
    channel: int = ...,
    indices: tuple[int | None, int | None, int | None] | None = ...,
    coordinates: tuple[float | None, float | None, float | None] | None = ...,
    axes: Sequence[Axes] | None = ...,
    cmap: str | Colormap | None = ...,
    percentiles: tuple[float, float] = ...,
    figsize: tuple[float, float] | None = ...,
    title: str | None = ...,
    output_path: TypePath | None = ...,
    savefig_kwargs: dict[str, Any] | None = ...,
    voxels: bool = ...,
    figsize_multiplier: float = ...,
    intersections: bool = ...,
    **imshow_kwargs: Any,
) -> Figure: ...


@overload
def plot_image(
    image: Image,
    *,
    show: Literal[True] = ...,
    channel: int = ...,
    indices: tuple[int | None, int | None, int | None] | None = ...,
    coordinates: tuple[float | None, float | None, float | None] | None = ...,
    axes: Sequence[Axes] | None = ...,
    cmap: str | Colormap | None = ...,
    percentiles: tuple[float, float] = ...,
    figsize: tuple[float, float] | None = ...,
    title: str | None = ...,
    output_path: TypePath | None = ...,
    savefig_kwargs: dict[str, Any] | None = ...,
    voxels: bool = ...,
    figsize_multiplier: float = ...,
    intersections: bool = ...,
    **imshow_kwargs: Any,
) -> None: ...


def plot_image(
    image: Image,
    *,
    channel: int = 0,
    indices: tuple[int | None, int | None, int | None] | None = None,
    coordinates: tuple[float | None, float | None, float | None] | None = None,
    axes: Sequence[Axes] | None = None,
    cmap: str | Colormap | None = None,
    percentiles: tuple[float, float] = (0.5, 99.5),
    figsize: tuple[float, float] | None = None,
    title: str | None = None,
    output_path: TypePath | None = None,
    show: bool = True,
    savefig_kwargs: dict[str, Any] | None = None,
    voxels: bool = False,
    figsize_multiplier: float = 2.0,
    intersections: bool = True,
    **imshow_kwargs: Any,
) -> Figure | None:
    """Plot 3 orthogonal slices of a 3D image.

    Always displays Sagittal, Coronal, Axial with fixed anatomical
    positions regardless of image orientation. Data is flipped and
    transposed as needed. Uses lazy ``Image.__getitem__`` so only
    the 3 requested planes are read from disk.

    Args:
        image: The image to plot.
        channel: Which channel to display.
        indices: Slice index for each spatial axis. ``None`` entries
            default to the mid-slice. Pass ``None`` for all mid-slices.
            Mutually exclusive with ``coordinates``.
        coordinates: World coordinates in mm for each slice.
            ``None`` entries default to the mid-slice. Converted to
            the nearest voxel index via the inverse affine. Mutually
            exclusive with ``indices``.
        axes: Pre-created sequence of 3 matplotlib ``Axes``. If
            ``None``, a new figure with correct proportions is created.
        cmap: Colormap. Defaults to ``'gray'`` for intensity images.
        percentiles: Intensity percentile range for display windowing.
            Ignored for label maps.
        figsize: Figure size in inches ``(width, height)``.
        title: Figure super-title.
        output_path: Save figure to this path.
        show: Call ``plt.show()`` after plotting.
        savefig_kwargs: Extra keyword arguments for ``fig.savefig()``.
        voxels: Show voxel indices on ticks instead of world
            coordinates in mm.
        figsize_multiplier: Scale factor applied to the default
            ``rcParams["figure.figsize"]`` when ``figsize`` is ``None``.
        intersections: Draw coloured cross-hair lines showing where
            the other two slices intersect each view.
        **imshow_kwargs: Forwarded to ``ax.imshow()``.

    Returns:
        The matplotlib ``Figure``, or ``None`` when ``show=True``
        (the figure is displayed and closed to prevent duplicate
        rendering in notebooks).
    """
    import torch

    from .data.image import LabelMap

    mpl, plt = _get_mpl()

    if indices is not None and coordinates is not None:
        msg = "indices and coordinates are mutually exclusive"
        raise ValueError(msg)

    # Read spatial metadata from headers (no data load)
    spatial_shape = image.spatial_shape
    spacing = image.spacing
    orientation = image.orientation
    origin = image.origin

    # Resolve to voxel indices
    if coordinates is not None:
        inv_affine = image.affine.inverse()
        voxel_coords = inv_affine.apply(
            torch.tensor(
                [[c if c is not None else float("nan") for c in coordinates]],
                dtype=torch.float64,
            ),
        )[0]
        c0, c1, c2 = coordinates
        indices = (
            None if c0 is None else round(float(voxel_coords[0])),
            None if c1 is None else round(float(voxel_coords[1])),
            None if c2 is None else round(float(voxel_coords[2])),
        )

    if indices is None:
        indices = (None, None, None)
    resolved = tuple(
        s // 2 if idx is None else idx
        for idx, s in zip(indices, spatial_shape, strict=True)
    )

    # Find tensor axis for each anatomical pair
    axis_for: dict[str, int] = {}
    for pair in ("LR", "AP", "SI"):
        axis_for[pair] = _find_axis(orientation, pair)

    # Extract 2D slices via lazy Image.__getitem__
    slices_2d: list[np.ndarray] = []
    for _view_name, slice_pair, x_pair, y_pair, x_left, y_top in _VIEWS:
        slice_axis = axis_for[slice_pair]
        x_axis = axis_for[x_pair]
        y_axis = axis_for[y_pair]

        # Slice through the appropriate axis
        sl: list[slice | int] = [slice(None), slice(None), slice(None), slice(None)]
        sl[0] = channel
        sl[slice_axis + 1] = resolved[slice_axis]
        plane = image[tuple(sl)]
        data_2d = plane.data.squeeze().cpu().numpy()

        # data_2d axes are the two remaining spatial axes in tensor order.
        # We need: rows = y_axis, cols = x_axis.
        # After slicing axis `slice_axis`, remaining axes are sorted.
        # If x_axis < y_axis: data_2d dims = (x, y) → transpose to (y, x)
        if x_axis < y_axis:
            data_2d = data_2d.T

        # Flip to match expected anatomical display
        # x: x_left should be at col 0 (left in imshow)
        # y: y_top should be at last row (top with origin='lower')
        x_code = orientation[x_axis]
        y_code = orientation[y_axis]
        if x_code == x_left:
            data_2d = np.flip(data_2d, axis=1)
        if y_code != y_top:
            data_2d = np.flip(data_2d, axis=0)

        slices_2d.append(np.ascontiguousarray(data_2d))

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

    # Compute physical extents for proportional subplot sizing
    lr_mm = spatial_shape[axis_for["LR"]] * spacing[axis_for["LR"]]
    ap_mm = spatial_shape[axis_for["AP"]] * spacing[axis_for["AP"]]
    width_ratios = [ap_mm, lr_mm, lr_mm]

    # Create figure if needed
    fig: Figure
    if axes is None:
        if figsize is None:
            default_w, default_h = plt.rcParams["figure.figsize"]
            figsize = (
                default_w * figsize_multiplier,
                default_h * figsize_multiplier,
            )
        gs = mpl.gridspec.GridSpec(1, 3, width_ratios=width_ratios)
        fig = plt.figure(figsize=figsize)
        plot_axes: Sequence[Axes] = [fig.add_subplot(gs[0, i]) for i in range(3)]
    else:
        if len(axes) < 3:
            msg = f"Expected 3 axes, got {len(axes)}"
            raise ValueError(msg)
        plot_axes = axes
        fig = cast("Figure", plot_axes[0].get_figure())

    # Plot each view
    for view_idx, (view_name, slice_pair, x_pair, y_pair, x_left, y_top) in enumerate(
        _VIEWS,
    ):
        ax = plot_axes[view_idx]
        data_2d = slices_2d[view_idx]

        slice_axis = axis_for[slice_pair]
        x_axis = axis_for[x_pair]
        y_axis = axis_for[y_pair]

        # Aspect ratio from spacing
        aspect = spacing[y_axis] / spacing[x_axis]
        ax.imshow(data_2d, aspect=aspect, **imshow_kwargs)

        # Axis labels: "J (A ↔ P)"
        x_code = orientation[x_axis]
        y_code = orientation[y_axis]
        x_label = f"{_axis_name(x_axis)} ({x_left} ↔ {_OPPOSITE[x_left]})"
        y_label = f"{_axis_name(y_axis)} ({_OPPOSITE[y_top]} ↔ {y_top})"
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)

        # Tick labels: world coordinates (mm) or voxel indices
        _set_ticks(
            ax,
            x_axis=x_axis,
            y_axis=y_axis,
            x_code=x_code,
            y_code=y_code,
            x_left=x_left,
            y_top=y_top,
            spacing=spacing,
            origin_mm=origin,
            spatial_shape=spatial_shape,
            voxels=voxels,
        )

        # Title
        ax.set_title(f"{view_name} [{resolved[slice_axis]}]")

    # Draw slice intersection lines
    if intersections:
        _draw_intersections(
            plot_axes,
            axis_for=axis_for,
            orientation=orientation,
            spatial_shape=spatial_shape,
            resolved=resolved,
        )

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


def _axis_name(axis: int) -> str:
    """Return the tensor axis name: I, J, or K."""
    return ("I", "J", "K")[axis]


def _draw_intersections(
    plot_axes: Sequence[Axes],
    *,
    axis_for: dict[str, int],
    orientation: tuple[str, str, str],
    spatial_shape: tuple[int, int, int],
    resolved: tuple[int, ...],
) -> None:
    """Draw coloured cross-hair lines showing slice positions."""
    for view_idx, (view_name, _slice_pair, x_pair, y_pair, x_left, y_top) in enumerate(
        _VIEWS,
    ):
        ax = plot_axes[view_idx]
        x_axis = axis_for[x_pair]
        y_axis = axis_for[y_pair]
        x_size = spatial_shape[x_axis]
        y_size = spatial_shape[y_axis]
        x_code = orientation[x_axis]
        y_code = orientation[y_axis]

        for other_name, other_slice_pair, _, _, _, _ in _VIEWS:
            if other_name == view_name:
                continue
            other_axis = axis_for[other_slice_pair]
            other_pos = resolved[other_axis]
            color = _VIEW_COLOR[other_name]

            if other_axis == x_axis:
                display_x = _display_pos(other_pos, x_size, x_code == x_left)
                ax.axvline(display_x, color=color, linewidth=0.8, alpha=0.8)
            elif other_axis == y_axis:
                display_y = _display_pos(other_pos, y_size, y_code != y_top)
                ax.axhline(display_y, color=color, linewidth=0.8, alpha=0.8)


def _display_pos(voxel: int, size: int, flipped: bool) -> float:
    """Convert a voxel index to display position, accounting for flips."""
    return float(size - 1 - voxel) if flipped else float(voxel)


def _set_ticks(
    ax: Axes,
    *,
    x_axis: int,
    y_axis: int,
    x_code: str,
    y_code: str,
    x_left: str,
    y_top: str,
    spacing: tuple[float, float, float],
    origin_mm: tuple[float, float, float],
    spatial_shape: tuple[int, int, int],
    voxels: bool,
) -> None:
    """Set tick labels for a subplot."""
    if voxels:
        # Show voxel indices
        x_size = spatial_shape[x_axis]
        y_size = spatial_shape[y_axis]

        # If the axis data was flipped, voxel indices run in reverse
        x_flipped = x_code == x_left
        y_flipped = y_code != y_top

        x_ticks = np.linspace(0, x_size - 1, min(5, x_size))
        y_ticks = np.linspace(0, y_size - 1, min(5, y_size))

        ax.set_xticks(x_ticks)
        ax.set_yticks(y_ticks)
        if x_flipped:
            ax.set_xticklabels([str(int(x_size - 1 - v)) for v in x_ticks])
        else:
            ax.set_xticklabels([str(int(v)) for v in x_ticks])
        if y_flipped:
            ax.set_yticklabels([str(int(y_size - 1 - v)) for v in y_ticks])
        else:
            ax.set_yticklabels([str(int(v)) for v in y_ticks])
    else:
        # Show world coordinates in mm
        x_size = spatial_shape[x_axis]
        y_size = spatial_shape[y_axis]

        x_flipped = x_code == x_left
        y_flipped = y_code != y_top

        x_sp = spacing[x_axis]
        y_sp = spacing[y_axis]
        x_sign = -1.0 if x_code in ("L", "P", "I") else 1.0
        y_sign = -1.0 if y_code in ("L", "P", "I") else 1.0
        x_origin = origin_mm[_world_dim(x_code)]
        y_origin = origin_mm[_world_dim(y_code)]

        x_ticks = np.linspace(0, x_size - 1, min(5, x_size))
        y_ticks = np.linspace(0, y_size - 1, min(5, y_size))

        ax.set_xticks(x_ticks)
        ax.set_yticks(y_ticks)

        def _voxel_to_mm(
            display_pos: float,
            size: int,
            flipped: bool,
            orig: float,
            sp: float,
            sign: float,
        ) -> float:
            voxel = (size - 1 - display_pos) if flipped else display_pos
            return orig + voxel * sp * sign

        ax.set_xticklabels(
            [
                f"{_voxel_to_mm(v, x_size, x_flipped, x_origin, x_sp, x_sign):.0f}"
                for v in x_ticks
            ]
        )
        ax.set_yticklabels(
            [
                f"{_voxel_to_mm(v, y_size, y_flipped, y_origin, y_sp, y_sign):.0f}"
                for v in y_ticks
            ]
        )


def _world_dim(code: str) -> int:
    """Map an orientation code to the world coordinate dimension (0=x, 1=y, 2=z)."""
    if code in ("R", "L"):
        return 0
    if code in ("A", "P"):
        return 1
    return 2
