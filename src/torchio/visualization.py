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
    from .data.subject import Subject
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


def _display_figure(fig: Figure) -> None:
    """Display a figure inline (notebooks) or interactively (scripts)."""
    try:
        from IPython.display import display

        display(fig)
    except ImportError:
        import matplotlib.pyplot as plt

        plt.show()
        plt.close(fig)


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

        if (cc := get_colorcet()) is not None:
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


def _extract_slices(
    image: Image,
    channel: int,
    resolved: tuple[int, ...],
    axis_for: dict[str, int],
) -> list[np.ndarray]:
    """Extract oriented 2D slices for each anatomical view."""
    orientation = image.orientation
    slices_2d: list[np.ndarray] = []
    for _view_name, slice_pair, x_pair, y_pair, x_left, y_top in _VIEWS:
        slice_axis = axis_for[slice_pair]
        x_axis = axis_for[x_pair]
        y_axis = axis_for[y_pair]

        sl: list[slice | int] = [slice(None), slice(None), slice(None), slice(None)]
        sl[0] = channel
        sl[slice_axis + 1] = resolved[slice_axis]
        plane = image[tuple(sl)]
        data_2d = plane.data.squeeze().cpu().numpy()

        if x_axis < y_axis:
            data_2d = data_2d.T

        x_code = orientation[x_axis]
        y_code = orientation[y_axis]
        if x_code == x_left:
            data_2d = np.flip(data_2d, axis=1)
        if y_code != y_top:
            data_2d = np.flip(data_2d, axis=0)

        slices_2d.append(np.ascontiguousarray(data_2d))
    return slices_2d


def _resolve_label_colors(
    image: Image,
    cmap: str | Colormap | dict[int, tuple[int, int, int]] | None,
    slices_2d: list[np.ndarray],
    kw: dict[str, Any],
) -> tuple[list[np.ndarray], bool]:
    """If cmap is a color dict (or image carries one), colorize slices."""
    color_map: dict[int, tuple[int, int, int]] | None = None
    if isinstance(cmap, dict):
        color_map = cast("dict[int, tuple[int, int, int]]", cmap)
    elif cmap is None and hasattr(image, "color_map"):
        meta = image.color_map
        if isinstance(meta, dict):
            color_map = cast("dict[int, tuple[int, int, int]]", meta)

    if color_map is not None:
        slices_2d = _colorize_labels(slices_2d, color_map)
        kw["origin"] = "lower"
        kw.setdefault("interpolation", "none")
        return slices_2d, True
    return slices_2d, False


def _build_imshow_kwargs(
    image: Image,
    slices_2d: list[np.ndarray],
    cmap: str | Colormap | dict[int, tuple[int, int, int]] | None,
    percentiles: tuple[float, float],
    imshow_kwargs: dict[str, Any],
) -> tuple[dict[str, Any], list[np.ndarray]]:
    """Prepare the keyword arguments for ``ax.imshow()``.

    Returns the kwargs dict and (possibly RGB-converted) slices.
    """
    from .data.image import LabelMap

    kw = dict(imshow_kwargs)
    is_label = isinstance(image, LabelMap)

    slices_2d, colorized = _resolve_label_colors(image, cmap, slices_2d, kw)
    if colorized:
        return kw, slices_2d

    if cmap is None:
        if is_label:
            cmap, norm = _get_categorical_cmap(slices_2d)
            kw.setdefault("norm", norm)
        else:
            cmap = "gray"
    kw.setdefault("cmap", cmap)
    kw["origin"] = "lower"
    if is_label:
        kw.setdefault("interpolation", "none")
    else:
        kw.setdefault("interpolation", "bilinear")

    if not is_label:
        all_values = np.concatenate([s.ravel() for s in slices_2d])
        vmin, vmax = np.percentile(all_values, percentiles)
        kw.setdefault("vmin", vmin)
        kw.setdefault("vmax", vmax)
    return kw, slices_2d


def _colorize_labels(
    slices_2d: list[np.ndarray],
    color_map: dict[int, tuple[int, int, int]],
) -> list[np.ndarray]:
    """Convert label slices to RGB using a label-to-color mapping."""
    result: list[np.ndarray] = []
    for label_slice in slices_2d:
        h, w = label_slice.shape[:2]
        rgb = np.zeros((h, w, 3), dtype=np.uint8)
        for label, color in color_map.items():
            rgb[label_slice == label] = color
        result.append(rgb)
    return result


def _plot_image_on_axes(
    image: Image,
    plot_axes: Sequence[Axes],
    *,
    channel: int = 0,
    resolved: tuple[int, ...],
    cmap: str | Colormap | dict[int, tuple[int, int, int]] | None = None,
    percentiles: tuple[float, float] = (0.5, 99.5),
    voxels: bool = False,
    intersections: bool = True,
    show_titles: bool = True,
    **imshow_kwargs: Any,
) -> None:
    """Plot 3 orthogonal views of a single image onto pre-created axes."""
    spatial_shape = image.spatial_shape
    spacing = image.spacing
    orientation = image.orientation
    origin = image.origin

    axis_for: dict[str, int] = {}
    for pair in ("LR", "AP", "SI"):
        axis_for[pair] = _find_axis(orientation, pair)

    slices_2d = _extract_slices(image, channel, resolved, axis_for)
    kw, slices_2d = _build_imshow_kwargs(
        image,
        slices_2d,
        cmap,
        percentiles,
        imshow_kwargs,
    )

    for view_idx, (view_name, slice_pair, x_pair, y_pair, x_left, y_top) in enumerate(
        _VIEWS,
    ):
        ax = plot_axes[view_idx]

        slice_axis = axis_for[slice_pair]
        x_axis = axis_for[x_pair]
        y_axis = axis_for[y_pair]

        aspect = spacing[y_axis] / spacing[x_axis]
        ax.imshow(slices_2d[view_idx], aspect=aspect, **kw)

        x_label = f"{_axis_name(x_axis)} ({x_left} ↔ {_OPPOSITE[x_left]})"
        y_label = f"{_axis_name(y_axis)} ({_OPPOSITE[y_top]} ↔ {y_top})"
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)

        _set_ticks(
            ax,
            x_axis=x_axis,
            y_axis=y_axis,
            x_code=orientation[x_axis],
            y_code=orientation[y_axis],
            x_left=x_left,
            y_top=y_top,
            spacing=spacing,
            origin_mm=origin,
            spatial_shape=spatial_shape,
            voxels=voxels,
        )

        if show_titles:
            ax.set_title(f"{view_name} [{resolved[slice_axis]}]")

    if intersections:
        _draw_intersections(
            plot_axes,
            axis_for=axis_for,
            orientation=orientation,
            spatial_shape=spatial_shape,
            resolved=resolved,
        )


@overload
def plot_image(
    image: Image,
    *,
    show: Literal[False],
    channel: int = ...,
    indices: tuple[int | None, int | None, int | None] | None = ...,
    coordinates: tuple[float | None, float | None, float | None] | None = ...,
    axes: Sequence[Axes] | None = ...,
    cmap: str | Colormap | dict[int, tuple[int, int, int]] | None = ...,
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
    cmap: str | Colormap | dict[int, tuple[int, int, int]] | None = ...,
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
    cmap: str | Colormap | dict[int, tuple[int, int, int]] | None = None,
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
    mpl, plt = _get_mpl()

    resolved = _resolve_indices(image, indices, coordinates)

    # Read spatial metadata from headers (no data load)
    spatial_shape = image.spatial_shape
    spacing = image.spacing
    orientation = image.orientation

    # Find tensor axis for each anatomical pair
    axis_for: dict[str, int] = {}
    for pair in ("LR", "AP", "SI"):
        axis_for[pair] = _find_axis(orientation, pair)

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
        plt.close(fig)
        plot_axes: Sequence[Axes] = [fig.add_subplot(gs[0, i]) for i in range(3)]
    else:
        if len(axes) < 3:
            msg = f"Expected 3 axes, got {len(axes)}"
            raise ValueError(msg)
        plot_axes = axes
        fig = cast("Figure", plot_axes[0].get_figure())

    _plot_image_on_axes(
        image=image,
        plot_axes=plot_axes,
        channel=channel,
        resolved=resolved,
        cmap=cmap,
        percentiles=percentiles,
        voxels=voxels,
        intersections=intersections,
        **imshow_kwargs,
    )

    if title is not None:
        fig.suptitle(title)
    fig.tight_layout()

    if output_path is not None:
        fig.savefig(output_path, **(savefig_kwargs or {}))
    if show:
        _display_figure(fig)
        return None

    return fig


def _coordinates_to_indices(
    image: Image,
    coordinates: tuple[float | None, float | None, float | None],
) -> tuple[int | None, int | None, int | None]:
    """Convert world coordinates (mm) to voxel indices."""
    import torch

    inv_affine = image.affine.inverse()
    voxel_coords = inv_affine.apply(
        torch.tensor(
            [[c if c is not None else float("nan") for c in coordinates]],
            dtype=torch.float64,
        ),
    )[0]
    c0, c1, c2 = coordinates
    return (
        None if c0 is None else round(float(voxel_coords[0])),
        None if c1 is None else round(float(voxel_coords[1])),
        None if c2 is None else round(float(voxel_coords[2])),
    )


def _resolve_indices(
    image: Image,
    indices: tuple[int | None, int | None, int | None] | None,
    coordinates: tuple[float | None, float | None, float | None] | None,
) -> tuple[int, ...]:
    """Resolve indices/coordinates to concrete voxel indices."""
    if indices is not None and coordinates is not None:
        msg = "indices and coordinates are mutually exclusive"
        raise ValueError(msg)

    if coordinates is not None:
        indices = _coordinates_to_indices(image, coordinates)

    if indices is None:
        indices = (None, None, None)
    return tuple(
        s // 2 if idx is None else idx
        for idx, s in zip(indices, image.spatial_shape, strict=True)
    )


@overload
def plot_subject(
    subject: Subject,
    *,
    show: Literal[False],
    channel: int = ...,
    indices: tuple[int | None, int | None, int | None] | None = ...,
    coordinates: tuple[float | None, float | None, float | None] | None = ...,
    cmap_dict: dict[str, Any] | None = ...,
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
def plot_subject(
    subject: Subject,
    *,
    show: Literal[True] = ...,
    channel: int = ...,
    indices: tuple[int | None, int | None, int | None] | None = ...,
    coordinates: tuple[float | None, float | None, float | None] | None = ...,
    cmap_dict: dict[str, Any] | None = ...,
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


def plot_subject(
    subject: Subject,
    *,
    channel: int = 0,
    indices: tuple[int | None, int | None, int | None] | None = None,
    coordinates: tuple[float | None, float | None, float | None] | None = None,
    cmap_dict: dict[str, Any] | None = None,
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
    """Plot all images in a subject as a grid.

    Each image gets a row (or column if >3 images) of Sagittal,
    Coronal, Axial views. LabelMaps are automatically detected and
    use categorical colormaps.

    Args:
        subject: The subject to plot.
        channel: Which channel to display.
        indices: Voxel indices for each slice. Mutually exclusive
            with ``coordinates``.
        coordinates: World coordinates in mm. Mutually exclusive
            with ``indices``.
        cmap_dict: Per-image colormap overrides, keyed by image name.
        percentiles: Intensity percentile range for windowing.
        figsize: Figure size in inches.
        title: Figure super-title.
        output_path: Save figure to this path.
        show: Call ``plt.show()`` after plotting.
        savefig_kwargs: Extra keyword arguments for ``fig.savefig()``.
        voxels: Show voxel ticks instead of world coordinates.
        figsize_multiplier: Scale factor for default figure size.
        intersections: Draw slice intersection cross-hairs.
        **imshow_kwargs: Forwarded to ``ax.imshow()``.

    Returns:
        The ``Figure``, or ``None`` when ``show=True``.
    """
    mpl, plt = _get_mpl()

    images = subject.images
    num_images = len(images)
    if num_images == 0:
        msg = "Subject has no images to plot"
        raise ValueError(msg)

    first_image = next(iter(images.values()))
    _resolve_indices(first_image, indices, coordinates)

    many = num_images > 3
    fig, all_axes = _create_subject_grid(
        first_image,
        num_images,
        many,
        figsize,
        figsize_multiplier,
        mpl,
        plt,
    )

    _populate_subject_grid(
        images,
        all_axes,
        many,
        indices,
        coordinates,
        channel=channel,
        cmap_dict=cmap_dict,
        percentiles=percentiles,
        voxels=voxels,
        intersections=intersections,
        **imshow_kwargs,
    )

    if title is not None:
        fig.suptitle(title)
    fig.tight_layout()

    if output_path is not None:
        fig.savefig(output_path, **(savefig_kwargs or {}))
    if show:
        _display_figure(fig)
        return None

    return fig


def _create_subject_grid(
    first_image: Image,
    num_images: int,
    many: bool,
    figsize: tuple[float, float] | None,
    figsize_multiplier: float,
    mpl: Any,
    plt: Any,
) -> tuple[Any, list[list[Any]]]:
    """Create the figure and axes grid for ``plot_subject``."""
    orientation = first_image.orientation
    spacing = first_image.spacing
    spatial_shape = first_image.spatial_shape
    axis_for: dict[str, int] = {}
    for pair in ("LR", "AP", "SI"):
        axis_for[pair] = _find_axis(orientation, pair)
    lr_mm = spatial_shape[axis_for["LR"]] * spacing[axis_for["LR"]]
    ap_mm = spatial_shape[axis_for["AP"]] * spacing[axis_for["AP"]]
    width_ratios = [ap_mm, lr_mm, lr_mm]

    if figsize is None:
        default_w, default_h = plt.rcParams["figure.figsize"]
        figsize = (default_w * figsize_multiplier, default_h * figsize_multiplier)

    n_views = 3
    if many:
        nrows, ncols = n_views, num_images
        gs = mpl.gridspec.GridSpec(nrows, ncols)
    else:
        nrows, ncols = num_images, n_views
        gs = mpl.gridspec.GridSpec(nrows, ncols, width_ratios=width_ratios)

    fig = plt.figure(figsize=figsize)
    plt.close(fig)
    all_axes = [[fig.add_subplot(gs[r, c]) for c in range(ncols)] for r in range(nrows)]
    return fig, all_axes


def _populate_subject_grid(
    images: dict[str, Image],
    all_axes: list[list[Any]],
    many: bool,
    indices: tuple[int | None, int | None, int | None] | None,
    coordinates: tuple[float | None, float | None, float | None] | None,
    *,
    channel: int,
    cmap_dict: dict[str, Any] | None,
    percentiles: tuple[float, float],
    voxels: bool,
    intersections: bool,
    **imshow_kwargs: Any,
) -> None:
    """Plot each image into its row/column of the subject grid."""
    n_views = 3

    for img_idx, (name, image) in enumerate(images.items()):
        cmap = cmap_dict.get(name) if cmap_dict else None
        img_resolved = _resolve_indices(image, indices, coordinates)
        img_axes = _get_image_axes(all_axes, img_idx, n_views, many)

        _plot_image_on_axes(
            image=image,
            plot_axes=img_axes,
            channel=channel,
            resolved=img_resolved,
            cmap=cmap,
            percentiles=percentiles,
            voxels=voxels,
            intersections=intersections,
            show_titles=False,
            **imshow_kwargs,
        )

        _label_image_header(img_axes, name, many)


def _get_image_axes(
    all_axes: list[list[Any]],
    img_idx: int,
    n_views: int,
    many: bool,
) -> list[Any]:
    """Get the 3 axes for a given image in the grid."""
    if many:
        return [all_axes[v][img_idx] for v in range(n_views)]
    return all_axes[img_idx]


def _label_image_header(
    img_axes: list[Any],
    name: str,
    many: bool,
) -> None:
    """Add image name as a row/column header."""
    if many:
        img_axes[0].set_title(name)
    else:
        # Prepend image name to the existing ylabel (orientation label)
        existing = img_axes[0].get_ylabel()
        img_axes[0].set_ylabel(f"{name}\n{existing}", fontsize=10)


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
    x_size = spatial_shape[x_axis]
    y_size = spatial_shape[y_axis]
    x_flipped = x_code == x_left
    y_flipped = y_code != y_top

    x_ticks = np.linspace(0, x_size - 1, min(5, x_size))
    y_ticks = np.linspace(0, y_size - 1, min(5, y_size))
    ax.set_xticks(x_ticks)
    ax.set_yticks(y_ticks)

    if voxels:
        ax.set_xticklabels(_voxel_tick_labels(x_ticks, x_size, x_flipped))
        ax.set_yticklabels(_voxel_tick_labels(y_ticks, y_size, y_flipped))
    else:
        x_sp = spacing[x_axis]
        y_sp = spacing[y_axis]
        x_sign = -1.0 if x_code in ("L", "P", "I") else 1.0
        y_sign = -1.0 if y_code in ("L", "P", "I") else 1.0
        x_origin = origin_mm[_world_dim(x_code)]
        y_origin = origin_mm[_world_dim(y_code)]
        ax.set_xticklabels(
            _mm_tick_labels(x_ticks, x_size, x_flipped, x_origin, x_sp, x_sign),
        )
        ax.set_yticklabels(
            _mm_tick_labels(y_ticks, y_size, y_flipped, y_origin, y_sp, y_sign),
        )


def _voxel_tick_labels(
    ticks: np.ndarray,
    size: int,
    flipped: bool,
) -> list[str]:
    """Generate voxel-index tick labels."""
    if flipped:
        return [str(int(size - 1 - v)) for v in ticks]
    return [str(int(v)) for v in ticks]


def _mm_tick_labels(
    ticks: np.ndarray,
    size: int,
    flipped: bool,
    origin: float,
    sp: float,
    sign: float,
) -> list[str]:
    """Generate world-coordinate (mm) tick labels."""
    labels: list[str] = []
    for v in ticks:
        voxel = (size - 1 - v) if flipped else v
        mm = origin + voxel * sp * sign
        labels.append(f"{mm:.0f}")
    return labels


def _world_dim(code: str) -> int:
    """Map an orientation code to the world coordinate dimension (0=x, 1=y, 2=z)."""
    match code:
        case "R" | "L":
            return 0
        case "A" | "P":
            return 1
        case _:
            return 2


# ── GIF and video export ─────────────────────────────────────────────


def make_gif(
    image: Image,
    output_path: TypePath,
    *,
    seconds: float = 5.0,
    direction: str = "I",
    loop: int = 0,
    optimize: bool = True,
    rescale: bool = True,
    reverse: bool = False,
) -> None:
    """Save an animated GIF sweeping through slices of a 3D image.

    The image is reoriented so slices appear in the expected anatomical
    view for the given direction, matching ``make_video`` behavior.

    Args:
        image: A :class:`~torchio.Image` instance.
        output_path: Path to the output ``.gif`` file.
        seconds: Duration of the full animation in seconds.
        direction: Anatomical sweep direction: one of
            ``"I"``, ``"S"``, ``"A"``, ``"P"``, ``"R"``, ``"L"``.
        loop: Number of loops (0 = infinite).
        optimize: Attempt to compress the GIF palette.
        rescale: Rescale intensities to ``[0, 255]`` before encoding.
        reverse: Reverse the temporal order of frames.
    """
    import warnings
    from importlib import import_module
    from pathlib import Path

    from .external.imports import get_pillow
    from .transforms import Reorient

    get_pillow()  # raises ImportError with install hint if missing
    pil_image = import_module("PIL.Image")

    # Reorient so the sweep direction is the first spatial axis and the
    # remaining two axes produce an anatomically correct 2D view.
    target = _video_orientation(direction)
    reoriented = Reorient(orientation=target)(image)
    tensor = reoriented.data

    tensor = _rescale_to_uint8(tensor) if rescale else tensor.byte()

    single_channel = tensor.shape[0] == 1

    # Tensor is (C, sweep, H, W). Iterate over the sweep axis.
    frames = tensor.cpu().byte().numpy()
    mode = "P" if single_channel else "RGB"

    images = []
    for i in range(frames.shape[1]):
        # Single channel: (H, W); multi-channel: (C, H, W) -> (H, W, C)
        frame_2d = frames[0, i] if single_channel else np.moveaxis(frames[:, i], 0, -1)
        images.append(pil_image.fromarray(frame_2d).convert(mode))

    if reverse:
        images = list(reversed(images))

    num_images = len(images)
    # GIF frame delay is stored in centiseconds (10ms units).
    # Most browsers/viewers silently clamp delays ≤ 20ms to ~100ms,
    # so we enforce a 20ms floor for reliable playback timing.
    min_frame_ms = 20
    frame_duration_ms = round(seconds / num_images * 1000 / 10) * 10
    frame_duration_ms = max(frame_duration_ms, min_frame_ms)
    actual_seconds = frame_duration_ms * num_images / 1000
    if abs(actual_seconds - seconds) > 0.5 * seconds / num_images:
        warnings.warn(
            f"GIF frame delay is quantized to 10ms steps (minimum"
            f" {min_frame_ms}ms for browser compatibility). Actual"
            f" duration will be {actual_seconds:.2f}s instead of"
            f" {seconds:.2f}s. Consider reducing the number of slices"
            f" or increasing the requested duration.",
            RuntimeWarning,
            stacklevel=2,
        )

    images[0].save(
        Path(output_path),
        save_all=True,
        append_images=images[1:],
        optimize=optimize,
        duration=frame_duration_ms,
        loop=loop,
    )


def make_video(
    image: Image,
    output_path: TypePath,
    *,
    seconds: float = 5.0,
    direction: str = "I",
    verbosity: str = "error",
) -> None:
    """Create an MP4 video sweeping through slices of a 3D image.

    The image is reoriented so slices are shown in the expected
    anatomical view for the given direction. Requires ``ffmpeg-python``.

    Args:
        image: A single-channel :class:`~torchio.ScalarImage`.
        output_path: Path to the output ``.mp4`` file.
        seconds: Duration of the full video in seconds.
        direction: Anatomical sweep direction: one of
            ``"I"``, ``"S"``, ``"A"``, ``"P"``, ``"R"``, ``"L"``.
        verbosity: ffmpeg log level.
    """
    import warnings
    from pathlib import Path

    import torch

    from .external.imports import get_ffmpeg

    ffmpeg = get_ffmpeg()

    if image.num_channels > 1:
        msg = "Only single-channel images are supported for video export."
        raise ValueError(msg)

    # Reorient to the target sweep direction.
    target_orientation = _video_orientation(direction)
    from .transforms import Reorient

    reoriented = Reorient(orientation=target_orientation)(image)
    tensor = reoriented.data

    # Rescale to [0, 255] uint8 if needed.
    if tensor.min() < 0 or tensor.max() > 255:
        warnings.warn(
            "Tensor values outside [0, 256). Rescaling to [0, 255].",
            RuntimeWarning,
            stacklevel=2,
        )
        tensor = _rescale_to_uint8(tensor)
    if tensor.dtype != torch.uint8:
        tensor = tensor.byte()

    # Crop to even dimensions (required by H.265).
    num_frames, height, width = (
        tensor.shape[-3],
        tensor.shape[-2],
        tensor.shape[-1],
    )
    if height % 2 != 0:
        tensor = tensor[:, :, : height - 1, :]
        height -= 1
    if width % 2 != 0:
        tensor = tensor[:, :, :, : width - 1]
        width -= 1

    frame_rate = num_frames / seconds

    out = Path(output_path)
    if out.suffix.lower() != ".mp4":
        msg = "Only .mp4 output is supported."
        raise NotImplementedError(msg)

    frames = tensor[0].cpu().numpy()

    process = (
        ffmpeg.input(
            "pipe:",
            format="rawvideo",
            pix_fmt="gray",
            s=f"{width}x{height}",
            framerate=frame_rate,
        )
        .output(
            str(out),
            vcodec="libx264",
            pix_fmt="yuv420p",
            movflags="+faststart",
            # Baseline profile for maximum browser/Jupyter compatibility.
            profile="baseline",
            level="3.0",
            loglevel=verbosity,
        )
        .overwrite_output()
        .run_async(pipe_stdin=True)
    )

    for frame in frames:
        process.stdin.write(frame.tobytes())

    process.stdin.close()
    process.wait()


def _rescale_to_uint8(tensor: Any) -> Any:
    """Rescale a tensor to ``[0, 255]`` uint8."""

    t = tensor.float()
    tmin = t.min()
    tmax = t.max()
    if tmax - tmin > 0:
        t = (t - tmin) / (tmax - tmin) * 255
    return t.byte()


_VIDEO_ORIENTATIONS: dict[str, str] = {
    "I": "IPL",
    "S": "SPL",
    "A": "AIL",
    "P": "PIL",
    "R": "RIP",
    "L": "LIP",
}


def _video_orientation(direction: str) -> str:
    """Map a sweep direction letter to a 3-character orientation string."""
    direction = direction.upper()
    if direction not in _VIDEO_ORIENTATIONS:
        msg = (
            f"Direction must be one of {list(_VIDEO_ORIENTATIONS)}, got {direction!r}."
        )
        raise ValueError(msg)
    return _VIDEO_ORIENTATIONS[direction]
