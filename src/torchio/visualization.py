from __future__ import annotations

import warnings
from itertools import cycle
from pathlib import Path
from typing import TYPE_CHECKING
from typing import Any

import numpy as np
import torch
from einops import rearrange

from .data.image import Image
from .data.image import LabelMap
from .data.image import ScalarImage
from .data.subject import Subject
from .external.imports import get_ffmpeg
from .transforms.preprocessing.intensity.rescale import RescaleIntensity
from .transforms.preprocessing.intensity.to import To
from .transforms.preprocessing.spatial.ensure_shape_multiple import EnsureShapeMultiple
from .transforms.preprocessing.spatial.resample import Resample
from .transforms.preprocessing.spatial.to_canonical import ToCanonical
from .transforms.preprocessing.spatial.to_orientation import ToOrientation
from .types import TypePath

if TYPE_CHECKING:
    from matplotlib.colors import BoundaryNorm
    from matplotlib.colors import ListedColormap
    from matplotlib.figure import Figure


def import_mpl_plt():
    try:
        import matplotlib as mpl
        import matplotlib.pyplot as plt
    except ImportError as e:
        raise ImportError('Install matplotlib for plotting support') from e
    return mpl, plt


def rotate(image: np.ndarray, *, radiological: bool = True, n: int = -1) -> np.ndarray:
    # Rotate for visualization purposes
    image = np.rot90(image, n, axes=(0, 1))
    if radiological:
        image = np.fliplr(image)
    return image


def _create_categorical_colormap(
    data: torch.Tensor,
    cmap_name: str = 'glasbey_category10',
) -> tuple[ListedColormap, BoundaryNorm]:
    num_classes = int(data.max())
    mpl, _ = import_mpl_plt()

    colors = [
        (0, 0, 0),  # black for background
        (1, 1, 1),  # white for class 1
    ]
    if num_classes > 1:
        from .external.imports import get_colorcet

        colorcet = get_colorcet()
        cmap = getattr(colorcet.cm, cmap_name)
        color_cycle = cycle(cmap.colors)
        distinct_colors = [next(color_cycle) for _ in range(num_classes - 1)]
        colors.extend(distinct_colors)
    boundaries = np.arange(-0.5, num_classes + 1.5, 1)
    colormap = mpl.colors.ListedColormap(colors)
    boundary_norm = mpl.colors.BoundaryNorm(boundaries, ncolors=colormap.N)
    return colormap, boundary_norm


def plot_volume(
    image: Image,
    radiological=True,
    channel=None,
    axes=None,
    cmap=None,
    output_path=None,
    show=True,
    xlabels=True,
    percentiles: tuple[float, float] = (0, 100),
    figsize=None,
    title=None,
    reorient=True,
    indices=None,
    rgb=True,
    savefig_kwargs: dict[str, Any] | None = None,
    **imshow_kwargs,
) -> Figure | None:
    _, plt = import_mpl_plt()
    fig: Figure | None = None
    if axes is None:
        fig, axes = plt.subplots(1, 3, figsize=figsize)

    if reorient:
        image = ToCanonical()(image)  # type: ignore[assignment]

    is_label = isinstance(image, LabelMap)
    if is_label:  # probabilistic label map
        data = image.data[np.newaxis, -1]
    elif rgb and image.num_channels == 3:
        data = image.data  # keep image as it is
    elif channel is None:
        data = image.data[0:1]  # just use the first channel
    else:
        data = image.data[np.newaxis, channel]
    data = rearrange(data, 'c x y z -> x y z c')
    data_numpy: np.ndarray = data.cpu().numpy()

    if indices is None:
        indices = np.array(data_numpy.shape[:3]) // 2
    i, j, k = indices
    slice_x = rotate(data_numpy[i, :, :], radiological=radiological)
    slice_y = rotate(data_numpy[:, j, :], radiological=radiological)
    slice_z = rotate(data_numpy[:, :, k], radiological=radiological)

    if isinstance(cmap, dict):
        slices = slice_x, slice_y, slice_z
        slice_x, slice_y, slice_z = color_labels(slices, cmap)
    else:
        boundary_norm = None
        if cmap is None:
            if is_label:
                cmap, boundary_norm = _create_categorical_colormap(data)
            else:
                cmap = 'gray'
        imshow_kwargs['cmap'] = cmap
        imshow_kwargs['norm'] = boundary_norm

    if is_label:
        imshow_kwargs['interpolation'] = 'none'
    else:
        if 'interpolation' not in imshow_kwargs:
            imshow_kwargs['interpolation'] = 'bicubic'

    imshow_kwargs['origin'] = 'lower'

    if not is_label:
        displayed_data = np.concatenate(
            [
                slice_x.flatten(),
                slice_y.flatten(),
                slice_z.flatten(),
            ]
        )
        p1, p2 = np.percentile(displayed_data, percentiles)
        if 'vmin' not in imshow_kwargs:
            imshow_kwargs['vmin'] = p1
        if 'vmax' not in imshow_kwargs:
            imshow_kwargs['vmax'] = p2

    spacing_r, spacing_a, spacing_s = image.spacing
    sag_axis, cor_axis, axi_axis = axes
    slices_dict = {
        'Sagittal': {
            'aspect': spacing_s / spacing_a,
            'slice': slice_x,
            'xlabel': 'A',
            'ylabel': 'S',
            'axis': sag_axis,
        },
        'Coronal': {
            'aspect': spacing_s / spacing_r,
            'slice': slice_y,
            'xlabel': 'R',
            'ylabel': 'S',
            'axis': cor_axis,
        },
        'Axial': {
            'aspect': spacing_a / spacing_r,
            'slice': slice_z,
            'xlabel': 'R',
            'ylabel': 'A',
            'axis': axi_axis,
        },
    }

    for axis_title, info in slices_dict.items():
        axis = info['axis']
        axis.imshow(info['slice'], aspect=info['aspect'], **imshow_kwargs)
        if xlabels:
            axis.set_xlabel(info['xlabel'])
        axis.set_ylabel(info['ylabel'])
        axis.invert_xaxis()
        axis.set_title(axis_title)

    plt.tight_layout()
    if title is not None:
        plt.suptitle(title)

    if output_path is not None and fig is not None:
        if savefig_kwargs is None:
            savefig_kwargs = {}
        fig.savefig(output_path, **savefig_kwargs)
    if show:
        plt.show()
    return fig


def plot_subject(
    subject: Subject,
    cmap_dict=None,
    show=True,
    output_path=None,
    figsize=None,
    clear_axes=True,
    **plot_volume_kwargs,
):
    _, plt = import_mpl_plt()
    num_images = len(subject)
    many_images = num_images > 2
    subplots_kwargs = {'figsize': figsize}
    try:
        if clear_axes:
            subject.check_consistent_spatial_shape()
            subplots_kwargs['sharex'] = 'row' if many_images else 'col'
            subplots_kwargs['sharey'] = 'row' if many_images else 'col'
    except RuntimeError:  # different shapes in subject
        pass
    args = (3, num_images) if many_images else (num_images, 3)
    fig, axes = plt.subplots(*args, **subplots_kwargs)
    # The array of axes must be 2D so that it can be indexed correctly within
    # the plot_volume() function
    axes = axes.T if many_images else axes.reshape(-1, 3)
    iterable = enumerate(subject.get_images_dict(intensity_only=False).items())
    axes_names = 'sagittal', 'coronal', 'axial'
    for image_index, (name, image) in iterable:
        image_axes = axes[image_index]
        cmap = None
        if cmap_dict is not None and name in cmap_dict:
            cmap = cmap_dict[name]
        last_row = image_index == len(axes) - 1
        plot_volume(
            image,
            axes=image_axes,
            show=False,
            cmap=cmap,
            xlabels=last_row,
            **plot_volume_kwargs,
        )
        for axis, axis_name in zip(image_axes, axes_names):
            axis.set_title(f'{name} ({axis_name})')
    plt.tight_layout()
    if output_path is not None:
        fig.savefig(output_path)
    if show:
        plt.show()


def get_num_bins(x: np.ndarray) -> int:
    """Get the optimal number of bins for a histogram.

    This method uses the Freedman–Diaconis rule to compute the histogram that
    minimizes "the integral of the squared difference between the histogram
    (i.e., relative frequency density) and the density of the theoretical
    probability distribution" (`Wikipedia <https://en.wikipedia.org/wiki/Freedman%E2%80%93Diaconis_rule>`_).

    Args:
        x: Input values.
    """
    # Freedman–Diaconis number of bins
    q25, q75 = np.percentile(x, [25, 75])
    bin_width = 2 * (q75 - q25) * len(x) ** (-1 / 3)
    bins = round((x.max() - x.min()) / bin_width)
    return bins


def plot_histogram(x: np.ndarray, show=True, **kwargs) -> None:
    _, plt = import_mpl_plt()
    plt.hist(x, bins=get_num_bins(x), **kwargs)
    plt.xlabel('Intensity')
    density = kwargs.pop('density', False)
    ylabel = 'Density' if density else 'Frequency'
    plt.ylabel(ylabel)
    if show:
        plt.show()


def color_labels(arrays, cmap_dict):
    results = []
    for slice_array in arrays:
        si, sj, _ = slice_array.shape
        rgb = np.zeros((si, sj, 3), dtype=np.uint8)
        for label, color in cmap_dict.items():
            if isinstance(color, str):
                mpl, _ = import_mpl_plt()
                color = mpl.colors.to_rgb(color)
                color = [255 * n for n in color]
            rgb[slice_array[..., 0] == label] = color
        results.append(rgb)
    return results


def make_gif(
    tensor: torch.Tensor,
    axis: int,
    duration: float,  # of full gif
    output_path: TypePath,
    loop: int = 0,
    optimize: bool = True,
    rescale: bool = True,
    reverse: bool = False,
) -> None:
    try:
        from PIL import Image as ImagePIL
    except ModuleNotFoundError as e:
        message = 'Please install Pillow to use Image.to_gif(): pip install Pillow'
        raise RuntimeError(message) from e
    transform = RescaleIntensity((0, 255))
    tensor = transform(tensor) if rescale else tensor  # type: ignore[assignment]
    single_channel = len(tensor) == 1

    # Move channels dimension to the end and bring selected axis to 0
    axes = np.roll(range(1, 4), -axis)
    tensor = tensor.permute(*axes, 0)

    if single_channel:
        mode = 'P'
        tensor = tensor[..., 0]
    else:
        mode = 'RGB'
    array = tensor.byte().numpy()
    n = 2 if axis == 1 else 1
    images = [ImagePIL.fromarray(rotate(i, n=n)).convert(mode) for i in array]
    num_images = len(images)
    images = list(reversed(images)) if reverse else images
    frame_duration_ms = duration / num_images * 1000
    if frame_duration_ms < 10:
        fps = round(1000 / frame_duration_ms)
        frame_duration_ms = 10
        new_duration = frame_duration_ms * num_images / 1000
        message = (
            'The computed frame rate from the given duration is too high'
            f' ({fps} fps). The highest possible frame rate in the GIF'
            ' file format specification is 100 fps. The duration has been set'
            f' to {new_duration:.1f} seconds, instead of {duration:.1f}'
        )
        warnings.warn(message, RuntimeWarning, stacklevel=2)
    images[0].save(
        output_path,
        save_all=True,
        append_images=images[1:],
        optimize=optimize,
        duration=frame_duration_ms,
        loop=loop,
    )


def make_video(
    image: ScalarImage,
    output_path: TypePath,
    seconds: float | None = None,
    frame_rate: float | None = None,
    direction: str = 'I',
    verbosity: str = 'error',
) -> None:
    ffmpeg = get_ffmpeg()

    if seconds is None and frame_rate is None:
        message = 'Either seconds or frame_rate must be provided.'
        raise ValueError(message)
    if seconds is not None and frame_rate is not None:
        message = 'Provide either seconds or frame_rate, not both.'
        raise ValueError(message)
    if image.num_channels > 1:
        message = 'Only single-channel tensors are supported for video output for now.'
        raise ValueError(message)
    tmin, tmax = image.data.min(), image.data.max()
    if tmin < 0 or tmax > 255:
        message = (
            'The tensor must be in the range [0, 256) for video output.'
            ' The image data will be rescaled to this range.'
        )
        warnings.warn(message, RuntimeWarning, stacklevel=2)
        image = RescaleIntensity((0, 255))(image)
    if image.data.dtype != torch.uint8:
        message = (
            'Only uint8 tensors are supported for video output. The image data'
            ' will be cast to uint8.'
        )
        warnings.warn(message, RuntimeWarning, stacklevel=2)
        image = To(torch.uint8)(image)

    # Reorient so the output looks like in typical visualization software
    direction = direction.upper()
    if direction == 'I':  # axial top to bottom
        target = 'IPL'
    elif direction == 'S':  # axial bottom to top
        target = 'SPL'
    elif direction == 'A':  # coronal back to front
        target = 'AIL'
    elif direction == 'P':  # coronal front to back
        target = 'PIL'
    elif direction == 'R':  # sagittal left to right
        target = 'RIP'
    elif direction == 'L':  # sagittal right to left
        target = 'LIP'
    else:
        message = (
            'Direction must be one of "I", "S", "P", "A", "R" or "L".'
            f' Got {direction!r}.'
        )
        raise ValueError(message)
    image = ToOrientation(target)(image)

    # Check isotropy
    spacing_f, spacing_h, spacing_w = image.spacing
    if spacing_h != spacing_w:
        message = (
            'The height and width spacings should be the same video output.'
            f' Got {spacing_h:.2f} and {spacing_w:.2f}.'
            f' Resampling both to {spacing_f:.2f}.'
        )
        warnings.warn(message, RuntimeWarning, stacklevel=2)
        spacing_iso = min(spacing_h, spacing_w)
        target_spacing = spacing_f, spacing_iso, spacing_iso
        image = Resample(target_spacing)(image)  # type: ignore[assignment]

    # Check that height and width are multiples of 2 for H.265 encoding
    num_frames, height, width = image.spatial_shape
    if height % 2 != 0 or width % 2 != 0:
        message = (
            f'The height ({height}) and width ({width}) must be even.'
            ' The image will be cropped to the nearest even number.'
        )
        warnings.warn(message, RuntimeWarning, stacklevel=2)
        image = EnsureShapeMultiple((1, 2, 2), method='crop')(image)

    if seconds is not None:
        frame_rate = num_frames / seconds

    output_path = Path(output_path)
    if output_path.suffix.lower() != '.mp4':
        message = 'Only .mp4 files are supported for video output.'
        raise NotImplementedError(message)

    frames = image.numpy()[0]
    first = frames[0]
    height, width = first.shape

    process = (
        ffmpeg.input(
            'pipe:',
            format='rawvideo',
            pix_fmt='gray',
            s=f'{width}x{height}',
            framerate=frame_rate,
        )
        .output(
            str(output_path),
            vcodec='libx265',
            pix_fmt='yuv420p',
            loglevel=verbosity,
            **{'x265-params': f'log-level={verbosity}'},
        )
        .overwrite_output()
        .run_async(pipe_stdin=True)
    )

    for array in frames:
        buffer = array.tobytes()
        process.stdin.write(buffer)

    process.stdin.close()
    process.wait()
