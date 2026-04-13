# Visualizing images

TorchIO provides a built-in plotting function that displays three
orthogonal slices (Sagittal, Coronal, Axial) with correct anatomical
orientation and proportions. It works with any image orientation —
RAS, LPS, or anything else.

!!! note "Optional dependency"
    Plotting requires the `plot` extra:

    === "uv"

        ```
        uv add torchio[plot]
        ```

    === "pip"

        ```
        pip install torchio[plot]
        ```

    This installs [matplotlib](https://matplotlib.org/) and
    [colorcet](https://colorcet.holoviz.org/) (for categorical
    colormaps).

## Basic usage

<!-- pytest-codeblocks:skip -->
```python
import torchio as tio

image = tio.ScalarImage("brain.nii.gz")
image.plot()
```

This shows mid-slices through each anatomical plane with:

- **Correct proportions** between views (from the voxel spacing)
- **Orientation labels** showing both the tensor axis and anatomical
  direction, e.g., `J (A ↔ P)`
- **World-coordinate ticks** in mm (derived from the affine matrix)
- **Coloured intersection lines** showing where the other slices are

## Choosing slices

By default, the mid-slice along each axis is shown. You can specify
slices by voxel index or by world coordinates in mm:

<!-- pytest-codeblocks:skip -->
```python
# By voxel index (None = mid-slice)
image.plot(indices=(80, None, 60))

# By world coordinates in mm
image.plot(coordinates=(-10.0, 25.5, 30.0))

# Mix None and values
image.plot(coordinates=(None, 0.0, None))
```

!!! tip
    `indices` and `coordinates` are mutually exclusive.

## Tick labels

By default, tick labels show world coordinates in mm. Pass
`voxels=True` to show voxel indices instead:

<!-- pytest-codeblocks:skip -->
```python
image.plot(voxels=True)
```

## Intensity windowing

For scalar images, the display range is set from the 0.5th to 99.5th
percentile by default. Adjust with:

<!-- pytest-codeblocks:skip -->
```python
image.plot(percentiles=(1, 99))
```

## Label maps

Label maps automatically use nearest-neighbour interpolation and a
categorical colormap (from [colorcet](https://colorcet.holoviz.org/)
if installed, otherwise matplotlib's `tab10`):

<!-- pytest-codeblocks:skip -->
```python
segmentation = tio.LabelMap("seg.nii.gz")
segmentation.plot()
```

## Customization

### Figure size

<!-- pytest-codeblocks:skip -->
```python
# Scale the default figure size
image.plot(figsize_multiplier=3.0)

# Or set an exact size
image.plot(figsize=(12, 4))
```

### Colormap and imshow options

<!-- pytest-codeblocks:skip -->
```python
image.plot(cmap="hot")
image.plot(vmin=0, vmax=1000)  # extra kwargs go to ax.imshow()
```

### Intersection lines

The coloured cross-hair lines can be turned off:

<!-- pytest-codeblocks:skip -->
```python
image.plot(intersections=False)
```

### Saving to file

<!-- pytest-codeblocks:skip -->
```python
image.plot(output_path="slices.png", show=False)
image.plot(
    output_path="slices.pdf",
    show=False,
    savefig_kwargs={"dpi": 300, "bbox_inches": "tight"},
)
```

### Plotting into existing axes

<!-- pytest-codeblocks:skip -->
```python
import matplotlib.pyplot as plt

fig, axes = plt.subplots(1, 3, figsize=(15, 5))
image.plot(axes=axes, show=False)
```

## Jupyter notebooks

In Jupyter, `Image` objects display an inline plot automatically via
`_repr_html_`:

<!-- pytest-codeblocks:skip -->
```python
image  # shows 3 slices + metadata table
```

Call `image.plot()` explicitly to get a larger, interactive figure.

## Subjects

Plot all images in a subject as a grid:

<!-- pytest-codeblocks:skip -->
```python
subject = tio.Subject(
    t1=tio.ScalarImage("t1.nii.gz"),
    seg=tio.LabelMap("seg.nii.gz"),
)
subject.plot()
```

Each image gets a row of Sagittal/Coronal/Axial views (or columns
if there are more than 3 images). LabelMaps are automatically
detected and use categorical colormaps.

### Per-image colormaps

<!-- pytest-codeblocks:skip -->
```python
subject.plot(cmap_dict={"t1": "hot", "seg": "viridis"})
```

### In Jupyter

`Subject` objects also display an inline plot via `_repr_html_`:

<!-- pytest-codeblocks:skip -->
```python
subject  # shows grid + metadata tables
```

## How it works

The display always follows the same anatomical convention regardless
of the image orientation:

| View      | Horizontal       | Vertical        |
|-----------|------------------|-----------------|
| Sagittal  | Anterior ↔ Posterior | Inferior ↔ Superior |
| Coronal   | Right ↔ Left        | Inferior ↔ Superior |
| Axial     | Right ↔ Left        | Posterior ↔ Anterior |

The data is flipped and transposed as needed so these directions are
always in the same screen positions. The axis labels show which tensor
axis (I, J, K) maps to each direction, making it easy to relate what
you see to the underlying data layout.

## Animated GIFs and videos

You can export an animation sweeping through slices along any
anatomical direction:

<!-- pytest-codeblocks:skip -->
```python
image.to_gif("brain.gif", seconds=5, direction="I")
image.to_video("brain.mp4", seconds=5, direction="S")
```

The ``direction`` parameter accepts ``"I"`` (inferior), ``"S"``
(superior), ``"A"`` (anterior), ``"P"`` (posterior), ``"R"`` (right),
or ``"L"`` (left). The image is automatically reoriented so slices
appear in the correct anatomical view.

From the command line:

<!-- pytest-codeblocks:skip -->
```bash
torchio animate brain.nii.gz brain.gif
torchio animate brain.nii.gz brain.mp4 --seconds 10 --direction S
```

!!! note "Optional dependencies"
    GIFs require ``Pillow`` (included in the ``[plot]`` extra).
    Videos require ``ffmpeg-python`` (``pip install torchio[video]``)
    and a working ``ffmpeg`` installation.
