# Visualizing images

TorchIO provides a built-in plotting function that displays three
orthogonal slices (Sagittal, Coronal, Axial) with correct anatomical
orientation and proportions. It works with any image orientation —
RAS, LPS, or anything else.

## Basic usage

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

```python
image.plot(voxels=True)
```

## Intensity windowing

For scalar images, the display range is set from the 0.5th to 99.5th
percentile by default. Adjust with:

```python
image.plot(percentiles=(1, 99))
```

## Label maps

Label maps automatically use nearest-neighbour interpolation and a
categorical colormap (from [colorcet](https://colorcet.holoviz.org/)
if installed, otherwise matplotlib's `tab10`):

```python
segmentation = tio.LabelMap("seg.nii.gz")
segmentation.plot()
```

## Customization

### Figure size

```python
# Scale the default figure size
image.plot(figsize_multiplier=3.0)

# Or set an exact size
image.plot(figsize=(12, 4))
```

### Colormap and imshow options

```python
image.plot(cmap="hot")
image.plot(vmin=0, vmax=1000)  # extra kwargs go to ax.imshow()
```

### Intersection lines

The coloured cross-hair lines can be turned off:

```python
image.plot(intersections=False)
```

### Saving to file

```python
image.plot(output_path="slices.png", show=False)
image.plot(
    output_path="slices.pdf",
    show=False,
    savefig_kwargs={"dpi": 300, "bbox_inches": "tight"},
)
```

### Plotting into existing axes

```python
import matplotlib.pyplot as plt

fig, axes = plt.subplots(1, 3, figsize=(15, 5))
image.plot(axes=axes, show=False)
```

## Jupyter notebooks

In Jupyter, `Image` objects display an inline plot automatically via
`_repr_html_`:

```python
image  # shows 3 slices + metadata table
```

Call `image.plot()` explicitly to get a larger, interactive figure.

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
