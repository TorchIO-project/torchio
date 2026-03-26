# Getting started

## Installation

=== "uv"

    ```
    uv add torchio
    ```

=== "pip"

    ```
    pip install torchio
    ```

!!! note "Optional extras"

    For NIfTI-Zarr support (chunked, lazy-loadable volumes):

    === "uv"

        ```
        uv add torchio --extra zarr
        ```

    === "pip"

        ```
        pip install "torchio[zarr]"
        ```

## Quick tour

### Loading an image

```python
import torchio as tio

# From a file (lazy -- no data read yet)
image = tio.ScalarImage("t1.nii.gz")
print(image.shape)    # reads header only: (1, 256, 256, 176)
print(image.spacing)  # (1.0, 1.0, 1.0)

# Data is loaded on first access
tensor = image.data   # shape: (1, 256, 256, 176), dtype: float32
```

### Creating from a tensor

```python
import torch

tensor = torch.randn(1, 128, 128, 128)
image = tio.ScalarImage.from_tensor(tensor)
```

### Slicing

Slicing follows the `(C, I, J, K)` layout and keeps things lazy -- only
the requested region is read from disk:

```python
image = tio.ScalarImage("big_volume.nii.gz")
patch = image[:, 100:200, 100:200, 50:100]  # no full load
patch.data.mean()  # reads only this region
```

The affine origin is updated automatically so the patch stays in the
correct world coordinates.

### Grouping images into a Subject

```python
subject = tio.Subject(
    t1=tio.ScalarImage("t1.nii.gz"),
    seg=tio.LabelMap("seg.nii.gz"),
    age=45,
    diagnosis="healthy",
)

subject.t1           # Image access
subject["seg"]       # dict-style access
subject.age          # metadata access (returns 45)
```

### Saving

```python
# Any format SimpleITK supports
image.save("output.nii.gz")
image.save("output.nrrd")

# NIfTI-Zarr (chunked, lazy-loadable)
image.save("output.nii.zarr")
```
