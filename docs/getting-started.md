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

### Grouping data into a Subject

A `Subject` holds images, spatial annotations, and metadata:

```python
import torch

subject = tio.Subject(
    t1=tio.ScalarImage("t1.nii.gz"),
    seg=tio.LabelMap("seg.nii.gz"),
    landmarks=tio.Points(torch.tensor([[64.0, 64.0, 32.0]])),
    tumors=tio.BoundingBoxes(
        torch.tensor([[10, 20, 30, 50, 60, 70]]),
        format=tio.BoundingBoxFormat.IJKIJK,
    ),
    age=45,
)

subject.t1           # Image access
subject.landmarks    # Points access
subject.tumors       # BoundingBoxes access
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

### Batching with a DataLoader

```python
from torch.utils.data import Dataset

class BrainDataset(Dataset):
    def __init__(self, paths):
        self.subjects = [
            tio.Subject(
                t1=tio.ScalarImage(p / "t1.nii.gz"),
                seg=tio.LabelMap(p / "seg.nii.gz"),
            )
            for p in paths
        ]

    def __len__(self):
        return len(self.subjects)

    def __getitem__(self, idx):
        return self.subjects[idx]

loader = tio.SubjectsLoader(BrainDataset(paths), batch_size=4)
batch = next(iter(loader))
batch["t1", "data"].shape  # (4, 1, 256, 256, 176)
```

See the [DataLoader how-to guide](how-to/dataloader.md) for more
details.
