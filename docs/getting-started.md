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

    For cloud storage (HTTP/HTTPS URLs work out of the box):

    === "S3"

        ```
        pip install "torchio[s3]"
        ```

    === "Azure Blob"

        ```
        pip install "torchio[azure]"
        ```

    === "Google Cloud"

        ```
        pip install "torchio[gcs]"
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

# From a URL or cloud path
image = tio.ScalarImage("https://example.com/t1.nii.gz")
image = tio.ScalarImage("s3://bucket/t1.nii.gz")

# From a file-like object
import io
buf = io.BytesIO(open("t1.nii.gz", "rb").read())
image = tio.ScalarImage(buf, suffix=".nii.gz")
```

### Creating from a tensor

```python
import torch

tensor = torch.randn(1, 128, 128, 128)
image = tio.ScalarImage.from_tensor(tensor)
```

### Creating from SimpleITK or NiBabel

```python
import SimpleITK as sitk
import nibabel as nib

# From a SimpleITK Image (preserves spacing, origin, direction)
sitk_image = sitk.ReadImage("t1.nii.gz")
image = tio.ScalarImage.from_sitk(sitk_image)

# From a NiBabel Nifti1Image (preserves affine)
nifti = nib.load("t1.nii.gz")
image = tio.ScalarImage.from_nifti(nifti)
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

### Applying transforms

Transforms accept Subjects, Images, Tensors, NumPy arrays,
SimpleITK Images, NiBabel images, or MONAI-style dicts — and return
the same type:

```python
# Single deterministic transform
flipped = tio.Flip(axes=(0,))(subject)

# Random augmentation pipeline
augment = tio.Compose([
    tio.Flip(axes=(0, 1, 2), p=0.5),
    tio.Noise(std=(0.01, 0.1)),   # random std each call
])
augmented = augment(subject)

# Works directly on tensors too
noisy_tensor = tio.Noise(std=0.05)(tensor)

# Works with MONAI-style dicts
data = {"image": tensor, "label": label_tensor}
augmented = tio.Noise(std=0.1)(data)  # returns dict
```

See the [transform design explanation](explanation/transforms.md) for
the full picture.
