# Save as NIfTI-Zarr

NIfTI-Zarr (`.nii.zarr`) stores image data in independently compressed
chunks, enabling lazy partial reads. This guide shows how to convert
existing volumes.

## Prerequisites

Install the `zarr` extra:

```
uv add torchio --extra zarr
```

## Convert a NIfTI file

<!-- pytest-codeblocks:skip -->
```python
import torchio as tio

image = tio.ScalarImage("input.nii.gz")
image.save("output.nii.zarr")
```

That is it. The output is a directory (`output.nii.zarr/`) containing
chunked Zarr arrays and the NIfTI header.

## Convert from a tensor

```python
import torch
import torchio as tio

tensor = torch.randn(1, 256, 256, 256)
image = tio.ScalarImage.from_tensor(tensor)
image.save("synthetic.nii.zarr")
```

## Verify the result

<!-- pytest-codeblocks:skip -->
```python
loaded = tio.ScalarImage("output.nii.zarr")
print(loaded.shape)    # reads only metadata
print(loaded.spacing)  # from the stored affine

# Lazy slice: reads only the needed chunks
patch = loaded[:, 50:100, 50:100, 50:100]
print(patch.data.mean())
```

## Chunk size

The default chunk size is 64 voxels per dimension. To customize it,
use `niizarr` directly:

<!-- pytest-codeblocks:skip -->
```python
import nibabel as nib
from niizarr import nii2zarr

nii = nib.load("input.nii.gz")
nii2zarr(nii, "output.nii.zarr", chunk=128)
```
