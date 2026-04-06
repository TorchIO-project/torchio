# Use a custom reader

TorchIO reads NIfTI files with NiBabel and everything else with
SimpleITK. If your data is in a format neither supports (e.g., a
custom binary format or a NumPy `.npy` file), you can pass a custom
reader.

## Write a reader function

A reader is a callable that takes a `Path` and returns a tuple of
`(tensor, affine_array)`:

```python
from pathlib import Path
import numpy as np
import torch

def npy_reader(path: Path) -> tuple[torch.Tensor, np.ndarray]:
    data = np.load(path)
    tensor = torch.from_numpy(data).unsqueeze(0)  # add channel dim
    affine = np.eye(4)  # identity affine (1mm isotropic)
    return tensor, affine
```

The tensor must be 4D with shape `(C, I, J, K)`.

## Use it

<!-- pytest-codeblocks:skip -->
```python
import torchio as tio

image = tio.ScalarImage("brain.npy", reader=npy_reader)
print(image.shape)   # triggers the reader
print(image.spacing)  # (1.0, 1.0, 1.0) from the identity affine
```

!!! warning

    When a custom reader is used, lazy backends (NibabelBackend,
    ZarrBackend) are not available. Operations like `.shape` will
    trigger a full load through your reader.
