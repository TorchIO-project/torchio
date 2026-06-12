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

!!! note "Simple readers load eagerly"

    A reader that only returns `(tensor, affine)` cannot read metadata or
    regions lazily, so operations like `.shape`, `.dtype`, and slicing trigger
    a full load through your reader. This is unchanged behavior. To opt into
    lazy access, make your reader a *lazy reader* (below).

## Lazy custom readers

!!! note "Advanced, rarely needed"

    Most custom readers can stay simple and eager. Only reach for a lazy reader
    when your format can cheaply read metadata or sub-regions and you actually
    care about avoiding full loads (for example, very large volumes).

If your format supports reading the shape, affine, dtype, or sub-regions
without loading everything, implement `create_backend` so TorchIO can access
it lazily. A lazy reader is any object that has a `create_backend` method
returning an object implementing the
[`ImageDataBackend`][torchio.data.backends.ImageDataBackend] protocol.

Building on the `.npy` reader above, this version is lazy: it returns a
memory-mapped backend, so `.shape` reads only the header and slicing reads only
the requested block.

<!-- pytest-codeblocks:skip -->
```python
from pathlib import Path

import numpy as np
import torch
from torchio.data.backends import BackendRequest, ImageDataBackend, normalize_index


class NpyBackend:
    """Lazy, memory-mapped backend for single-channel ``.npy`` volumes."""

    def __init__(self, path: Path):
        self._memmap = np.load(path, mmap_mode="r")  # shape (I, J, K), unread

    @property
    def shape(self):
        i, j, k = self._memmap.shape
        return (1, i, j, k)

    @property
    def affine(self):
        return torch.eye(4, dtype=torch.float64)

    @property
    def dtype(self):
        return self._memmap.dtype

    def __getitem__(self, index):
        sc, si, sj, sk = normalize_index(index)
        return torch.from_numpy(np.array(self._memmap[si, sj, sk]))[None][sc]

    def to_tensor(self):
        return torch.from_numpy(np.array(self._memmap))[None]


class LazyNpyReader:
    """A custom reader for ``.npy`` that supports lazy access."""

    def __call__(self, path: Path, **kwargs) -> tuple:
        # Eager fallback, used only if create_backend is unavailable.
        backend = self.create_backend(BackendRequest(path=path))
        return backend.to_tensor(), backend.affine.numpy()

    def create_backend(self, request: BackendRequest) -> ImageDataBackend:
        return NpyBackend(request.path)


image = tio.ScalarImage("volume.npy", reader=LazyNpyReader())
print(image.shape)  # read from the header, no full load
```

With a lazy reader, `.shape`, `.affine`, `.dtype`, and `image.dataobj[...]`
slicing all go through your backend without materializing the full tensor.

Passing `reader=...` is per image. If instead you want *every* `.npy` file to
use this backend, register it once globally with
[`register_backend`][torchio.data.backends.register_backend]; see
[Lazy loading and backends](../concepts/lazy-loading.md) for the backend
contract and that registry-based alternative.
