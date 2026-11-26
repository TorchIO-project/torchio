# Write a custom transform

TorchIO ships with many transforms, but your preprocessing or augmentation
idea may not be among them. A custom transform is a small class: you
declare the parameters and you say how they change the data.

This guide writes two transforms, one deterministic and one random, and
shows how to plug them into a pipeline.

## The two methods to implement

Every transform subclasses one of the abstract bases (`Transform`,
`IntensityTransform`, `SpatialTransform`, ...) and implements:

- `make_params(batch)`: returns a dictionary of parameters for one
  application. This is where randomness is sampled.
- `apply_transform(batch, params)`: returns the transformed batch. This is
  where the data is modified.

The parameters returned by `make_params` are stored in the subject history,
so they must be JSON-serializable: plain Python numbers, strings, lists and
booleans. Tensors and NumPy arrays do not belong there.

## A deterministic transform

`IntensityTransform` is the right base for anything that changes voxel
values: it selects `ScalarImage` instances and leaves label maps alone.

```python
import torch
import torchio as tio


class AddConstant(tio.IntensityTransform):
    """Add a constant value to every intensity image."""

    def __init__(self, value: float, **kwargs):
        super().__init__(**kwargs)
        self.value = value

    def make_params(self, batch):
        return {"value": self.value}

    def apply_transform(self, batch, params):
        for _, img_batch in self._get_images(batch).items():
            img_batch.data = img_batch.data + params["value"]
        return batch


subject = tio.Subject(image=tio.ScalarImage(torch.zeros(1, 8, 8, 8)))
out = AddConstant(2.0)(subject)
print(out.image.data.mean().item())  # 2.0
```

Two details matter here:

- `_get_images(batch)` returns a dictionary of image batches, already
  filtered to `ScalarImage` and to the transform's `include` / `exclude`
  scopes. Iterate over it instead of reaching into `batch.images`.
- `img_batch.data` is a 5D tensor with shape `(B, C, I, J, K)`, where `B`
  is the number of subjects in the batch. Assign a tensor of the same shape
  and dtype back to it.

## Sample one value per subject

Real augmentation usually draws a different value for each subject in the
batch. `make_params` receives the batch, so it can draw as many values as
there are subjects:

```python
import torch
import torchio as tio


class Offset(tio.IntensityTransform):
    """Add a random offset, drawn independently for each subject."""

    def __init__(self, max_offset: float = 0.1, **kwargs):
        super().__init__(**kwargs)
        self.max_offset = max_offset

    def make_params(self, batch):
        offsets = torch.empty(len(batch)).uniform_(-self.max_offset, self.max_offset)
        return {"offsets": offsets.tolist()}

    def apply_transform(self, batch, params):
        offsets = torch.tensor(params["offsets"]).view(-1, 1, 1, 1, 1)
        for _, img_batch in self._get_images(batch).items():
            img_batch.data = img_batch.data + offsets.to(img_batch.data)
        return batch


subjects = [
    tio.Subject(image=tio.ScalarImage(torch.zeros(1, 8, 8, 8))) for _ in range(2)
]
loader = tio.SubjectsLoader(subjects, batch_size=2)
for batch in loader:
    out = Offset(max_offset=0.5)(batch)
    print([round(float(subject.image.data.mean()), 2) for subject in out.unbatch()])
```

The `view(-1, 1, 1, 1, 1)` reshapes the per-subject vector so that it
broadcasts over channels and spatial dimensions. The `.to(img_batch.data)`
call keeps the offsets on the same device and dtype as the image, which
matters on GPU and with half precision.

## What the base class already handles

Do not reimplement these; `Transform` does them for you:

- `p`: the probability of applying the transform. When it is not applied,
  the data is returned unchanged.
- `include` and `exclude`: which images the transform touches.
- History: `subject.applied_transforms` records the class name and the
  parameters returned by `make_params`.
- Registration: subclasses are registered automatically, so the class can
  be found by name when replaying history.

## Making it invertible

If your transform has an exact inverse, override the `invertible` property
and return an `inverse` transform from `inverse(params)`. The inverse
receives the saved parameters and must undo the operation. If there is no
exact inverse, leave the default `invertible = False`; TorchIO will then
skip the transform when an inverse composition is requested.

## Test it

A transform is easiest to trust with a small deterministic test:

<!-- pytest-codeblocks:skip -->

```python
def test_add_constant():
    image = tio.ScalarImage(torch.zeros(1, 4, 4, 4))
    out = AddConstant(3.0)(tio.Subject(image=image))
    assert out.image.data.min().item() == 3.0
```

For random transforms, fix the seed and assert on the recorded parameters
rather than on the output values, which keeps the test stable across
PyTorch versions.

## Where to put the code

Keep custom transforms in your own package, not in TorchIO. Import the base
class you need:

```python
from torchio.transforms import IntensityTransform, SpatialTransform
```

If you think the transform belongs in TorchIO itself, open an issue first
and describe the use case; the maintainers will tell you whether it fits
the scope of the library.
