# Transform design

TorchIO transforms are `torch.nn.Module` subclasses. They accept
Subjects, Images, Tensors, NumPy arrays, SimpleITK images, NiBabel
images, MONAI-style dicts, `ImagesBatch`, or `SubjectsBatch`,
and always return the same type.

## Unified batch architecture

Internally, **all inputs are converted to a `SubjectsBatch`**
before the transform runs. A single `Image` becomes a batch of
size 1; a `SubjectsBatch` from a `DataLoader` passes through
directly. This means transform authors write **one batch-oriented
application method** that works identically for single samples and batches
(`apply_transform`), plus `make_params` when parameter construction is
needed:

```python
from typing import Any

import torch
import torchio as tio


class AddValue(tio.Transform):
    """Add a fixed value to every image in a batch."""

    def __init__(self, value: float) -> None:
        super().__init__()
        self.value = value

    def make_params(self, batch: tio.SubjectsBatch) -> dict[str, Any]:
        """Return the value to add."""
        return {"value": self.value}

    def apply_transform(
        self,
        batch: tio.SubjectsBatch,
        params: dict[str, Any],
    ) -> tio.SubjectsBatch:
        """Add the value to each 5D image tensor."""
        for image_batch in batch.images.values():
            image_batch.data = image_batch.data + params["value"]
        return batch


subject = tio.Subject(
    image=tio.ScalarImage(torch.zeros(1, 2, 3, 4)),
    site="A",
)
batch = tio.SubjectsBatch.from_subjects([subject])
assert subject.image.data.shape == (1, 2, 3, 4)
assert batch.image.data.shape == (1, 1, 2, 3, 4)

transformed = AddValue(2)(subject)
assert isinstance(transformed, tio.Subject)
assert transformed.image.data.shape == (1, 2, 3, 4)
assert torch.all(transformed.image.data == 2)
```

The public call performs the complete round trip:

```text
Subject -> SubjectsBatch -> apply_transform -> Subject
```

An image tensor shaped `(C, I, J, K)` therefore reaches
`apply_transform` as `(B, C, I, J, K)`. For a single `Subject`,
`B` is 1. Negative dimension indices (`-3`, `-2`, `-1`) identify
the spatial axes for both single-element and multi-element batches.

When a `SubjectsBatch` is passed (e.g., from `SubjectsLoader`),
transforms that support it sample **independent parameters per batch
element** by default, so a single call produces diverse augmentations
(see [Per-instance augmentation](per-instance-augmentation.md)). Pass
`per_instance=False` to share one sampled parameter set across all
elements. Fixed parameters are not sampled and therefore remain shared.
Single inputs are unaffected.

## The `make_params` / `apply_transform` split

Every transform has two methods:

- **`make_params(batch)`**: create or sample parameters for the
  `SubjectsBatch`.
- **`apply_transform(batch, params)`**: apply those parameters to the
  `SubjectsBatch`.

This separation (inspired by Torchvision V2) means the same random
parameters are applied consistently to all images in a `Subject`.
Parameters are saved in history for inspection and inversion.

!!! warning "`apply_transform` is a low-level kernel"
    Application code should call the transform itself, for example
    `result = transform(subject)`. Calling `apply_transform` directly
    bypasses input wrapping, copying, probability handling, history
    recording, and output-type restoration. It requires a
    `SubjectsBatch`, not a `Subject`.

### Metadata in a batch

`Subject.metadata` is a `dict[str, Any]`. After batching,
`batch.metadata` is a `dict[str, list[Any]]`, with one value per batch
element:

```python
import torch
import torchio as tio

subjects = [
    tio.Subject(
        image=tio.ScalarImage(torch.zeros(1, 2, 3, 4)),
        site="A",
        age=30,
    ),
    tio.Subject(
        image=tio.ScalarImage(torch.ones(1, 2, 3, 4)),
        site="B",
        age=40,
    ),
]
batch = tio.SubjectsBatch.from_subjects(subjects)
assert batch.metadata == {"site": ["A", "B"], "age": [30, 40]}
```

The first subject defines the image, metadata, point, and bounding-box
key order of the batch. All subjects must have the same schema,
although their local key order may differ. A custom transform should
preserve that shared schema and keep every per-element list aligned
with the batch dimension.

## Scalar, range, or distribution: one class for both

Transform parameters accept three forms. No separate
`RandomNoise` class:

<!-- pytest-codeblocks:skip -->
```python
# Deterministic: always std=0.1
tio.Noise(std=0.1)

# Random: sample std ~ U(0.05, 0.2) each call
tio.Noise(std=(0.05, 0.2))

# Custom distribution: sample from any torch.distributions.Distribution
from torch.distributions import LogNormal
tio.Noise(std=LogNormal(loc=-2, scale=0.5))
```

This parsing and sampling is handled internally. Any
`torch.distributions.Distribution` can be used
for full control over the sampling strategy.

!!! note "No arguments means no augmentation"
    Augmentation transforms whose strength is sampled from a range
    (e.g. `Affine`, `Blur`, `Gamma`) default to a deterministic
    **identity** (no-op) when constructed with no arguments, and emit a
    warning. Pass a range like `(a, b)` for random augmentation, or a
    scalar for a fixed effect. Transforms that draw a random realisation
    instead of sampling a scalar parameter (e.g. `Noise`) still apply
    with their default parameters.

## Input flexibility

Transforms accept multiple input types and return the same type:

<!-- pytest-codeblocks:skip -->
```python
result = transform(subject)      # Subject → Subject
result = transform(image)        # Image → Image
result = transform(tensor)       # 4D Tensor → 4D Tensor
result = transform(ndarray)      # NumPy array → NumPy array
result = transform(sitk_image)   # SimpleITK → SimpleITK
result = transform(nifti_image)  # NiBabel → NiBabel
result = transform(data_dict)    # dict → dict (MONAI-compatible)
```

Non-Subject inputs are wrapped in a temporary Subject internally.
Spatial metadata (spacing, affine) is preserved through the
round-trip.

### MONAI interoperability

Dict input makes TorchIO transforms usable in MONAI pipelines:

<!-- pytest-codeblocks:skip -->
```python
# MONAI-style dict
data = {"image": tensor, "label": label_tensor, "age": 42}

# TorchIO transforms work directly
augmented = tio.Noise(std=0.1)(data)   # returns dict
augmented = tio.Flip(axes=(0,))(data)  # returns dict
```

Tensor values are treated as images; non-tensor values pass through
unchanged. See also [`MonaiAdapter`](../how-to/monai.md) for wrapping
MONAI transforms in TorchIO pipelines.

## Transform types

- **`SpatialTransform`**: modifies image geometry and applies to all
  images (ScalarImage and LabelMap). Spatial transforms currently raise
  an error when a `Subject` or batch contains Points or BoundingBoxes,
  because annotation-coordinate updates are not implemented yet.
- **`IntensityTransform`**: modifies voxel values. Applies only to
  ScalarImage, leaving LabelMap and annotations untouched.

## Composition

**`Compose`** runs transforms sequentially. It deep-copies the input
by default so the original data is preserved:

<!-- pytest-codeblocks:skip -->
```python
pipeline = tio.Compose([
    tio.Flip(axes=(0,)),
    tio.Noise(std=0.05),
])
result = pipeline(subject)  # original unchanged
```

**`OneOf`** picks one transform at random (with optional weights):

<!-- pytest-codeblocks:skip -->
```python
augment = tio.OneOf({
    tio.Noise(std=0.1): 0.7,
    tio.Blur(std=1.0): 0.3,
})
```

**`SomeOf`** picks N transforms:

<!-- pytest-codeblocks:skip -->
```python
augment = tio.SomeOf(
    [tio.Noise(std=0.1), tio.Blur(std=(0, 2)), tio.Gamma(log_gamma=(-0.3, 0.3))],
    num_transforms=2,
)
```

## History, traceability, and replay

Every transform records an `AppliedTransform` in the Subject's
`applied_transforms` list:

```python
import torch
import torchio as tio

subject = tio.Subject(image=tio.ScalarImage(torch.zeros(1, 2, 3, 4)))
result = tio.Noise(std=0.1)(subject)
trace = result.applied_transforms[-1]
assert trace.name == "Noise"
assert trace.params["std"] == 0.1
replayed = tio.Noise().apply_with_params(subject, trace.params)
torch.testing.assert_close(replayed.image.data, result.image.data)
```

Use `apply_with_params()` to apply an exact saved parameter dictionary
without sampling again.

This bypasses `p` and `make_params()`, but retains normal copying,
wrapping, history recording, and output-type restoration. Do not use
`apply_transform(new_subject, params)` for replay: the method requires
an already wrapped `SubjectsBatch` and omits the public-call lifecycle.

See [Write a custom transform](../how-to/custom-transform.md) for
vectorized image, batched metadata, and subject-wise examples.

## Hydra configuration

Transforms can export themselves as Hydra-compatible YAML configs
for reproducible experiment management:

<!-- pytest-codeblocks:skip -->
```python
pipeline = tio.Compose([
    tio.Flip(axes=(0, 1), p=0.5),
    tio.Noise(std=(0.05, 0.2)),
])
cfg = pipeline.to_hydra()
```

```python
{
    "_target_": "torchio.Compose",
    "transforms": [
        {"_target_": "torchio.Flip", "p": 0.5, "axes": [0, 1]},
        {"_target_": "torchio.Noise", "std": [0.05, 0.2]},
    ],
}
```

Instantiate with `hydra.utils.instantiate(cfg)`.

## GPU and differentiability

All transforms are pure PyTorch operations. Spatial transforms use
`torch.nn.functional.grid_sample`, which is differentiable and
GPU-compatible:

<!-- pytest-codeblocks:skip -->
```python
# Augmentation on GPU or MPS
subject = subject.to("cuda")  # or "mps" on Apple Silicon
result = transform(subject)   # stays on device

# Gradients flow through
with torch.enable_grad():
    result = transform(subject)
    loss = model(result.t1.data)
    loss.backward()
```
