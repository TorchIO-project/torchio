# Transform design

TorchIO transforms are `torch.nn.Module` subclasses. They accept
Subjects, Images, Tensors, NumPy arrays, SimpleITK images, NiBabel
images, MONAI-style dicts, ``ImagesBatch``, or ``SubjectsBatch``,
and always return the same type.

## Unified batch architecture

Internally, **all inputs are converted to a ``SubjectsBatch``**
before the transform runs. A single ``Image`` becomes a batch of
size 1; a ``SubjectsBatch`` from a ``DataLoader`` passes through
directly. This means transform authors write **one method** that
works identically for single samples and batches:

<!-- pytest-codeblocks:skip -->
```python
class Flip(SpatialTransform):
    def apply_transform(self, batch, params):
        dims = [a - 3 for a in params["axes"]]
        for name, img_batch in self._get_images(batch).items():
            img_batch._data = torch.flip(img_batch.data, dims)
        return batch
```

The negative dim indexing (``-3``, ``-2``, ``-1`` for spatial axes)
works for both 5D ``(B, C, I, J, K)`` batch tensors and 5D
``(1, C, I, J, K)`` single-sample tensors.

When a ``SubjectsBatch`` is passed (e.g., from ``SubjectsLoader``),
``make_params`` is called **once** and the same parameters are
applied to all samples, enabling vectorised batch transforms on
GPU.

## The `make_params` / `apply` split

Every transform has two methods:

- **`make_params(subject)`**: sample random parameters (called once
  per transform invocation).
- **`apply(subject, params)`**: apply the transform using those
  parameters.

This separation (inspired by Torchvision V2) means the same random
parameters are applied consistently to all images, points, and bounding
boxes in a Subject. Params are saved in history for replay.

## Scalar, range, or distribution: one class for both

Transform parameters accept three forms. No separate
``RandomNoise`` class:

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

This is powered by `ParameterRange`, which handles all the parsing
and sampling. Any ``torch.distributions.Distribution`` can be used
for full control over the sampling strategy.

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

- **`SpatialTransform`**: modifies geometry. Applies to all images
  (ScalarImage and LabelMap) and transforms attached Points and
  BoundingBoxes.
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
    [tio.Noise(), tio.Blur(), tio.Gamma()],
    num_transforms=2,
)
```

## History, traceability, and replay

Every transform records an `AppliedTransform` in the Subject's
`applied_transforms` list:

<!-- pytest-codeblocks:skip -->
```python
result = pipeline(subject)
for trace in result.applied_transforms:
    print(trace.name, trace.params)
```

**Replay** applies the exact same augmentation to different data:

<!-- pytest-codeblocks:skip -->
```python
# Get the params from history
params = result.applied_transforms[0].params

# Replay on a new subject
noise = tio.Noise(std=0.1)
replayed = noise.apply_transform(new_subject, params)
```

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
