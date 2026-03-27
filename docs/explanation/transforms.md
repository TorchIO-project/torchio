# Transform design

TorchIO transforms are `torch.nn.Module` subclasses. They accept
Subjects, Images, Tensors, NumPy arrays, SimpleITK images, NiBabel
images, or MONAI-style dicts — and always return the same type.

## The `make_params` / `apply` split

Every transform has two methods:

- **`make_params(subject)`** — sample random parameters (called once
  per transform invocation).
- **`apply(subject, params)`** — apply the transform using those
  parameters.

This separation (inspired by Torchvision V2) means the same random
parameters are applied consistently to all images, points, and bounding
boxes in a Subject. Params are saved in history for replay.

## Scalar or range — one class for both

Transform parameters accept a scalar (deterministic) or a
``(lo, hi)`` tuple (random). No separate ``RandomNoise`` class:

```python
# Deterministic: always std=0.1
tio.Noise(std=0.1)

# Random: sample std ~ U(0.05, 0.2) each call
tio.Noise(std=(0.05, 0.2))

# Both random
tio.Noise(mean=(-0.1, 0.1), std=(0.05, 0.2))
```

This is powered by `ParameterRange`, which handles all the parsing
and sampling.

## Input flexibility

Transforms accept multiple input types and return the same type:

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

- **`SpatialTransform`** — modifies geometry. Applies to all images
  (ScalarImage and LabelMap) and transforms attached Points and
  BoundingBoxes.
- **`IntensityTransform`** — modifies voxel values. Applies only to
  ScalarImage, leaving LabelMap and annotations untouched.

## Composition

**`Compose`** runs transforms sequentially. It deep-copies the input
by default so the original data is preserved:

```python
pipeline = tio.Compose([
    tio.Flip(axes=(0,)),
    tio.Noise(std=0.05),
])
result = pipeline(subject)  # original unchanged
```

**`OneOf`** picks one transform at random (with optional weights):

```python
augment = tio.OneOf({
    tio.Noise(std=0.1): 0.7,
    tio.Blur(std=1.0): 0.3,
})
```

**`SomeOf`** picks N transforms:

```python
augment = tio.SomeOf(
    [tio.Noise(), tio.Blur(), tio.Gamma()],
    num_transforms=2,
)
```

## History, traceability, and replay

Every transform records an `AppliedTransform` in the Subject's
`applied_transforms` list:

```python
result = pipeline(subject)
for trace in result.applied_transforms:
    print(trace.name, trace.params)
```

**Replay** applies the exact same augmentation to different data:

```python
# Get the params from history
params = result.applied_transforms[0].params

# Replay on a new subject
noise = tio.Noise(std=0.1)
replayed = noise.apply(new_subject, params)
```

## GPU and differentiability

All transforms are pure PyTorch operations. Spatial transforms use
`torch.nn.functional.grid_sample`, which is differentiable and
GPU-compatible:

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
