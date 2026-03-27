# Transform design

TorchIO transforms are `torch.nn.Module` subclasses that operate on
Subjects, Images, or raw Tensors.

## The `make_params` / `apply` split

Every transform has two methods:

- **`make_params(subject)`** — sample random parameters (called once
  per transform invocation).
- **`apply(subject, params)`** — apply the transform using those
  parameters.

This separation (inspired by Torchvision V2) means the same random
parameters are applied consistently to all images, points, and bounding
boxes in a Subject.

```python
class Noise(tio.IntensityTransform):
    def __init__(self, std=0.1, **kwargs):
        super().__init__(**kwargs)
        self.std = std

    def make_params(self, subject):
        return {"std": self.std}

    def apply(self, subject, params):
        for name, image in self._get_images(subject).items():
            noise = torch.randn_like(image.data) * params["std"]
            image.set_data(image.data + noise)
        return subject
```

## Input flexibility

Transforms accept three input types and return the same type:

```python
# Subject → Subject
result = transform(subject)

# Image → Image
result = transform(image)

# 4D Tensor → 4D Tensor
result = transform(tensor)
```

Non-Subject inputs are wrapped in a temporary Subject internally.

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

## History and traceability

Every transform records an `AppliedTransform` in the Subject's
`applied_transforms` list:

```python
result = pipeline(subject)
for trace in result.applied_transforms:
    print(trace.name, trace.params)
```

This enables replay (apply the same augmentation to different data)
and inversion (undo spatial transforms for test-time augmentation).

## GPU and differentiability

All transforms are pure PyTorch operations. Spatial transforms use
`torch.nn.functional.grid_sample`, which is differentiable and
GPU-compatible:

```python
# Augmentation on GPU
subject = subject.to("cuda")
result = transform(subject)  # stays on GPU

# Gradients flow through
with torch.enable_grad():
    result = transform(subject)
    loss = model(result.t1.data)
    loss.backward()
```
