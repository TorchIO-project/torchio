# Migrating from v1 to v2

This guide covers every breaking change between TorchIO v1 and v2.

!!! note "Report bugs or request features"

    Hit a snag migrating, or have an idea to improve TorchIO? Please
    [start a discussion](https://github.com/TorchIO-project/torchio/discussions)
    or [open an issue](https://github.com/TorchIO-project/torchio/issues) on
    GitHub to report bugs or request features.

## Quick checklist

- Replace `Random*` transform names with their base names (`RandomFlip` → `Flip`)
- Pass explicit ranges for augmentation (renamed transforms are a no-op without arguments): `RandomAffine()` → `Affine(degrees=(-10, 10), scales=(0.9, 1.1))`
- Replace `path=` with positional arg or `source=` in Image constructors
- Replace `.affine` (numpy array) with `.affine.data` where a raw array is needed
- Replace `RescaleIntensity(out_min_max=...)` with `Normalize(out_min=..., out_max=...)`
- Replace `SubjectsDataset` with any `Dataset` passed to `SubjectsLoader`
- Replace `GridAggregator` with `PatchAggregator`
- Rewrite custom transforms to accept a `SubjectsBatch` in
  `make_params(batch)` and `apply_transform(batch, params)`

## Image construction

**v1:**

<!-- pytest-codeblocks:skip -->
```python
image = tio.ScalarImage(path="t1.nii.gz")
image = tio.ScalarImage(tensor=tensor, affine=affine_array)
```

**v2:**

<!-- pytest-codeblocks:skip -->
```python
image = tio.ScalarImage("t1.nii.gz")
image = tio.ScalarImage(tensor, affine=tio.AffineMatrix(affine_array))
```

Changes:

- First positional argument accepts a path, tensor, numpy array,
  NiBabel image, SimpleITK image, or `bytes`.  The `path` and
  `tensor` keyword names are gone: use positional or `source=`.
- `type` parameter removed.  Use `ScalarImage` or `LabelMap` directly.
- `affine` accepts `AffineMatrix` objects in addition to arrays.
- New `channels_last` parameter for tensor sources shaped
  `(I, J, K, C)`.

## Affine access

**v1:**

<!-- pytest-codeblocks:skip -->
```python
affine_array = image.affine           # np.ndarray (4, 4)
spacing = image.spacing               # tuple
direction = image.direction            # 9-tuple of floats
```

**v2:**

<!-- pytest-codeblocks:skip -->
```python
affine_obj = image.affine             # AffineMatrix object
affine_array = image.affine.data      # np.ndarray (4, 4)
spacing = image.spacing               # tuple (unchanged)
orientation = image.affine.orientation # e.g. ("R", "A", "S")
```

The `.affine` property now returns an `AffineMatrix` object.  Use
`.affine.data` when you need the raw 4×4 numpy array.

## Subject construction

**v1:**

<!-- pytest-codeblocks:skip -->
```python
subject = tio.Subject({"t1": image, "seg": label})  # dict positional arg
subject = tio.Subject(t1=image, seg=label)
```

**v2:**

<!-- pytest-codeblocks:skip -->
```python
subject = tio.Subject(t1=image, seg=label)  # keyword args only
```

The positional dictionary form is removed.  Use keyword arguments.

## Transform naming

v2 removes the `Random*` prefix.  Stochasticity is controlled by
parameter type: a scalar is deterministic, a tuple samples uniformly,
and a `Distribution` or `Choice` gives full control.

!!! warning "Renaming a `Random*` transform changes its default behavior"
    In v1, `RandomAffine()` (no arguments) applied random augmentation.
    In v2, the renamed `Affine()` (no arguments) is a deterministic
    identity (no-op) that emits a warning (randomness is opt-in). Pass a
    range like `(a, b)` for random augmentation, or a scalar for a fixed
    effect. Transforms that draw a random realisation rather than
    sampling a scalar parameter (e.g. `Noise`, `BiasField`,
    `ElasticDeformation`, `Swap`) still apply with their default
    parameters.

| v1 | v2 |
|---|---|
| `RandomFlip` | `Flip` |
| `RandomAffine` | `Affine` |
| `RandomElasticDeformation` | `ElasticDeformation` |
| `RandomNoise` | `Noise` |
| `RandomBlur` | `Blur` |
| `RandomMotion` | `Motion` |
| `RandomGhosting` | `Ghosting` |
| `RandomBiasField` | `BiasField` |
| `RandomGamma` | `Gamma` |
| `RandomSpike` | `Spike` |
| `RandomSwap` | `Swap` |
| `RandomAnisotropy` | `Anisotropy` |
| `RandomLabelsToImage` | `LabelsToImage` |
| `RescaleIntensity` | `Normalize` (alias `RescaleIntensity` available) |
| `ZNormalization` | `Standardize` (alias `ZNormalization` available) |

## Transform parameter changes

### Flip

`flip_probability` default changed from **0.5** to **1.0**.  If you
relied on the old default, set it explicitly:

<!-- pytest-codeblocks:skip -->
```python
# v1 (implicit 0.5)
tio.RandomFlip(axes=(0, 1, 2))

# v2 (explicit 0.5)
tio.Flip(axes=(0, 1, 2), flip_probability=0.5)
```

### Affine

`scales` and `degrees` now expect explicit ranges instead of
half-widths:

<!-- pytest-codeblocks:skip -->
```python
# v1: scales=0.1 means range (0.9, 1.1)
tio.RandomAffine(scales=0.1, degrees=10)

# v2: specify the range directly
tio.Affine(scales=(0.9, 1.1), degrees=(-10, 10))
```

### Normalize (was RescaleIntensity)

Tuple parameters are split into individual keyword arguments:

<!-- pytest-codeblocks:skip -->
```python
# v1
tio.RescaleIntensity(
    out_min_max=(0, 1),
    percentiles=(0.5, 99.5),
)

# v2
tio.Normalize(
    out_min=0,
    out_max=1,
    percentile_low=0.5,
    percentile_high=99.5,
)
```

Each parameter can independently be a scalar (fixed), a tuple
(uniform range), a `Distribution`, or a `Choice`.

### HistogramStandardization

Landmark computation is now a standalone function instead of a
classmethod:

<!-- pytest-codeblocks:skip -->
```python
# v1
landmarks = tio.HistogramStandardization.train(paths)
transform = tio.HistogramStandardization({"t1": landmarks})

# v2
from torchio.transforms.intensity.histogram_standardization import (
    compute_histogram_landmarks,
)
landmarks = compute_histogram_landmarks(images)
transform = tio.HistogramStandardization(landmarks, include=["t1"])
```

One instance per modality.  For multi-modal subjects, compose:

<!-- pytest-codeblocks:skip -->
```python
tio.Compose([
    tio.HistogramStandardization(t1_landmarks, include=["t1"]),
    tio.HistogramStandardization(t2_landmarks, include=["t2"]),
])
```

## Custom transforms

### Rewrite the transform hooks

In v1, custom transforms implemented a subject-level hook:

<!-- pytest-codeblocks:skip -->
```python
# v1
def apply_transform(self, subject: tio.Subject) -> tio.Subject:
    ...
```

In v2, parameter creation and application are separate batch-level
hooks:

<!-- pytest-codeblocks:skip -->
```python
# v2
def make_params(self, batch: tio.SubjectsBatch) -> dict[str, Any]:
    ...

def apply_transform(
    self,
    batch: tio.SubjectsBatch,
    params: dict[str, Any],
) -> tio.SubjectsBatch:
    ...
```

Call the transform normally rather than calling either hook yourself.
For a single subject, TorchIO performs this conversion automatically:

```text
Subject -> SubjectsBatch -> apply_transform -> Subject
```

The following complete transform works for both a single `Subject` and
a `SubjectsBatch`:

```python
from typing import Any

import torch
import torchio as tio


class AddValue(tio.Transform):
    """Add a fixed value to every batched image."""

    def __init__(self, value: float) -> None:
        super().__init__()
        self.value = value
        self.received_shape: tuple[int, ...] | None = None

    def make_params(self, batch: tio.SubjectsBatch) -> dict[str, Any]:
        """Return the value to add."""
        return {"value": self.value}

    def apply_transform(
        self,
        batch: tio.SubjectsBatch,
        params: dict[str, Any],
    ) -> tio.SubjectsBatch:
        """Add the value to every image tensor."""
        for image_batch in batch.images.values():
            self.received_shape = tuple(image_batch.data.shape)
            image_batch.data = image_batch.data + params["value"]
        return batch


subject = tio.Subject(image=tio.ScalarImage(torch.zeros(1, 2, 3, 4)))
transform = AddValue(2)
result = transform(subject)
assert isinstance(result, tio.Subject)
assert transform.received_shape == (1, 1, 2, 3, 4)
assert result.image.data.shape == (1, 2, 3, 4)
assert torch.all(result.image.data == 2)
```

The image is 4D `(C, I, J, K)` before and after the public call, but it
is 5D `(B, C, I, J, K)` inside `apply_transform`. For a single subject,
`B` is 1.

!!! warning "`apply_transform` is not a public replay method"
    It is the low-level batch kernel. Calling it directly bypasses
    wrapping, copying, probability handling, history recording, and
    output-type restoration. Pass a supported input to the transform
    itself instead.

### Migrate metadata access

In v1, subject metadata values were scalars or arbitrary objects. In a
v2 batch, each metadata key maps to a list containing one value per
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

Treat `batch.metadata` as `dict[str, list[Any]]`. Metadata transforms
must keep each list aligned with the batch dimension. Subjects in one
batch must have equivalent image names and metadata keys. The first
subject determines the shared key order; later subjects may use a
different local order, but custom transforms should preserve the batch
schema rather than adding, removing, or renaming keys for only some
elements.

### Choose deterministic or per-instance behavior

A fixed scalar is not sampled: transforms such as `Gamma` use that
value for every batch element. For built-in stochastic transforms that
support per-instance sampling, ranges and distributions produce
independent parameters for each batch element by default. Set
`per_instance=False` to share one sampled parameter set:

```python
import torchio as tio

independent = tio.Gamma(log_gamma=(-0.3, 0.3))
shared = tio.Gamma(log_gamma=(-0.3, 0.3), per_instance=False)
deterministic = tio.Gamma(log_gamma=0.2)
```

Custom transforms do not gain per-instance sampling automatically.
Unless a transform explicitly implements and advertises that
capability, its parameters remain batch-shared. See
[Per-instance augmentation](../concepts/per-instance-augmentation.md)
for the capability contract and stochastic-realisation caveats.

### Migrate inherently per-subject logic

Prefer vectorized operations on 5D tensors or metadata lists. If logic
must call a subject-oriented external API, the current low-level escape
hatch is to unbatch, process every subject without changing its schema,
restack, and adopt the prior history:

```python
from typing import Any

import torch
import torchio as tio


class StripIdentifier(tio.Transform):
    """Strip whitespace from subject identifiers."""

    def make_params(self, batch: tio.SubjectsBatch) -> dict[str, Any]:
        """Return no parameters."""
        return {}

    def apply_transform(
        self,
        batch: tio.SubjectsBatch,
        params: dict[str, Any],
    ) -> tio.SubjectsBatch:
        """Process metadata one subject at a time."""
        subjects = batch.unbatch()
        for subject in subjects:
            identifier = subject.metadata["identifier"]
            subject.metadata["identifier"] = identifier.strip()
        rebuilt = tio.SubjectsBatch.from_subjects(subjects)
        rebuilt.adopt_history(batch, subjects)
        return rebuilt


subject = tio.Subject(
    image=tio.ScalarImage(torch.zeros(1, 2, 3, 4)),
    identifier=" sub-01 ",
)
result = StripIdentifier()(subject)
assert result.identifier == "sub-01"
```

This pattern is more expensive than vectorized code and requires every
resulting subject to retain a compatible image and metadata schema. A
supported mapping utility is planned, but it is not part of the current
API.

## New features

### Choice

Sample from a discrete set of values:

<!-- pytest-codeblocks:skip -->
```python
tio.Affine(degrees=tio.Choice([-90, 0, 90, 180]))
```

### SomeOf

Apply a random subset of transforms:

<!-- pytest-codeblocks:skip -->
```python
tio.SomeOf(
    [tio.Flip(axes=(0,)), tio.Noise(std=0.1), tio.Gamma(log_gamma=(-0.3, 0.3))],
    num_transforms=(1, 2),
)
```

### Operator sugar

<!-- pytest-codeblocks:skip -->
```python
pipeline = tio.Flip(axes=(0,)) + tio.Noise(std=0.1)  # Compose
artifact = tio.Ghosting(intensity=(0.5, 1)) | tio.Spike(intensity=(1, 3))  # OneOf
```

### Compose copy control

`Compose` deep-copies the input once, then all inner transforms
operate in-place.  Disable with `copy=False` for nested pipelines:

<!-- pytest-codeblocks:skip -->
```python
inner = tio.Compose([tio.Flip(axes=(0,))], copy=False)
outer = tio.Compose([inner, tio.Noise(std=0.1)])
```

## Data loading

### SubjectsDataset removed

v1 required wrapping subjects in `SubjectsDataset`.
v2 removes this class. Pass any `Dataset` returning `Subject` instances to
`SubjectsLoader`:

<!-- pytest-codeblocks:skip -->
```python
# v1
dataset = tio.SubjectsDataset(subjects, transform=augment)
loader = DataLoader(dataset, batch_size=4)

# v2
loader = tio.SubjectsLoader(subjects, transform=augment, batch_size=4)
```

### Queue

<!-- pytest-codeblocks:skip -->
```python
# v1
dataset = tio.SubjectsDataset(subjects)
sampler = tio.UniformSampler(patch_size=96)
queue = tio.Queue(dataset, max_length=300, samples_per_volume=10)

# v2
queue = tio.Queue(
    subjects,
    patch_sampler=tio.UniformSampler(patch_size=96),
    max_length=300,
    patches_per_volume=10,
)
```

### PatchAggregator (was GridAggregator)

<!-- pytest-codeblocks:skip -->
```python
# v1
aggregator = tio.GridAggregator(sampler)

# v2
aggregator = tio.PatchAggregator(sampler)
```

## Transform history

v2 simplifies the history API:

<!-- pytest-codeblocks:skip -->
```python
# Both versions
restored = subject.apply_inverse_transform()
inverse = subject.get_inverse_transform()

# v1 only (removed in v2)
subject.history
subject.get_applied_transforms()
subject.get_composed_history()
```

## Imports

All transforms are available at the top level:

<!-- pytest-codeblocks:skip -->
```python
# v1
from torchio.transforms import RandomFlip, RandomAffine
from torchio.transforms.augmentation.intensity import RandomNoise

# v2
import torchio as tio
tio.Flip
tio.Affine
tio.Noise
```

New exports in v2:

- `AffineMatrix`: the affine matrix class
- `Points`, `BoundingBoxes`, `BoundingBoxFormat`: annotation types
- `SubjectsBatch`, `ImagesBatch`: batch containers
- `Choice`: discrete parameter sampling utility
- `SomeOf`: random subset composition
- `PatchAggregator`: renamed from `GridAggregator`
- `apply_inverse_transform`: standalone inverse function
