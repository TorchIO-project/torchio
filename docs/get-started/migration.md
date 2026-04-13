# Migrating from v1 to v2

This guide covers every breaking change between TorchIO v1 and v2.

## Quick checklist

- Replace `Random*` transform names with their base names (`RandomFlip` → `Flip`)
- Replace `path=` with positional arg or `source=` in Image constructors
- Replace `.affine` (numpy array) with `.affine.data` where a raw array is needed
- Replace `RescaleIntensity(out_min_max=...)` with `Normalize(out_min=..., out_max=...)`
- Replace `SubjectsDataset` with any `Dataset` passed to `SubjectsLoader`
- Replace `GridAggregator` with `PatchAggregator`

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
  `tensor` keyword names are gone — use positional or `source=`.
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
from torchio.transforms.histogram_standardization import (
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
    [tio.Flip(axes=(0,)), tio.Noise(std=0.1), tio.Gamma()],
    num_transforms=(1, 2),
)
```

### Operator sugar

<!-- pytest-codeblocks:skip -->
```python
pipeline = tio.Flip(axes=(0,)) + tio.Noise(std=0.1)  # Compose
artifact = tio.Ghosting() | tio.Spike()               # OneOf
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

v1 required wrapping subjects in `SubjectsDataset`.  v2 removes this
class — pass any `Dataset` returning `Subject` instances to
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

- `AffineMatrix` — the affine matrix class
- `Points`, `BoundingBoxes`, `BoundingBoxFormat` — annotation types
- `SubjectsBatch`, `ImagesBatch` — batch containers
- `Choice`, `ParameterRange` — parameter sampling utilities
- `SomeOf` — random subset composition
- `PatchAggregator` — renamed from `GridAggregator`
- `apply_inverse_transform` — standalone inverse function
