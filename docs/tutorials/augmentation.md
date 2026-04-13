# Augmentation pipelines

This tutorial shows how to build data augmentation pipelines with
TorchIO transforms.  You will learn how to apply spatial, intensity,
and artifact transforms, compose them into pipelines, and use
`Choice` for discrete parameter sampling.

## A simple pipeline

<!-- pytest-codeblocks:skip -->
```python
import torchio as tio

augmentation = tio.Compose([
    tio.Flip(axes=(0, 1, 2), flip_probability=0.5),
    tio.Affine(degrees=10, translation=5),
    tio.Noise(std=(0.01, 0.1)),
    tio.Gamma(log_gamma=(-0.3, 0.3)),
])

subject = tio.Subject(
    t1=tio.ScalarImage("t1.nii.gz"),
    seg=tio.LabelMap("seg.nii.gz"),
)
augmented = augmentation(subject)
```

Every call to `augmentation(subject)` produces a different result.
Spatial transforms (Flip, Affine) are applied consistently to all
images in the subject — the T1 and segmentation are transformed
together.

## Deterministic vs random

The same class handles both cases.  A scalar gives a fixed value;
a tuple gives a uniform range:

<!-- pytest-codeblocks:skip -->
```python
# Always rotate 90°
tio.Affine(degrees=90)

# Rotate uniformly between -15° and 15°
tio.Affine(degrees=15)

# Rotate uniformly between 5° and 20°
tio.Affine(degrees=(5, 20))
```

For discrete choices, use `Choice`:

<!-- pytest-codeblocks:skip -->
```python
# Rotate by exactly -90, 0, 90, or 180 degrees
tio.Affine(degrees=tio.Choice([-90, 0, 90, 180]))
```

You can mix `Choice`, ranges, and fixed values per axis:

<!-- pytest-codeblocks:skip -->
```python
# Fixed along I, random along J, discrete along K
tio.Affine(degrees=(0, (-10, 10), tio.Choice([-90, 0, 90])))
```

## Probability control

Every transform has a `p` parameter (probability of being applied):

<!-- pytest-codeblocks:skip -->
```python
# 50% chance of adding noise
tio.Noise(std=0.1, p=0.5)
```

## Composition strategies

### Compose — apply all in sequence

<!-- pytest-codeblocks:skip -->
```python
pipeline = tio.Compose([
    tio.Affine(degrees=10),
    tio.Noise(std=0.05),
    tio.Gamma(log_gamma=0.3),
])
```

### OneOf — pick one at random

<!-- pytest-codeblocks:skip -->
```python
artifact = tio.OneOf({
    tio.Ghosting(intensity=0.5): 0.4,
    tio.Spike(intensity=1.0): 0.3,
    tio.Motion(degrees=5): 0.3,
})
```

### SomeOf — pick N at random

<!-- pytest-codeblocks:skip -->
```python
augment = tio.SomeOf(
    [
        tio.Flip(axes=(0, 1, 2)),
        tio.Blur(std=1.0),
        tio.Noise(std=0.05),
        tio.Gamma(log_gamma=0.3),
    ],
    num_transforms=(1, 3),  # apply 1 to 3 of the 4
)
```

### Operator sugar

You can use `+` for Compose and `|` for OneOf:

<!-- pytest-codeblocks:skip -->
```python
pipeline = tio.Flip(axes=(0,)) + tio.Noise(std=0.05) + tio.Gamma()
artifact = tio.Ghosting() | tio.Spike() | tio.Motion()
```

## Preprocessing + augmentation

A common pattern separates preprocessing (applied once) from
augmentation (applied each epoch):

<!-- pytest-codeblocks:skip -->
```python
preprocessing = tio.Compose([
    tio.Resample(target=1.0),            # isotropic 1mm
    tio.CropOrPad(target_shape=128),     # fixed shape
    tio.Normalize(out_min=-1, out_max=1), # rescale to [-1, 1]
])

augmentation = tio.Compose([
    tio.Flip(axes=(0, 1, 2), flip_probability=0.5),
    tio.Affine(degrees=15, translation=5, p=0.8),
    tio.OneOf({
        tio.Noise(std=(0.01, 0.1)): 0.5,
        tio.Blur(std=(0.5, 2.0)): 0.3,
        tio.BiasField(coefficients=0.5): 0.2,
    }),
    tio.Gamma(log_gamma=(-0.3, 0.3), p=0.5),
])

full_pipeline = preprocessing + augmentation
```

## MRI artifact simulation

TorchIO includes several MRI-specific artifact transforms:

<!-- pytest-codeblocks:skip -->
```python
artifacts = tio.Compose([
    tio.Motion(degrees=5, num_transforms=2, p=0.3),
    tio.Ghosting(num_ghosts=5, intensity=0.5, p=0.3),
    tio.Spike(num_spikes=1, intensity=1.5, p=0.2),
    tio.Anisotropy(downsampling=3, p=0.3),
    tio.BiasField(coefficients=0.5, p=0.3),
])
```

These are useful for training models that are robust to common MRI
acquisition artifacts.

## Label-aware transforms

Some transforms only affect label maps:

<!-- pytest-codeblocks:skip -->
```python
label_pipeline = tio.Compose([
    tio.SequentialLabels(),                # renumber to 0, 1, 2, ...
    tio.RemoveLabels([4, 5]),              # drop unwanted labels
    tio.KeepLargestComponent(labels=[1]),   # clean up label 1
])
```

## SynthSeg-style synthesis

Generate synthetic images from label maps:

<!-- pytest-codeblocks:skip -->
```python
synth = tio.Compose([
    tio.LabelsToImage(label_key="seg"),
    tio.Blur(std=(0.5, 1.5)),
    tio.BiasField(coefficients=0.5),
    tio.Gamma(log_gamma=(-0.3, 0.3)),
    tio.Noise(std=(0.01, 0.05)),
])
```

## Summary

| Goal | Transform(s) |
|------|-------------|
| Random flipping | `Flip` |
| Rotation / scaling / shearing | `Affine` |
| Elastic deformation | `ElasticDeformation` |
| Change resolution | `Resample`, `Resize` |
| Fixed shape | `CropOrPad` |
| Intensity normalization | `Normalize`, `Standardize`, `HistogramStandardization` |
| Gaussian noise | `Noise` |
| Gaussian blur | `Blur` |
| Gamma correction | `Gamma` |
| Bias field | `BiasField` |
| MRI motion | `Motion` |
| MRI ghosting | `Ghosting` |
| K-space spikes | `Spike` |
| Simulate low-res axis | `Anisotropy` |
| Label cleanup | `RemoveLabels`, `KeepLargestComponent`, `SequentialLabels` |
| Synthetic images | `LabelsToImage` |
| Self-supervised | `Swap` |
