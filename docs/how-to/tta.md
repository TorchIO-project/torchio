# Test-time augmentation (TTA)

Test-time augmentation improves prediction accuracy by averaging
multiple augmented versions of the same input. TorchIO's invertible
transforms make this easy. The pattern mirrors
[v1's TTA workflow](https://docs.torchio.org/transforms/#invertibility).

## Basic TTA

The idea: augment the input, predict, copy the history to the
prediction, invert, then average:

<!-- pytest-codeblocks:skip -->
```python
import torch
import torchio as tio

model = ...  # your trained model
subject = tio.Subject(t1=tio.ScalarImage("t1.nii.gz"))

augment = tio.Compose([
    tio.Flip(axes=(0, 1, 2), flip_probability=0.5),
])

predictions = []
n_augmentations = 8

for _ in range(n_augmentations):
    # Augment
    augmented = augment(subject)

    # Predict
    with torch.no_grad():
        pred = model(augmented.t1.data.unsqueeze(0))

    # Wrap prediction and copy transform history
    pred_subject = tio.Subject(
        pred=tio.ScalarImage(pred.squeeze(0)),
    )
    pred_subject.applied_transforms = augmented.applied_transforms

    # Invert augmentation on the prediction
    restored = pred_subject.apply_inverse_transform(
        ignore_intensity=True,
    )
    predictions.append(restored.pred.data)

# Average predictions in the original space
mean_prediction = torch.stack(predictions).mean(0)
```

## API

**On Subject:**

<!-- pytest-codeblocks:skip -->
```python
# Get a Compose that inverts the history
inverse_transform = subject.get_inverse_transform()

# Or apply directly
restored = transformed.apply_inverse_transform()

# Skip intensity transforms (useful for TTA)
restored = transformed.apply_inverse_transform(ignore_intensity=True)
```

**Standalone function** (works on any type with history):

<!-- pytest-codeblocks:skip -->
```python
restored = tio.apply_inverse_transform(data)
```

## Which transforms are invertible?

| Transform | Invertible | Notes |
|-----------|-----------|-------|
| `Flip` | ✅ | Self-inverse (flip twice = identity) |
| `Crop` | ✅ | Inverse is Pad (lost voxels filled with zeros) |
| `Pad` | ✅ | Inverse is Crop |
| `Resample` | ✅ | Restores the original output grid |
| `Affine` | ✅ | Uses the inverse affine matrix |
| `ElasticDeformation` | ✅ | Negates the sampled displacement field |
| `Spatial` | ✅ | Inverts resampling, affine, and elastic parts together |
| `Noise` | ❌ | Skipped silently when ``ignore_intensity=True`` |

Non-invertible transforms are **skipped with a warning** (not
errored), so TTA works even with mixed pipelines:

<!-- pytest-codeblocks:skip -->
```python
pipeline = tio.Compose([
    tio.Flip(axes=(0, 1, 2), flip_probability=0.5),
    tio.Noise(std=0.1),  # skipped during inversion
])
transformed = pipeline(subject)
restored = transformed.apply_inverse_transform()  # only Flip is inverted
```

Use ``warn=False`` to suppress the warnings.
