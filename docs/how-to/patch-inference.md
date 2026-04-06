# Patch-based inference

3D medical images often don't fit in GPU memory. TorchIO provides
samplers and an aggregator to process volumes in patches.

## Dense inference with GridSampler

Extract all patches on a regular grid, run the model on each batch,
and reassemble the output:

```python
import torchio as tio
from torch.utils.data import DataLoader
from torchio.loader import SubjectsLoader

subject = tio.ScalarImage("brain.nii.gz")
subject = tio.Subject(t1=subject)

# 1. Sample patches
sampler = tio.GridSampler(subject, patch_size=64, patch_overlap=8)

# 2. Batch with DataLoader
loader = SubjectsLoader(sampler, batch_size=4)

# 3. Run inference and aggregate
aggregator = tio.PatchAggregator(
    spatial_shape=subject.spatial_shape,
    overlap_mode="hann",
    patch_overlap=8,
)

for batch in loader:
    input_tensor = batch.t1.data
    output = model(input_tensor)
    locations = [
        batch.metadata["patch_location"][i]
        for i in range(batch.batch_size)
    ]
    aggregator.add_batch(output, locations)

result = aggregator.get_output()
```

## Overlap modes

| Mode | Best for | How it works |
|------|----------|-------------|
| `"crop"` | Argmax segmentation | Keeps only the non-overlapping center of each patch |
| `"average"` | Probabilistic outputs | Averages all overlapping predictions |
| `"hann"` | Continuous outputs | Weights with a Hann window for smooth blending |

## Training with random samplers

For training, use `UniformSampler`, `WeightedSampler`, or
`LabelSampler`. These are `IterableDataset`s that yield patches
on-the-fly:

```python
sampler = tio.UniformSampler(subject, patch_size=64, num_patches=200)
loader = SubjectsLoader(sampler, batch_size=8)

for batch in loader:
    output = model(batch.t1.data)
    loss = criterion(output, batch.seg.data)
    loss.backward()
```

### WeightedSampler

Sample more patches from regions of interest using a probability map:

```python
sampler = tio.WeightedSampler(
    subject,
    patch_size=64,
    probability_map="sampling_weights",
    num_patches=200,
)
```

### LabelSampler

Center patches on labeled voxels, with optional per-class weights:

```python
sampler = tio.LabelSampler(
    subject,
    patch_size=64,
    label_name="seg",
    label_probabilities={0: 0.5, 1: 0.5},
    num_patches=200,
)
```

## Downsampled outputs

If your model produces spatially smaller outputs (e.g., a feature
encoder with stride 2), pass `output_shape` to the aggregator:

```python
aggregator = tio.PatchAggregator(
    spatial_shape=(256, 256, 176),
    output_shape=(128, 128, 88),
    overlap_mode="average",
)
```

## Multiple outputs

Pass a dict of tensors to aggregate multiple outputs simultaneously:

```python
aggregator.add_batch(
    {"segmentation": seg_output, "embedding": emb_output},
    locations,
)
seg = aggregator.get_output("segmentation")
emb = aggregator.get_output("embedding")
```
