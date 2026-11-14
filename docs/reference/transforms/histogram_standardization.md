# HistogramStandardization

::: torchio.HistogramStandardization
    options:
      show_root_heading: false

## Landmark computation

Use `num_workers` to load training images and compute their percentiles in
parallel. The default, `num_workers=0`, processes them sequentially.

```python
import torch
import torchio as tio
from torchio.transforms.intensity.histogram_standardization import (
    compute_histogram_landmarks,
)

# Paths to training images can be used instead of in-memory images.
generator = torch.Generator().manual_seed(970)
images = [
    tio.ScalarImage(torch.randn(1, 8, 8, 8, generator=generator))
    for _ in range(4)
]
sequential = compute_histogram_landmarks(images)
parallel = compute_histogram_landmarks(images, num_workers=2)
torch.testing.assert_close(parallel, sequential, rtol=0, atol=0)
```

Choose the number of workers according to available memory and storage speed.
Each worker may hold a full image, its mask, and intermediate percentile arrays.
For large datasets, pass file paths rather than loading all images beforehand.
Using more workers is not always faster, especially for small datasets or when
other numerical libraries already use multiple CPU threads.

Workers are threads, so a custom `masking_method` can be a local function or
lambda. It must be thread-safe and must not modify shared state or input images.
Results are collected in input order before averaging the landmarks.

::: torchio.transforms.intensity.histogram_standardization.compute_histogram_landmarks
    options:
      show_root_heading: true
