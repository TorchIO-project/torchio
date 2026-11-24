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

# File paths can be passed instead of in-memory images.
images = [tio.ScalarImage(torch.randn(1, 8, 8, 8)) for _ in range(4)]
landmarks = compute_histogram_landmarks(images, num_workers=2)
```

Choose the number of workers according to available memory and storage speed,
as each worker holds a full image while it is processed. Passing a lazy
`ScalarImage` loads its data and leaves it cached on the object, so pass file
paths for large datasets.

Workers are threads, so a custom `masking_method` can be a local function or
lambda, but it must be thread-safe and must not modify input images.

::: torchio.transforms.intensity.histogram_standardization.compute_histogram_landmarks
    options:
      show_root_heading: true
