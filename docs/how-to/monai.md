# MONAI interoperability

TorchIO and MONAI can work together in both directions.

## Use MONAI transforms in TorchIO

Use `MonaiAdapter` to wrap any MONAI transform for use inside TorchIO
pipelines.

### Installation

```
pip install "torchio[monai]"
```

### Array transforms

Array transforms (e.g., `NormalizeIntensity`) are applied to each
`ScalarImage` in the subject individually:

```python
from monai.transforms import NormalizeIntensity
import torchio as tio

adapter = tio.MonaiAdapter(NormalizeIntensity())
result = adapter(subject)
```

Use `include` / `exclude` to control which images are affected:

```python
adapter = tio.MonaiAdapter(NormalizeIntensity(), include=["t1"])
```

### Dictionary transforms

Dictionary transforms (e.g., `NormalizeIntensityd`) operate on the
full subject dictionary — only the keys specified in the MONAI
transform are modified:

```python
from monai.transforms import NormalizeIntensityd

adapter = tio.MonaiAdapter(NormalizeIntensityd(keys=["t1"]))
result = adapter(subject)
```

Spatial dictionary transforms (e.g., `RandSpatialCropd`) propagate
affine changes back to the TorchIO images automatically.

### Inside a pipeline

`MonaiAdapter` works in `Compose` like any other transform:

```python
pipeline = tio.Compose([
    tio.Flip(axes=(0,), p=0.5),
    tio.MonaiAdapter(NormalizeIntensity()),
    tio.Noise(std=(0.01, 0.05)),
])
result = pipeline(subject)
```

!!! note
    `MonaiAdapter` does **not** record itself in the subject's
    transform history, because MONAI transform objects are not
    serializable.

## Use TorchIO transforms in MONAI

TorchIO transforms accept `dict[str, Tensor]` directly, so they
work in MONAI dict-based pipelines without any adapter:

```python
from monai.transforms import Compose as MonaiCompose
from monai.transforms import NormalizeIntensityd
import torchio as tio

pipeline = MonaiCompose([
    NormalizeIntensityd(keys=["image"]),
    tio.Noise(std=0.1),             # works on dicts
    tio.Flip(axes=(0,), p=0.5),     # works on dicts
])

data = {"image": tensor, "label": label_tensor}
result = pipeline(data)
```
