# RandomLabelsToImage

::: torchio.transforms.RandomLabelsToImage

```python plot
import torch
import torchio as tio
torch.manual_seed(42)
colin = tio.datasets.Colin27(2008)
label_map = colin.cls
colin.remove_image('t1')
colin.remove_image('t2')
colin.remove_image('pd')
downsample = tio.Resample(1)
blurring_transform = tio.RandomBlur(std=0.6)
create_synthetic_image = tio.RandomLabelsToImage(
    image_key='synthetic',
    ignore_background=True,
)
transform = tio.Compose((
    downsample,
    create_synthetic_image,
    blurring_transform,
))
colin_synth = transform(colin)
colin_synth.plot()
```
