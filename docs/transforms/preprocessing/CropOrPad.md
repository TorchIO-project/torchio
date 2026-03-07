# CropOrPad

::: torchio.transforms.CropOrPad
    options:
      members:
        - _get_six_bounds_parameters

```python plot
import torchio as tio
t1 = tio.datasets.Colin27().t1
crop_pad = tio.CropOrPad((512, 512, 32))
t1_pad_crop = crop_pad(t1)
subject = tio.Subject(t1=t1, crop_pad=t1_pad_crop)
subject.plot()
```
