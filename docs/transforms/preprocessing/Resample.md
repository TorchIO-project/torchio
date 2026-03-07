# Resample

::: torchio.transforms.Resample

```python plot
import torchio as tio
subject = tio.datasets.FPG()
subject.remove_image('seg')
resample = tio.Resample(8)
t1_resampled = resample(subject.t1)
subject.add_image(t1_resampled, 'Downsampled')
subject.plot()
```
