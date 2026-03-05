# Mask

::: torchio.transforms.Mask

```python plot
import torchio as tio
subject = tio.datasets.Colin27()
subject.remove_image('head')
mask = tio.Mask('brain')
masked = mask(subject)
subject.add_image(masked.t1, 'Masked')
subject.plot()
```
