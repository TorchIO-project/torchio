# Clamp

::: torchio.transforms.Clamp

```python plot
import torchio as tio
subject = tio.datasets.Slicer('CTChest')
ct = subject.CT_chest
HOUNSFIELD_AIR, HOUNSFIELD_BONE = -1000, 1000
clamp = tio.Clamp(out_min=HOUNSFIELD_AIR, out_max=HOUNSFIELD_BONE)
ct_clamped = clamp(ct)
subject.add_image(ct_clamped, 'Clamped')
subject.plot()
```
