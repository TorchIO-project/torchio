# RandomAffine

::: torchio.transforms.RandomAffine

```python plot
import torchio as tio
subject = tio.datasets.Slicer('CTChest')
ct = subject.CT_chest
transform = tio.RandomAffine()
ct_transformed = transform(ct)
subject.add_image(ct_transformed, 'Transformed')
subject.plot()
```
