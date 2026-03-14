# RandomGamma

::: torchio.transforms.RandomGamma

```python plot
import torchio as tio
subject = tio.datasets.FPG()
subject.remove_image('seg')
transform = tio.RandomGamma(log_gamma=(-0.3, -0.3))
transformed = transform(subject)
subject.add_image(transformed.t1, 'log -0.3')
transform = tio.RandomGamma(log_gamma=(0.3, 0.3))
transformed = transform(subject)
subject.add_image(transformed.t1, 'log 0.3')
subject.plot()
```
