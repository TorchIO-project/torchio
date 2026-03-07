# Mask

::: torchio.transforms.Mask

![plot](../../images/plots/plot_612ef0e8d8bd.png)

??? note "Source code"
    ```python
    import torchio as tio
    subject = tio.datasets.Colin27()
    subject.remove_image('head')
    mask = tio.Mask('brain')
    masked = mask(subject)
    subject.add_image(masked.t1, 'Masked')
    subject.plot()
    ```
