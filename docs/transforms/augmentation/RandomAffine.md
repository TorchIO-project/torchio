# RandomAffine

::: torchio.transforms.RandomAffine

![plot](../../images/plots/plot_a572518398fa.png)

??? note "Source code"
    ```python
    import torchio as tio
    subject = tio.datasets.Slicer('CTChest')
    ct = subject.CT_chest
    transform = tio.RandomAffine()
    ct_transformed = transform(ct)
    subject.add_image(ct_transformed, 'Transformed')
    subject.plot()
    ```
