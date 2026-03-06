# RandomAffineElasticDeformation

::: torchio.transforms.RandomAffineElasticDeformation

![plot](../../images/plots/plot_62818659fa56.png)

??? note "Source code"
    ```python
    import torchio as tio
    subject = tio.datasets.Slicer('CTChest')
    ct = subject.CT_chest
    elastic_kwargs = {'max_displacement': (17, 12, 2)}
    transform = tio.RandomAffineElasticDeformation(elastic_kwargs=elastic_kwargs)
    ct_transformed = transform(ct)
    subject.add_image(ct_transformed, 'Transformed')
    subject.plot()
    ```
