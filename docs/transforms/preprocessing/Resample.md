# Resample

::: torchio.transforms.Resample

![plot](../../images/plots/plot_58ade710c663.png)

??? note "Source code"
    ```python
    import torchio as tio
    subject = tio.datasets.FPG()
    subject.remove_image('seg')
    resample = tio.Resample(8)
    t1_resampled = resample(subject.t1)
    subject.add_image(t1_resampled, 'Downsampled')
    subject.plot()
    ```
