# RemapLabels

::: torchio.transforms.RemapLabels

![plot](../../images/plots/plot_e849e446ad0a.png)

??? note "Source code"
    ```python
    import torchio as tio

    subject = tio.datasets.FPG()
    subject.remove_image('t1')

    background_labels = (0, 1, 2, 3, 4)
    csf_labels = (5, 12, 16, 47, 52, 53)
    white_matter_labels = (
        45, 46,
        66, 67,
        81, 82,
        83, 84,
        85, 86,
        87,
        89, 90,
        91, 92,
        93, 94,
    )

    not_gray_matter_labels = (
        background_labels
        + csf_labels
        + white_matter_labels
    )

    gray_matter_labels = [
        label for label in subject.GIF_COLORS
        if label not in not_gray_matter_labels
    ]

    labels_groups = (
        background_labels,
        gray_matter_labels,
        white_matter_labels,
        csf_labels,
    )
    remapping = {}
    for target, labels in enumerate(labels_groups):
        for label in labels:
            remapping[label] = target

    parcellation_to_tissues = tio.RemapLabels(remapping)
    tissues = parcellation_to_tissues(subject).seg
    subject.add_image(tissues, 'remapped')
    subject.plot()
    ```
