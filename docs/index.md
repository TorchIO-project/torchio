# TorchIO

TorchIO is an open-source Python library for efficient loading,
preprocessing, augmentation, and patch-based sampling of 3D medical images
in deep learning, following the design of PyTorch.

## Quick example

```python
import torch
import torchio as tio

# Load images lazily (no data read yet)
subject = tio.Subject(
    t1=tio.ScalarImage("t1.nii.gz"),
    seg=tio.LabelMap("seg.nii.gz"),
    landmarks=tio.Points(torch.tensor([[128.0, 100.0, 90.0]])),
    age=45,
)

# Slice without loading the full volume
cropped = subject.t1[:, 100:200, 100:200, 50:100]

# Or load everything
subject.load()
print(subject.t1.shape)    # (1, 256, 256, 176)
print(subject.t1.spacing)  # (1.0, 1.0, 1.0)
```

## Where to go next

<div class="grid cards" markdown>

-   **Getting started**

    ---

    Install TorchIO and run your first code.

    [:octicons-arrow-right-24: Getting started](getting-started.md)

-   **Tutorials**

    ---

    Step-by-step walkthroughs for common workflows.

    [:octicons-arrow-right-24: Tutorials](tutorials/index.md)

-   **How-to guides**

    ---

    Recipes for specific tasks: custom readers, NIfTI-Zarr, etc.

    [:octicons-arrow-right-24: How-to guides](how-to/index.md)

-   **Explanation**

    ---

    Understand the design: lazy loading, affines, backends.

    [:octicons-arrow-right-24: Explanation](explanation/index.md)

-   **API Reference**

    ---

    Complete reference for all classes and functions.

    [:octicons-arrow-right-24: Reference](reference/index.md)

</div>

## Credits

If you use this library for your research, please cite our paper:

> F. Perez-Garcia, R. Sparks, and S. Ourselin.
> *TorchIO: a Python library for efficient loading, preprocessing,
> augmentation and patch-based sampling of medical images in deep learning.*
> Computer Methods and Programs in Biomedicine (June 2021), p. 106236.
> [doi:10.1016/j.cmpb.2021.106236](https://doi.org/10.1016/j.cmpb.2021.106236).
