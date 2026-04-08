# TorchIO

TorchIO is an open-source Python library for efficient loading,
preprocessing, augmentation, and patch-based sampling of 3D medical images
in deep learning, following the design of PyTorch.

## Quick example

<!-- pytest-codeblocks:skip -->
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

    → [Getting started](getting-started.md)

-   **Tutorials**

    ---

    Step-by-step walkthroughs for common workflows.

    → [Tutorials](tutorials/index.md)

-   **How-to guides**

    ---

    Recipes for specific tasks: custom readers, NIfTI-Zarr, etc.

    → [How-to guides](how-to/index.md)

-   **Concepts**

    ---

    Understand the design: lazy loading, affines, backends.

    → [Concepts](concepts/index.md)

-   **API Reference**

    ---

    Complete reference for all classes and functions.

    → [Reference](reference/index.md)

</div>

## Credits

If you use this library for your research, please cite our paper:

> F. Perez-Garcia, R. Sparks, and S. Ourselin.
> *TorchIO: a Python library for efficient loading, preprocessing,
> augmentation and patch-based sampling of medical images in deep learning.*
> Computer Methods and Programs in Biomedicine (June 2021), p. 106236.
> [doi:10.1016/j.cmpb.2021.106236](https://doi.org/10.1016/j.cmpb.2021.106236).

## Related projects

- [MONAI](https://monai.readthedocs.io)
- [Cornucopia](https://cornucopia.readthedocs.io/)
- [batchgenerators](https://github.com/MIC-DKFZ/batchgenerators)[[v2](https://github.com/MIC-DKFZ/batchgeneratorsv2)]
- [volumentations](https://github.com/ZFTurbo/volumentations) (low activity)
- [Rising](https://rising.readthedocs.io) (archived)
- [pymia](https://pymia.readthedocs.io) (low activity)
- [MedicalTorch](https://medicaltorch.readthedocs.io) (abandoned)
- [Eisen](https://github.com/eisen-ai/eisen-core) (abandoned)
