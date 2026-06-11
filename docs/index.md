# TorchIO

TorchIO is an open-source Python library for efficient loading,
preprocessing, augmentation, and patch-based sampling of 3D medical images
in deep learning, following the design of PyTorch.

> *Tools like TorchIO are a symptom of the maturation of medical AI research
> using deep learning techniques*.
>
> Jack Clark, Policy Director at [OpenAI](https://openai.com/), Co-Founder and
> Head of Policy of Anthropic ([link](https://jack-clark.net/2020/03/17/))

!!! warning "TorchIO v2 is an experimental pre-release"

    This site documents **TorchIO v2**, which is under active development and
    may contain bugs and breaking changes. The current stable release is **v1**
    (`pip install torchio`). To try v2, run `pip install --pre torchio`. Use the
    version selector in the header to switch between versions.

## Quick example

Augment a whole batch of subjects on the GPU in a few lines:

<!-- pytest-codeblocks:skip -->
```python
import torchio as tio

# Build subjects (lazy: only headers are read until .data is accessed)
dirs = ["sub-01", "sub-02", "sub-03", "sub-04"]
subjects = [
    tio.Subject(
        t1=tio.ScalarImage(f"{dir}/t1.nii.gz"),
        seg=tio.LabelMap(f"{dir}/seg.nii.gz"),
    )
    for dir in dirs
]

# Random augmentation pipeline
transform = tio.Compose([
    tio.Flip(),
    tio.Affine(degrees=(-15, 15)),
    tio.Standardize(),
    tio.Noise(std=(0, 0.1)),
])

# Stack into a batch and augment all subjects on the GPU in one call
batch = tio.SubjectsBatch.from_subjects(subjects).to("cuda")  # or "mps"
augmented = transform(batch)

print(augmented.t1.data.shape)   # (4, 1, 256, 256, 176)
print(augmented.t1.data.device)  # cuda:0
```

## Where to go next

New to TorchIO? Start with the [quickstart](get-started/quickstart.md).
Upgrading from v1? See the [migration guide](get-started/migration.md).

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
- [BatchAug](https://github.com/halleewong/batchaug)
- [volumentations](https://github.com/ZFTurbo/volumentations) (low activity)
- [Rising](https://rising.readthedocs.io) (archived)
- [pymia](https://pymia.readthedocs.io) (low activity)
- [MedicalTorch](https://medicaltorch.readthedocs.io) (abandoned)
- [Eisen](https://github.com/eisen-ai/eisen-core) (abandoned)
