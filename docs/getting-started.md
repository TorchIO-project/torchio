# Getting started

## Installation

The Python package is hosted on the
[Python Package Index (PyPI)](https://pypi.org/project/torchio/).

=== "uv"

    ```
    uv add torchio
    ```

=== "pip"

    ```
    pip install torchio
    ```

=== "conda"

    ```
    conda install -c conda-forge torchio
    ```

!!! note "Optional extras"

    TorchIO provides optional extras for additional functionality:

    - **`plot`** – Plotting support ([Matplotlib](https://matplotlib.org/), [colorcet](https://colorcet.holoviz.org/)): `torchio[plot]`
    - **`csv`** – CSV/tabular data support ([pandas](https://pandas.pydata.org/)): `torchio[csv]`
    - **`video`** – Video export ([ffmpeg-python](https://github.com/kkroening/ffmpeg-python)): `torchio[video]`
    - **`sklearn`** – Scikit-learn integration ([scikit-learn](https://scikit-learn.org/)): `torchio[sklearn]`

    Install extras with your package manager, e.g.:

    === "uv"

        ```
        uv add torchio --extra plot --extra csv
        ```

    === "pip"

        ```
        pip install "torchio[plot,csv]"
        ```

## Hello, World!

This example shows the basic usage of TorchIO, where an instance of
`SubjectsDataset` is passed to
a PyTorch `SubjectsLoader` to generate training batches
of 3D images that are loaded, preprocessed and augmented on the fly,
in parallel:

```python
import torch
import torchio as tio

# Each instance of tio.Subject is passed arbitrary keyword arguments.
# Typically, these arguments will be instances of tio.Image
subject_a = tio.Subject(
    t1=tio.ScalarImage('subject_a.nii.gz'),
    label=tio.LabelMap('subject_a.nii'),
    diagnosis='positive',
    age=36,
)

# Image files can be in any format supported by SimpleITK or NiBabel, including DICOM
subject_b = tio.Subject(
    t1=tio.ScalarImage('subject_b_dicom_folder/'),
    label=tio.LabelMap('subject_b_seg.nrrd'),
    diagnosis='negative',
    age=24,
)

# Images may also be created using PyTorch tensors or NumPy arrays
tensor_4d = torch.rand(4, 100, 100, 100)
subject_c = tio.Subject(
    t1=tio.ScalarImage(tensor=tensor_4d),
    label=tio.LabelMap(tensor=(tensor_4d > 0.5)),
    diagnosis='negative',
    age=19,
)

subjects_list = [
    subject_a,
    subject_b,
    subject_c,
]

# Let's use one preprocessing transform and one augmentation transform
# This transform will be applied only to scalar images:
rescale = tio.RescaleIntensity(out_min_max=(0, 1))

# As RandomAffine is faster then RandomElasticDeformation, we choose to
# apply RandomAffine 80% of the times and RandomElasticDeformation the rest
# Also, there is a 25% chance that none of them will be applied
spatial = tio.OneOf({
        tio.RandomAffine(): 0.8,
        tio.RandomElasticDeformation(): 0.2,
    },
    p=0.75,
)

# Transforms can be composed as in torchvision.transforms
transforms = [rescale, spatial]
transform = tio.Compose(transforms)

# SubjectsDataset is a subclass of torch.data.utils.Dataset
subjects_dataset = tio.SubjectsDataset(subjects_list, transform=transform)

# Images are processed in parallel thanks to a SubjectsLoader
# (which inherits from torch.utils.data.DataLoader)
training_loader = tio.SubjectsLoader(
    subjects_dataset,
    batch_size=4,
    num_workers=4,
    shuffle=True,
)

# Training epoch
for subjects_batch in training_loader:
    inputs = subjects_batch['t1'][tio.DATA]
    target = subjects_batch['label'][tio.DATA]
```

## Tutorials

[![Google Colab notebook](https://colab.research.google.com/assets/colab-badge.svg)](https://github.com/TorchIO-project/torchio/blob/main/tutorials/README.md)

The best way to quickly understand and try the library is the
[Jupyter Notebooks](https://github.com/TorchIO-project/torchio/blob/main/tutorials/README.md)
hosted on Google Colab.

They include multiple examples and visualization of most of the classes,
including training of a [3D U-Net](https://github.com/fepegar/unet) for
brain segmentation on $T_1$-weighted MRI with full volumes and
with subvolumes (aka patches or windows).
