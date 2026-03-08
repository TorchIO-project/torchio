# TorchIO

[![PyPI downloads](https://img.shields.io/pypi/dm/torchio.svg?label=PyPI%20downloads&logo=python&logoColor=white)](https://pypi.org/project/torchio/)
[![PyPI version](https://img.shields.io/pypi/v/torchio?label=PyPI%20version&logo=python&logoColor=white)](https://pypi.org/project/torchio/)
[![Conda version](https://img.shields.io/conda/v/conda-forge/torchio.svg?label=conda-forge&logo=conda-forge)](https://anaconda.org/conda-forge/torchio)
[![Google Colab notebooks](https://colab.research.google.com/assets/colab-badge.svg)](https://github.com/TorchIO-project/torchio/blob/main/tutorials/README.md)
[![Documentation status](https://github.com/TorchIO-project/torchio/actions/workflows/docs.yml/badge.svg)](https://github.com/TorchIO-project/torchio/actions/workflows/docs.yml)
[![Tests status](https://github.com/TorchIO-project/torchio/actions/workflows/tests.yml/badge.svg)](https://github.com/TorchIO-project/torchio/actions/workflows/tests.yml)
[![Code style: Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/charliermarsh/ruff/main/assets/badge/v1.json)](https://docs.astral.sh/ruff/)
[![Coverage status](https://codecov.io/gh/TorchIO-project/torchio/branch/main/graphs/badge.svg)](https://app.codecov.io/github/TorchIO-project/torchio)
[![Code quality](https://img.shields.io/scrutinizer/g/TorchIO-project/torchio.svg?label=Code%20quality&logo=scrutinizer)](https://scrutinizer-ci.com/g/TorchIO-project/torchio/?branch=main)
[![YouTube](https://img.shields.io/youtube/views/UEUVSw5-M9M?label=watch&style=social)](https://www.youtube.com/watch?v=UEUVSw5-M9M)

TorchIO is an open-source Python library for efficient loading, preprocessing,
augmentation and patch-based sampling of 3D medical images in deep learning,
following the design of PyTorch.

It includes multiple intensity and spatial transforms for data augmentation and
preprocessing.
These transforms include typical computer vision operations
such as random affine transformations and also domain-specific ones such as
simulation of intensity artifacts due to
[MRI magnetic field inhomogeneity (bias)](https://mriquestions.com/why-homogeneity.html)
or [k-space motion artifacts](http://proceedings.mlr.press/v102/shaw19a.html).

TorchIO is part of the official [PyTorch Ecosystem](https://pytorch.org/ecosystem/),
and was featured at
the [PyTorch Ecosystem Day 2021](https://pytorch.org/blog/ecosystem_day_2021) and
the [PyTorch Developer Day 2021](https://pytorch.org/blog/pytorch-developer-day-2021).

Many groups have used TorchIO for their research.
The complete list of citations is available on [Google Scholar](https://scholar.google.co.uk/scholar?cites=8711392719159421861&sciodt=0,5&hl=en), and the
[dependents list](https://github.com/TorchIO-project/torchio/network/dependents) is
available on GitHub.

The code is available on [GitHub](https://github.com/TorchIO-project/torchio).
If you like TorchIO, please go to the repository and star it!

<a class="github-button" href="https://github.com/TorchIO-project/torchio" data-icon="octicon-star" data-show-count="true" aria-label="Star TorchIO-project/torchio on GitHub">Star</a>
<script async defer src="https://buttons.github.io/buttons.js"></script>

See [Getting started](getting-started.md) for installation instructions and a
usage overview.

If you are looking for module-oriented documentation, visit the
[API reference](api/index.md).

Contributions are more than welcome.
Please check our [contributing guide](https://github.com/TorchIO-project/torchio/blob/main/CONTRIBUTING.rst)
if you would like to contribute.

If you have questions, feel free to ask in the
[discussions tab](https://github.com/TorchIO-project/torchio/discussions).

<a class="github-button" href="https://github.com/TorchIO-project/torchio/discussions" data-icon="octicon-comment-discussion" aria-label="Discuss TorchIO-project/torchio on GitHub">Discuss</a>

If you found a bug or have a feature request, please
[open an issue](https://github.com/TorchIO-project/torchio/issues).

<a class="github-button" href="https://github.com/TorchIO-project/torchio/issues" data-icon="octicon-issue-opened" data-show-count="true" aria-label="Issue TorchIO-project/torchio on GitHub">Issue</a>

## Credits

If you use this library for your research,
please cite our paper:

[F. Pérez-García, R. Sparks, and S. Ourselin. TorchIO: a Python library for
efficient loading, preprocessing, augmentation and patch-based sampling of
medical images in deep learning. Computer Methods and Programs in Biomedicine
(June 2021), p. 106236. ISSN:
0169-2607. doi:10.1016/j.cmpb.2021.106236.](https://doi.org/10.1016/j.cmpb.2021.106236)

BibTeX:

```bibtex
@article{perez-garcia_torchio_2021,
   title = {{TorchIO}: a {Python} library for efficient loading, preprocessing, augmentation and patch-based sampling of medical images in deep learning},
   journal = {Computer Methods and Programs in Biomedicine},
   pages = {106236},
   year = {2021},
   issn = {0169-2607},
   doi = {https://doi.org/10.1016/j.cmpb.2021.106236},
   url = {https://www.sciencedirect.com/science/article/pii/S0169260721003102},
   author = {P{\'e}rez-Garc{\'i}a, Fernando and Sparks, Rachel and Ourselin, S{\'e}bastien},
}
```

This project was originally supported by the following institutions:

- [Engineering and Physical Sciences Research Council (EPSRC) & UK Research and Innovation (UKRI)](https://epsrc.ukri.org/)
- [EPSRC Centre for Doctoral Training in Intelligent, Integrated Imaging In Healthcare (i4health)](https://www.ucl.ac.uk/intelligent-imaging-healthcare/) (University College London)
- [Wellcome / EPSRC Centre for Interventional and Surgical Sciences (WEISS)](https://www.ucl.ac.uk/interventional-surgical-sciences/) (University College London)
- [School of Biomedical Engineering & Imaging Sciences (BMEIS)](https://www.kcl.ac.uk/bmeis) (King's College London)

This library has been greatly inspired by
[NiftyNet](https://github.com/NifTK/NiftyNet), which is no longer maintained.
