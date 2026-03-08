# Data API

The data API covers image containers, subjects, datasets, patch sampling,
inference helpers, and image I/O.

For usage examples and narrative guidance, see
[Data structures](../data/index.md) and
[Patch-based pipelines](../patches/index.md).

## Core data structures

::: torchio.data
    options:
      members:
        - Image
        - ScalarImage
        - LabelMap
        - Subject
        - SubjectsDataset
        - SubjectsLoader
        - Queue

## Patch sampling

::: torchio.data.sampler
    options:
      members: true

## Inference utilities

::: torchio.data.inference
    options:
      members: true

## Image I/O

::: torchio.data.io
    options:
      members: true
