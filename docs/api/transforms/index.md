# Transforms API

The transforms API is split into core building blocks, augmentation transforms,
and preprocessing transforms.

For usage examples and conceptual guidance, see the
[Transforms guide](../../transforms/index.md),
[Augmentation guide](../../transforms/augmentation/index.md), and
[Preprocessing guide](../../transforms/preprocessing/index.md).

## Core classes and composition

::: torchio.transforms
    options:
      members:
        - Transform
        - FourierTransform
        - SpatialTransform
        - IntensityTransform
        - LabelTransform
        - Lambda
        - OneOf
        - Compose
        - train_histogram

## Random transform base class

::: torchio.transforms.augmentation
    options:
      members:
        - RandomTransform

## Normalization base class

::: torchio.transforms.preprocessing.intensity
    options:
      members:
        - NormalizationTransform

## Interpolation utilities

::: torchio.transforms.interpolation
    options:
      members: true
