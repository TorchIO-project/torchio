# API Reference

Auto-generated documentation from source code docstrings.

## Data structures

- [Image, ScalarImage, LabelMap](image.md) -- medical image classes
- [Subject / Study](subject.md) -- container for images, annotations, and metadata
- [Points](points.md) -- sets of 3D coordinates
- [BoundingBoxes](bboxes.md) -- 3D bounding boxes with format conversion
- [Affine](affine.md) -- affine matrix class
- [Axes](axes.md) -- axis validation and conversion utilities

## Data loading

- [SubjectsLoader, ImagesLoader, SubjectsBatch, ImagesBatch](loader.md) -- batching and data loading

## Transforms

- [Transform, SpatialTransform, IntensityTransform](transforms.md) -- base classes and history
- [ParameterRange](parameter_range.md) -- scalar, range, or distribution parameters
- [Compose, OneOf, SomeOf](compose.md) -- pipeline composition

### Spatial

- [Flip](transforms/flip.md) -- flip along spatial axes
- [Crop](transforms/crop.md) -- remove border voxels
- [Pad](transforms/pad.md) -- add border voxels

### Intensity

- [Noise](transforms/noise.md) -- additive Gaussian noise

### Other

- [To](transforms/to.md) -- move data to a device or dtype
- [MonaiAdapter](transforms/monai_adapter.md) -- wrap MONAI transforms
