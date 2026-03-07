# Preprocessing

## Intensity

| Transform | Description |
|-----------|-------------|
| [`RescaleIntensity`](RescaleIntensity.md) | Rescale intensity values to a certain range |
| [`ZNormalization`](ZNormalization.md) | Subtract mean and divide by standard deviation |
| [`HistogramStandardization`](HistogramStandardization.md) | Standardize histogram of foreground intensities |
| [`Mask`](Mask.md) | Mask an image using a label map |
| [`Clamp`](Clamp.md) | Clamp intensity values into a range |
| [`PCA`](PCA.md) | Reduce the number of channels using PCA |
| [`To`](To.md) | Change the dtype or device of image data |

::: torchio.transforms.preprocessing.intensity.NormalizationTransform
    options:
      show_root_heading: true

## Spatial

| Transform | Description |
|-----------|-------------|
| [`CropOrPad`](CropOrPad.md) | Crop or pad an image to a target shape |
| [`Crop`](Crop.md) | Crop an image |
| [`Pad`](Pad.md) | Pad an image |
| [`Resize`](Resize.md) | Resize an image to a target shape |
| [`Resample`](Resample.md) | Resample an image to a different voxel spacing |
| [`ToCanonical`](ToCanonical.md) | Reorder data to canonical orientation |
| [`ToOrientation`](ToOrientation.md) | Reorder data to a given orientation |
| [`ToReferenceSpace`](ToReferenceSpace.md) | Resample to a reference image space |
| [`Transpose`](Transpose.md) | Transpose spatial dimensions |
| [`EnsureShapeMultiple`](EnsureShapeMultiple.md) | Pad to ensure shape is a multiple of a value |
| [`CopyAffine`](CopyAffine.md) | Copy the affine matrix from one image to another |

## Label

| Transform | Description |
|-----------|-------------|
| [`RemapLabels`](RemapLabels.md) | Remap integer labels in a segmentation |
| [`RemoveLabels`](RemoveLabels.md) | Remove labels from a segmentation |
| [`SequentialLabels`](SequentialLabels.md) | Map labels to sequential integers |
| [`OneHot`](OneHot.md) | Convert a label map to one-hot encoding |
| [`Contour`](Contour.md) | Create a binary image with contour of each label |
| [`KeepLargestComponent`](KeepLargestComponent.md) | Keep the largest connected component |
