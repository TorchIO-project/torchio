# Add annotations to a Subject

Use [`Points`][torchio.Points] and
[`BoundingBoxes`][torchio.BoundingBoxes] to attach spatial annotations
to a [`Subject`][torchio.Subject] alongside its images.

## Landmarks and fiducials

If you have a set of 3D coordinates (e.g., anatomical landmarks or
fiducial markers), store them as a `Points` object:

```python
import torch
import torchio as tio

landmarks = tio.Points(
    torch.tensor([
        [128.0, 100.0, 90.0],  # anterior commissure
        [128.0, 130.0, 90.0],  # posterior commissure
    ]),
)

subject = tio.Subject(
    t1=tio.ScalarImage("t1.nii.gz"),
    landmarks=landmarks,
)
```

Coordinates are in **voxel space** (`IJK`) by default. Convert to a
different axis convention:

```python
# To world (mm) coordinates via the affine
world_coords = subject.landmarks.to_world()

# To any axis convention
ras_points = subject.landmarks.to_axes("RAS")
lpi_points = subject.landmarks.to_axes("LPI")
```

## Region-of-interest boxes

For regions of interest such as lesion detections or organ bounding
boxes, use `BoundingBoxes`:

```python
detections = tio.BoundingBoxes(
    torch.tensor([
        [50, 60, 40, 100, 110, 90],   # lesion 1
        [120, 80, 70, 160, 130, 110],  # lesion 2
    ]),
    format=tio.BoundingBoxFormat.IJKIJK,
)

subject = tio.Subject(
    t1=tio.ScalarImage("t1.nii.gz"),
    seg=tio.LabelMap("seg.nii.gz"),
    detections=detections,
)
```

## Attaching class labels

Pass an integer tensor of labels to track which class each box belongs
to:

```python
detections = tio.BoundingBoxes(
    torch.tensor([
        [50, 60, 40, 100, 110, 90],
        [120, 80, 70, 160, 130, 110],
    ]),
    format=tio.BoundingBoxFormat.IJKIJK,
    labels=torch.tensor([1, 2]),  # e.g., 1=tumor, 2=edema
)
```

## Switching box format

Convert between representations (corners vs center+size) and axis
conventions:

```python
# Corners → center + size
whd = detections.to_format(tio.BoundingBoxFormat.IJKWHD)

# To a different axis convention
from torchio import BoundingBoxFormat
ras_boxes = detections.to_format(BoundingBoxFormat("RAS", "corners"))

# Custom: KJI center+size
kji_cs = detections.to_format(BoundingBoxFormat("KJI", "center_size"))
```

## Accessing annotations from the Subject

Iterate over specific annotation types:

```python
for name, pts in subject.points().items():
    print(f"{name}: {pts.num_points} points")

for name, bbs in subject.bounding_boxes().items():
    print(f"{name}: {bbs.num_boxes} boxes")
```
