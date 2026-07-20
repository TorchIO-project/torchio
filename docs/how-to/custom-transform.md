# Write a custom transform

Custom transforms subclass `Transform` and implement a batch-native
kernel. TorchIO wraps every supported input as a `SubjectsBatch`,
calls the kernel, and restores the original input type.

## Transform image tensors

Image tensors inside a transform have shape `(B, C, I, J, K)`. Operate
on the leading batch dimension directly and use negative indices for
spatial dimensions when practical.

```python
from typing import Any

import torch
import torchio as tio


class AddValue(tio.IntensityTransform):
    """Add a fixed value to every image."""

    def __init__(self, value: float) -> None:
        super().__init__()
        self.value = value

    def make_params(self, batch: tio.SubjectsBatch) -> dict[str, Any]:
        """Return the value to add."""
        return {"value": self.value}

    def apply_transform(
        self,
        batch: tio.SubjectsBatch,
        params: dict[str, Any],
    ) -> tio.SubjectsBatch:
        """Add the value to all 5D image tensors."""
        for image_batch in self._get_images(batch).values():
            assert image_batch.data.ndim == 5
            image_batch.data = image_batch.data + params["value"]
        return batch


subject = tio.Subject(image=tio.ScalarImage(torch.zeros(1, 2, 3, 4)))
result = AddValue(2)(subject)
assert isinstance(result, tio.Subject)
assert result.image.data.shape == (1, 2, 3, 4)
assert torch.all(result.image.data == 2)
```

Call `transform(data)`, not `apply_transform` directly. The public call
handles copying, probability, wrapping, history, and output-type
restoration.

## Transform batched metadata

`batch.metadata` is a `dict[str, list[Any]]`. Each list must remain
aligned with `batch.batch_size`.

```python
from typing import Any

import torchio as tio


class NormalizeAge(tio.Transform):
    """Convert age in years to a fraction of a fixed maximum."""

    def __init__(self, maximum: float) -> None:
        super().__init__()
        self.maximum = maximum

    def make_params(self, batch: tio.SubjectsBatch) -> dict[str, Any]:
        """Return the normalization denominator."""
        return {"maximum": self.maximum}

    def apply_transform(
        self,
        batch: tio.SubjectsBatch,
        params: dict[str, Any],
    ) -> tio.SubjectsBatch:
        """Normalize every age in the batch."""
        batch.metadata["age"] = [
            age / params["maximum"] for age in batch.metadata["age"]
        ]
        return batch


batch = tio.SubjectsBatch.from_subjects([
    tio.Subject(age=20),
    tio.Subject(age=40),
])
result = NormalizeAge(100)(batch)
assert result.metadata["age"] == [0.2, 0.4]
```

Subjects in one batch must have compatible image, metadata, point, and
bounding-box schemas. Reordered equivalent keys are accepted, but no
field is silently discarded.

## Map a subject-oriented operation

Use `SubjectsBatch.map_subjects()` for logic that cannot be vectorized,
such as text processing or an external library that accepts one subject
at a time.

```python
from typing import Any

import torchio as tio


class NormalizeReport(tio.Transform):
    """Normalize report whitespace one subject at a time."""

    def make_params(self, batch: tio.SubjectsBatch) -> dict[str, Any]:
        """Return no parameters."""
        return {}

    def apply_transform(
        self,
        batch: tio.SubjectsBatch,
        params: dict[str, Any],
    ) -> tio.SubjectsBatch:
        """Normalize each report."""
        return batch.map_subjects(self._normalize_subject)

    @staticmethod
    def _normalize_subject(subject: tio.Subject) -> tio.Subject:
        subject.metadata["report"] = " ".join(subject.report.split())
        return subject


batch = tio.SubjectsBatch.from_subjects([
    tio.Subject(report="No   acute finding."),
    tio.Subject(report="Stable\nappearance."),
])
result = NormalizeReport()(batch)
assert result.metadata["report"] == [
    "No acute finding.",
    "Stable appearance.",
]
```

The callback must return a `Subject`. Uniform schema changes are
allowed; changes that make batch elements incompatible raise a
`ValueError`. History added by callbacks is retained, using
exact per-element histories when callback results differ.

## Apply exact parameters

Use `apply_with_params()` to apply a saved parameter dictionary without
sampling again:

```python
import torch
import torchio as tio

subject = tio.Subject(image=tio.ScalarImage(torch.zeros(1, 4, 4, 4)))
transform = tio.Noise(mean=(-1, 1), std=(0.1, 0.5))
transformed = transform(subject)
params = transformed.applied_transforms[-1].params

replayed = transform.apply_with_params(subject, params)
torch.testing.assert_close(replayed.image.data, transformed.image.data)
```

`apply_with_params()` bypasses `p` and `make_params()`, honors `copy`,
restores the input type, validates per-instance parameter dimensions,
and records the supplied parameters in history. `Compose`, `OneOf`,
`SomeOf`, `CropOrPad`, `EnsureShapeMultiple`, `MonaiAdapter`, and
`CornucopiaAdapter` do not expose a compatible exact-parameter kernel
and therefore reject this method.

For a transformed batch, `batch.history(index)` returns that element's
trace tuple. Read `batch.history(index)[-1].params` for the latest clean
parameter dictionary. Internal batching keys are never persisted in
public history.

## Handle annotations safely

Batching preserves subject- and image-level `Points` and
`BoundingBoxes`. Spatial transforms do not yet update annotation
coordinates, so they raise an error when annotations are present.
Remove annotations first or use an annotation-aware spatial operation.

See [Transform design](../concepts/transforms.md) for the execution
model and [Migrating from v1 to v2](../get-started/migration.md) for
the old and new subclass hooks.
