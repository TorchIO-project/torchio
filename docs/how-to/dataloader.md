# Load subjects with a DataLoader

Use `SubjectsLoader` to iterate over batches of subjects during
training. It wraps PyTorch's `DataLoader` and handles collation
automatically via `tensordict`.

## Basic usage

```python
from torch.utils.data import Dataset
import torchio as tio


class MyDataset(Dataset):
    def __init__(self, paths):
        self.subjects = [
            tio.Subject(
                image=tio.ScalarImage(p / "image.nii.gz"),
                seg=tio.LabelMap(p / "seg.nii.gz"),
            )
            for p in paths
        ]

    def __len__(self):
        return len(self.subjects)

    def __getitem__(self, idx):
        return self.subjects[idx]


dataset = MyDataset(paths)
loader = tio.SubjectsLoader(dataset, batch_size=4, num_workers=4)

for batch in loader:
    images = batch["image", "data"]   # (4, 1, H, W, D)
    segs = batch["seg", "data"]       # (4, 1, H, W, D)
    # ... train your model
```

## Accessing metadata in a batch

Scalar metadata and annotations are stored as non-tensor entries
and can be retrieved per sample:

```python
subject = tio.Subject(
    image=tio.ScalarImage("t1.nii.gz"),
    age=42,
    landmarks=tio.Points(torch.rand(5, 3)),
)

# After batching:
batch = next(iter(loader))
age = batch[0].get_non_tensor("_meta_age")       # 42
```

## Using a plain DataLoader

If you prefer not to use `SubjectsLoader`, pass `collate_subjects`
as the collation function:

```python
from torch.utils.data import DataLoader
import torchio as tio

loader = DataLoader(
    dataset,
    batch_size=4,
    collate_fn=tio.collate_subjects,
)
```

## How it works

Each `Subject` or `Image` is converted to a `TensorDict` before
stacking:

- **Images** become TensorDicts with `data` and `affine` tensors.
- **Points**, **BoundingBoxes**, and **metadata** are stored as
  non-tensor entries, so variable-size data (e.g., different numbers
  of landmarks per subject) is handled gracefully.
- `torch.stack` merges the list into a single batched `TensorDict`.

You can also call `.to_tensordict()` and `.from_tensordict(td)`
directly on `Subject` or `Image` if you need manual control.

## Loading images without a Subject

If your dataset returns individual `Image` objects (not `Subject`),
use `ImagesLoader`:

```python
class SliceDataset(Dataset):
    def __init__(self, paths):
        self.images = [tio.ScalarImage(p) for p in paths]

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        return self.images[idx]

loader = tio.ImagesLoader(SliceDataset(paths), batch_size=4)
batch = next(iter(loader))
batch["data"].shape    # (4, 1, H, W, D)
batch["affine"].shape  # (4, 4, 4)
```
