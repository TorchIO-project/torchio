# Load subjects with a DataLoader

Use `SubjectsLoader` to iterate over batches of subjects during
training. It wraps PyTorch's `DataLoader` and returns
`SubjectsBatch` instances with stacked 5D tensors.

## Basic usage

<!-- pytest-codeblocks:skip -->
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
    images = batch.image.data   # (4, 1, H, W, D)
    segs = batch.seg.data       # (4, 1, H, W, D)
    # ... train your model
```

## Accessing metadata in a batch

Metadata is stored as lists (one value per sample):

<!-- pytest-codeblocks:skip -->
```python
batch.metadata["age"]   # [42, 35, 60, 28]
batch.metadata["name"]  # ["sub_0", "sub_1", "sub_2", "sub_3"]
```

## Unbatching

Split a batch back into individual subjects:

<!-- pytest-codeblocks:skip -->
```python
subjects = batch.unbatch()
for subject in subjects:
    print(subject.image.shape)  # (1, H, W, D)
```

## Using a plain DataLoader

If you prefer not to use `SubjectsLoader`, pass `collate_subjects`
as the collation function:

<!-- pytest-codeblocks:skip -->
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

Each image's 4D tensor is stacked into a 5D ``ImagesBatch``
``(B, C, I, J, K)``. Per-sample affine matrices are stored as a
list. Metadata is collected into lists.

## Applying transforms to batches

Transforms work directly on ``SubjectsBatch``. Parameters are
sampled once and applied identically to all samples:

<!-- pytest-codeblocks:skip -->
```python
batch = next(iter(loader))
augmented = tio.Flip(axes=(0,), p=0.5)(batch)
augmented.image.data.shape  # (4, 1, H, W, D)
```

## Loading images without a Subject

If your dataset returns individual `Image` objects (not `Subject`),
use `ImagesLoader`:

<!-- pytest-codeblocks:skip -->
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
batch.data.shape     # (4, 1, H, W, D)
batch.affines        # list of 4 AffineMatrix instances
```
