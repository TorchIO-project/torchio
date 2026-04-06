# Patch-based training

For training on large 3D volumes that don't fit in GPU memory, TorchIO
provides a [`Queue`][torchio.data.Queue] that loads subjects
in background threads, applies transforms, and extracts random patches
into a buffer.

## Basic usage

<!-- pytest-codeblocks:skip -->
```python
import torchio as tio
from torchio.loader import SubjectsLoader

subjects = [
    tio.Subject(t1=tio.ScalarImage(path))
    for path in training_paths
]

transform = tio.Compose([
    tio.Flip(axes=(0,), p=0.5),
    tio.Noise(std=0.1),
])

sampler = tio.UniformSampler(subjects[0], patch_size=64)

queue = tio.Queue(
    subjects,
    patch_sampler=sampler,
    max_length=300,
    patches_per_volume=10,
    num_workers=4,
    transform=transform,
)

loader = SubjectsLoader(queue, batch_size=16)

for epoch in range(num_epochs):
    for batch in loader:
        inputs = batch.t1.data
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
```

## How it works

1. **Subjects** are loaded and preprocessed in background threads
   (`num_workers` controls the parallelism)
2. **Patches** are extracted via the sampler (up to
   `patches_per_volume` per subject)
3. Patches accumulate in a **buffer** (up to `max_length`)
4. When the buffer is full, patches are shuffled and yielded
5. The external `SubjectsLoader` batches them for GPU training

## Memory estimation

<!-- pytest-codeblocks:skip -->
```python
print(queue.max_memory_pretty)  # e.g., "200.0 MiB"
```

## Distributed training

Pass a `DistributedSampler` as `subject_sampler` so each rank
processes its own subset of subjects:

<!-- pytest-codeblocks:skip -->
```python
from torch.utils.data.distributed import DistributedSampler

subject_sampler = DistributedSampler(subjects, shuffle=True)
queue = tio.Queue(
    subjects,
    patch_sampler=sampler,
    subject_sampler=subject_sampler,
    shuffle_subjects=False,  # sampler controls order
)

for epoch in range(num_epochs):
    subject_sampler.set_epoch(epoch)
    for batch in loader:
        ...
```

Each rank processes only its assigned subjects. The epoch length
is `len(subject_sampler) * patches_per_volume`, not the full
dataset size.

## Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `max_length` | 300 | Buffer capacity. Larger = more diversity, more RAM |
| `patches_per_volume` | 10 | Max patches per subject (ceiling) |
| `num_workers` | 0 | Background loading threads |
| `shuffle_subjects` | True | Randomize subject order per epoch |
| `shuffle_patches` | True | Randomize patch order in buffer |
| `transform` | None | Applied to each subject before sampling |
| `subject_sampler` | None | For distributed training |
