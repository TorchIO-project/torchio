# TorchIO batching redesign

## Executive summary

The current batching system is too complicated, but the core idea is sound.
TorchIO should remain a **struct of arrays**:

- image fields are stacked into 5D tensors `(B, C, I, J, K)`;
- metadata and ragged annotations remain Python lists;
- transforms operate once on the batch and return the same input type.

The accidental complexity comes from representing the same concepts in multiple
ways:

- `ImagesBatch` may or may not have image templates;
- batch history may be shared or per-element;
- per-element history is reconstructed later from private keys embedded in
  transform parameters;
- `OneOf`, `SomeOf`, adapters, inversion, and unbatching each need special
  history handling;
- raw constructors expose internal assembly details such as
  `image_templates`.

The redesign keeps the useful features but enforces one representation per
concept:

1. Every image batch always has private per-element prototypes.
2. Every batch element always has its own exact transform history.
3. Public batches are created through factories, not internal constructors.
4. Images remain stacked; metadata, points, and boxes remain explicit typed
   stores.

This avoids replacing simple dictionaries with a generic proxy framework. The
explicit stores are not the main problem; the optional state and history
duality are.

## Features that remain

| Feature | Redesign |
| --- | --- |
| Vectorized GPU transforms | Image fields remain 5D `ImagesBatch` tensors. |
| Exact `Subject -> batch -> Subject` round-trip | Private image prototypes and explicit object stores preserve payload. |
| Custom `Image` subclasses | Prototypes rebuild images through `Image.new_like()`. |
| Image metadata and image-level annotations | Stored in each private image prototype. |
| Subject metadata, points, and boxes | Remain explicit dictionaries of per-element lists. |
| Metadata-only and annotation-only batches | Batch size comes from stored elements, not the first image. |
| Mutable `batch.metadata` | Remains a real dictionary of live lists; no proxy mapping is introduced. |
| `map_subjects()` | Unbatches exact rows, invokes the callback, and re-batches through the same schema path. |
| Per-instance transforms and probability | Parameters are still sampled per element where supported. |
| `OneOf` / `SomeOf` per-element branches | Exact per-element histories make branching ordinary rather than special. |
| Inversion | Uniform histories use the vectorized batch inverse; divergent histories use per-element inversion. |
| Exact replay | `apply_with_params()` remains, but internal batch bookkeeping is not persisted in history. |

## New public model

### Construction

The lossless factories become the primary public API:

```python
images = tio.ImagesBatch.from_images(image_list)
images = tio.ImagesBatch.from_tensor(
    data,
    affines,
    image_class=tio.ScalarImage,
)

subjects = tio.SubjectsBatch.from_subjects(subject_list)
```

Raw constructors that expose internal assembly details are removed from the
public API. In particular:

- `ImagesBatch(..., image_templates=...)` is removed;
- `SubjectsBatch(images, points=..., bounding_boxes=..., metadata=...)` becomes
  a private `_from_parts()` constructor for TorchIO internals.

### Image batches

`ImagesBatch` has one invariant:

```text
data + affines + one private prototype per element
```

Prototypes are always present. `from_tensor()` synthesizes minimal prototypes;
`from_images()` derives them from the input images. This removes:

- the `templates is None` branches;
- the duplicated `_image_class` state;
- the public `image_templates` argument.

`image_class` and `is_label` become public derived properties.

### Subject batches

`SubjectsBatch` keeps the explicit stores:

```text
images: dict[str, ImagesBatch]
points: dict[str, list[Points]]
bounding_boxes: dict[str, list[BoundingBoxes]]
metadata: dict[str, list[Any]]
```

These dictionaries already provide the simplest correct mutation semantics.
A generic `EntryStore` plus filtered mutable proxies would add indirection and
new aliasing rules without removing meaningful complexity.

Construction is driven by one private `SubjectSchema`:

```text
SubjectSchema
  ordered image specifications
  metadata keys
  point specifications
  bounding-box specifications
```

The schema is derived from the first subject, validates every later subject, and
drives stacking and unbatching. This replaces the collection of loosely coupled
validation helpers with one explicit invariant.

## Simpler history model

### Current model

Today a batch has:

```text
applied_transforms
optional _per_element_history
```

Per-instance parameters are stored once with private fields:

```text
_batch_size
_batched_keys
_keep
```

Unbatching later interprets those fields to reconstruct each element's history.
This makes history handling leak into transforms, inversion, composition,
adapters, and replay validation.

### New model

Every batch stores exact histories directly:

```python
batch.histories: list[list[AppliedTransform]]
batch.history(index)
```

When a transform finishes:

1. Its transient parameter dictionary is split into one clean parameter
   dictionary per element.
2. Gated-out elements receive no trace.
3. Each element history receives its exact `AppliedTransform`.

Private batching keys may still exist transiently while built-in transforms are
migrated, but they are never persisted in public history.

Every element receives an independent trace record, including uniform
applications. This avoids cross-element aliasing through mutable parameter
dictionaries and keeps the exact-history model simple.

This removes:

- `_per_element_history`;
- `set_per_element_history()`;
- `adopt_history()`;
- read-time `_slice_history()`;
- special `OneOf` / `SomeOf` history freezing;
- batch-level history parameters containing `_batch_size`, `_batched_keys`, or
  `_keep`.

### Inversion

Vectorization is retained:

- if every element history is equal, build one inverse `Compose` and apply it to
  the whole batch;
- if histories differ, invert each element and re-batch when schemas and shapes
  remain compatible.

`get_inverse_transform()` is available only for uniform histories.
`apply_inverse_transform()` handles both cases.

## What is deliberately not introduced

### No list-of-subjects with lazy tensor stacking

That model looks elegant, but in-place tensor mutation makes cache invalidation
and scatter-back semantics difficult. It moves complexity into invisible proxy
behavior and risks losing vectorized performance.

### No generic mutable column proxy framework

Metadata must support operations such as:

```python
batch.metadata["age"][0] = 42
batch.metadata["site"] = ["A", "B"]
```

Plain dictionaries of live lists already implement this correctly. A filtered
`MutableMapping` proxy would need custom assignment, deletion, ordering,
namespace, and aliasing rules.

### No always-per-element inversion

Uniform histories are common and can be inverted efficiently on a 5D batch.
The redesign keeps that fast path.

## Expected simplification

The redesign removes or consolidates:

- duplicated history methods across `ImagesBatch` and `SubjectsBatch`;
- optional-template branches;
- public internal-construction arguments;
- shared/per-element history duality;
- read-time history slicing;
- special composition/adaptor history handling;
- private batch metadata persisted in `AppliedTransform.params`.

The target is a 25–35% reduction in the batching/history implementation, but
the more important improvement is conceptual:

```text
one image representation
one schema representation
one history representation
```

## Replacement PR stack

The current open PRs #1494–#1496 should be superseded rather than force-rewritten
again. Keep them open until replacement PRs exist, then close them with links to
the new stack.

0. [#1500 Fix padding mode type narrowing](https://github.com/TorchIO-project/torchio/pull/1500)
   - independent prerequisite restoring a clean `prek` baseline.
1. [#1501 Redesign batch construction and schema](https://github.com/TorchIO-project/torchio/pull/1501)
   - add `ImagesBatch.from_tensor()`;
   - make image prototypes private and mandatory;
   - add public `image_class` / `is_label`;
   - make `SubjectsBatch.from_subjects()` the public construction path;
   - centralize schema validation.
2. [#1502 Store exact history per batch element](https://github.com/TorchIO-project/torchio/pull/1502)
   - add the new history container/API;
   - split transform params eagerly when recording;
   - retain vectorized inversion for uniform histories;
   - remove shared/per-element history duality and persisted private keys.
3. [#1503 Integrate subject mapping and exact replay](https://github.com/TorchIO-project/torchio/pull/1503)
   - simplify wrapping/unwrapping;
   - remove special `OneOf` / `SomeOf` history handling;
   - simplify `map_subjects()`;
   - migrate MONAI and Cornucopia adapters;
   - keep exact replay with transient validation only.
4. **Documentation, migration, and performance validation**
   - update the data-model, transform, migration, loader, and custom-transform
     documentation;
   - document all public arguments;
   - add CPU/GPU batch and unbatch benchmarks;
   - close #1494–#1496 as superseded with links.

## Success criteria

- All existing user-visible features in the table above remain.
- No public history contains private batch bookkeeping fields.
- `ImagesBatch` has no optional prototype mode.
- Batch constructors no longer expose internal payload arguments.
- Uniform inverse transforms remain vectorized.
- Metadata mutation remains direct and unsurprising.
- Full tests, type checking, Ruff, docs tests/build, `prek`, and Xenon pass.
- Benchmarks show no material regression in vectorized transform throughput.
