# API reference

The API reference centralizes the public TorchIO surface rendered with
[`mkdocstrings`](https://mkdocstrings.github.io/).
The rest of the documentation remains focused on concepts, workflows, and
examples.

Most classes and functions documented here can also be imported from the
top-level `torchio` namespace, but the reference pages are grouped by module
family to keep the navigation manageable.

## Guide-to-reference map

| Area | Guide | API reference |
| --- | --- | --- |
| Data structures and loaders | [Data structures](../data/index.md) | [Data API](data.md) |
| Patch sampling and inference | [Patch-based pipelines](../patches/index.md) | [Data API](data.md) |
| Transforms | [Transforms guide](../transforms/index.md) | [Transforms API](transforms/index.md) |
| Datasets | [Medical image datasets](../datasets.md) | [Datasets API](datasets.md) |
| Command-line tools | [Command-line tools](../interfaces/cli.md) | [CLI API](cli.md) |
| Utilities and helpers | N/A | [Utilities API](utilities.md) |

## Sections

- [Data API](data.md) for images, subjects, datasets, samplers, inference, and I/O
- [Transforms API](transforms/index.md) for base classes, augmentation, and preprocessing
- [Datasets API](datasets.md) for the built-in public datasets
- [Utilities API](utilities.md) for helpers, visualization, constants, and types
- [CLI API](cli.md) for the Typer-based command modules behind `tiotr` and `tiohd`
