"""Type aliases for TorchIO."""

from __future__ import annotations

import os
from typing import TypeAlias

import numpy as np
from jaxtyping import Float
from torch import Tensor

# Path type for user-facing APIs
TypePath: TypeAlias = str | os.PathLike[str]

# Jaxtyping tensor types (document shape in annotations)
TypeImageData = Float[Tensor, "channels size_i size_j size_k"]
TypeAffineMatrix = Float[np.ndarray, "4 4"]
TypeDirection = Float[np.ndarray, "3 3"]
TypeWorldPoints = Float[np.ndarray, "num_points 3"]

# Tuple types
TypeSpatialShape: TypeAlias = tuple[int, int, int]
TypeTensorShape: TypeAlias = tuple[int, int, int, int]
TypeSpacing: TypeAlias = tuple[float, float, float]
TypeOrigin: TypeAlias = tuple[float, float, float]
TypeOrientationCodes: TypeAlias = tuple[str, str, str]
TypeThreeInts: TypeAlias = tuple[int, int, int]
TypeSixInts: TypeAlias = tuple[int, int, int, int, int, int]

__all__ = [
    "TypeAffineMatrix",
    "TypeDirection",
    "TypeImageData",
    "TypeOrientationCodes",
    "TypeOrigin",
    "TypePath",
    "TypeSpacing",
    "TypeSpatialShape",
    "TypeTensorShape",
    "TypeWorldPoints",
]
