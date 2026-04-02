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
TypeThreeInts: TypeAlias = tuple[int, int, int]
TypeFourInts: TypeAlias = tuple[int, int, int, int]
TypeSixInts: TypeAlias = tuple[int, int, int, int, int, int]
TypeThreeFloats: TypeAlias = tuple[float, float, float]
TypeSpatialShape = TypeThreeInts
TypeTensorShape = TypeFourInts
TypeSpacing = TypeThreeFloats
TypeOrigin = TypeThreeFloats
TypeOrientationCodes: TypeAlias = tuple[str, str, str]
