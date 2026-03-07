from __future__ import annotations

from collections.abc import Callable
from collections.abc import Sequence
from pathlib import Path
from typing import TypeAlias

import numpy as np
import torch

# For typing hints
TypePath: TypeAlias = str | Path
TypeNumber: TypeAlias = int | float
TypeKeys: TypeAlias = Sequence[str] | None
TypeData: TypeAlias = torch.Tensor | np.ndarray
TypeDataAffine: TypeAlias = tuple[torch.Tensor, np.ndarray]
TypeSlice: TypeAlias = int | slice

TypeDoubletInt: TypeAlias = tuple[int, int]
TypeTripletInt: TypeAlias = tuple[int, int, int]
TypeQuartetInt: TypeAlias = tuple[int, int, int, int]
TypeSextetInt: TypeAlias = tuple[int, int, int, int, int, int]

TypeDoubleFloat: TypeAlias = tuple[float, float]
TypeTripletFloat: TypeAlias = tuple[float, float, float]
TypeQuartetFloat: TypeAlias = tuple[float, float, float, float]
TypeSextetFloat: TypeAlias = tuple[float, float, float, float, float, float]

TypeTuple: TypeAlias = int | TypeTripletInt
TypeRangeInt: TypeAlias = int | TypeDoubletInt
TypeSpacing: TypeAlias = float | TypeTripletFloat
TypeSpatialShape: TypeAlias = int | TypeTripletInt
TypeRangeFloat: TypeAlias = float | TypeDoubleFloat
TypeCallable: TypeAlias = Callable[[torch.Tensor], torch.Tensor]
TypeDirection2D: TypeAlias = TypeQuartetFloat
TypeDirection3D: TypeAlias = tuple[
    float,
    float,
    float,
    float,
    float,
    float,
    float,
    float,
    float,
]
TypeDirection: TypeAlias = TypeDirection2D | TypeDirection3D
