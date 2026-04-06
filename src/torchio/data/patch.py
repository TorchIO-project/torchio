"""Patch location metadata for patch-based pipelines."""

from __future__ import annotations

from dataclasses import dataclass

from ..types import TypeThreeInts


@dataclass(frozen=True)
class PatchLocation:
    """Spatial location of an extracted patch within a volume.

    Attributes:
        index: ``(i, j, k)`` voxel indices of the patch corner
            (the corner closest to the origin).
        size: ``(si, sj, sk)`` spatial shape of the patch.
        subject_index: Optional identifier for multi-subject batches.
    """

    index: TypeThreeInts
    size: TypeThreeInts
    subject_index: int | None = None

    @property
    def index_ini(self) -> TypeThreeInts:
        """Starting voxel indices ``(i, j, k)``."""
        return self.index

    @property
    def index_fin(self) -> TypeThreeInts:
        """One-past-the-end voxel indices."""
        return (
            self.index[0] + self.size[0],
            self.index[1] + self.size[1],
            self.index[2] + self.size[2],
        )

    def to_slices(self) -> tuple[slice, slice, slice]:
        """Convert to spatial slices for tensor indexing."""
        ini = self.index_ini
        fin = self.index_fin
        return (
            slice(ini[0], fin[0]),
            slice(ini[1], fin[1]),
            slice(ini[2], fin[2]),
        )

    def scaled(self, factor: tuple[float, float, float]) -> PatchLocation:
        """Return a new location with indices and size scaled by factor."""
        return PatchLocation(
            index=(
                round(self.index[0] * factor[0]),
                round(self.index[1] * factor[1]),
                round(self.index[2] * factor[2]),
            ),
            size=(
                round(self.size[0] * factor[0]),
                round(self.size[1] * factor[1]),
                round(self.size[2] * factor[2]),
            ),
            subject_index=self.subject_index,
        )
