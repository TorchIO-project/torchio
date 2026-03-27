"""Axis validation and conversion utilities.

Axis strings are 3-character uppercase strings describing the ordering and
orientation of coordinate axes. Two families are supported:

- **Voxel**: permutations of ``"IJK"`` (6 options).
- **Anatomical**: one letter from each pair {R, L}, {A, P}, {S, I}
  in any order (48 options).

The string ``"IJK"`` is always interpreted as voxel (not
Inferior-J-K), because J and K have no anatomical meaning.
"""

from __future__ import annotations

from enum import Enum

_VOXEL_LETTERS = frozenset("IJK")

# Each tuple is a pair of opposite anatomical directions.
ANATOMICAL_PAIRS: tuple[tuple[str, str], ...] = (
    ("R", "L"),
    ("A", "P"),
    ("S", "I"),
)

# Flat set of all anatomical letters.
_ANATOMICAL_LETTERS = frozenset(letter for pair in ANATOMICAL_PAIRS for letter in pair)

# Map each anatomical letter to its opposite.
_OPPOSITE: dict[str, str] = {}
for _a, _b in ANATOMICAL_PAIRS:
    _OPPOSITE[_a] = _b
    _OPPOSITE[_b] = _a

# Map each anatomical letter to its canonical pair (sorted tuple).
_LETTER_TO_PAIR: dict[str, tuple[str, str]] = {}
for _pair in ANATOMICAL_PAIRS:
    for _letter in _pair:
        _LETTER_TO_PAIR[_letter] = _pair


class AxesType(Enum):
    """Whether an axis string describes voxel or anatomical coordinates."""

    VOXEL = "voxel"
    ANATOMICAL = "anatomical"


def validate_axes(axes: str) -> str:
    """Validate a 3-character axis string.

    Args:
        axes: Axis string to validate.

    Returns:
        The validated string (unchanged).

    Raises:
        ValueError: If the string is not a valid axis specification.
    """
    if len(axes) != 3:
        msg = f"Axis string must be 3 characters, got {len(axes)}: {axes!r}"
        raise ValueError(msg)
    if _is_voxel(axes) or _is_anatomical(axes):
        return axes
    msg = (
        f"Invalid axis string {axes!r}. Must be a permutation of 'IJK'"
        " (voxel) or one letter from each anatomical pair"
        " {R,L}, {A,P}, {S,I}."
    )
    raise ValueError(msg)


def axes_type(axes: str) -> AxesType:
    """Return whether *axes* is a voxel or anatomical axis string.

    The string must already be valid (call :func:`validate_axes` first).
    """
    if _is_voxel(axes):
        return AxesType.VOXEL
    return AxesType.ANATOMICAL


def get_axis_mapping(
    src: str,
    tgt: str,
) -> tuple[tuple[int, int, int], tuple[bool, bool, bool]]:
    """Compute the column permutation and flips to go from *src* to *tgt*.

    Both strings must be the same type (both voxel or both anatomical).

    Args:
        src: Source axis string.
        tgt: Target axis string.

    Returns:
        A tuple ``(permutation, flips)`` where *permutation* gives the
        source column index for each target column and *flips* indicates
        whether each target column should be negated.

    Raises:
        ValueError: If the two strings are not the same type.
    """
    src_type = axes_type(src)
    tgt_type = axes_type(tgt)
    if src_type != tgt_type:
        msg = (
            f"Cannot compute axis mapping between different types:"
            f" {src!r} ({src_type.value}) and {tgt!r} ({tgt_type.value})."
            " Use the affine to convert between voxel and anatomical."
            " Both must be the same type."
        )
        raise ValueError(msg)

    if src_type == AxesType.VOXEL:
        return _voxel_mapping(src, tgt)
    return _anatomical_mapping(src, tgt)


# --- Private helpers ---


def _is_voxel(axes: str) -> bool:
    return set(axes) == _VOXEL_LETTERS and len(axes) == 3


def _is_anatomical(axes: str) -> bool:
    if len(axes) != 3:
        return False
    if not all(c in _ANATOMICAL_LETTERS for c in axes):
        return False
    # Each pair must be represented exactly once.
    seen_pairs: set[tuple[str, str]] = set()
    for c in axes:
        pair = _LETTER_TO_PAIR[c]
        if pair in seen_pairs:
            return False
        seen_pairs.add(pair)
    return len(seen_pairs) == 3


def _voxel_mapping(
    src: str,
    tgt: str,
) -> tuple[tuple[int, int, int], tuple[bool, bool, bool]]:
    perm = tuple(src.index(c) for c in tgt)
    flips = (False, False, False)
    return perm, flips  # type: ignore[return-value]


def _anatomical_mapping(
    src: str,
    tgt: str,
) -> tuple[tuple[int, int, int], tuple[bool, bool, bool]]:
    perm: list[int] = []
    flips: list[bool] = []
    for tgt_letter in tgt:
        tgt_pair = _LETTER_TO_PAIR[tgt_letter]
        # Find which source column belongs to the same pair.
        for src_idx, src_letter in enumerate(src):
            if _LETTER_TO_PAIR[src_letter] == tgt_pair:
                perm.append(src_idx)
                flips.append(src_letter != tgt_letter)
                break
    return tuple(perm), tuple(flips)  # type: ignore[return-value]
