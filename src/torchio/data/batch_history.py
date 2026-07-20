"""Exact per-element history support for batch containers."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from typing_extensions import Self

from .invertible import Invertible


class _BatchedHistoryMixin(Invertible):
    """Store one exact transform history per batch element."""

    _histories: list[list[Any]]

    @property
    def batch_size(self) -> int:
        """Number of batch elements."""
        raise NotImplementedError

    def unbatch(self) -> list[Any]:
        """Return individual batch elements."""
        raise NotImplementedError

    def _initialize_histories(
        self,
        batch_size: int,
        histories: Sequence[Sequence[Any]] | None = None,
    ) -> None:
        """Initialize exact per-element histories."""
        if histories is None:
            self._histories = [[] for _ in range(batch_size)]
            return
        self._set_histories(histories)

    @property
    def histories(self) -> tuple[tuple[Any, ...], ...]:
        """Immutable view of every element's exact history."""
        return tuple(tuple(history) for history in self._histories)

    def history(self, index: int) -> tuple[Any, ...]:
        """Return one element's exact history.

        Args:
            index: Batch element index.

        Returns:
            Immutable transform-history view for the element.
        """
        if not 0 <= index < self.batch_size:
            msg = (
                f"Cannot get history for element {index}:"
                f" batch size is {self.batch_size}"
            )
            raise IndexError(msg)
        return tuple(self._histories[index])

    @property
    def has_divergent_history(self) -> bool:
        """Whether element histories differ."""
        first = self._histories[0]
        return any(
            not _histories_equal(first, history) for history in self._histories[1:]
        )

    @property
    def applied_transforms(self) -> list[Any]:
        """Uniform batch history for compatibility.

        Raises:
            RuntimeError: If element histories differ.
        """
        if self.has_divergent_history:
            msg = (
                "This batch has divergent element histories. Use `histories`"
                " or `history(index)` instead of `applied_transforms`."
            )
            raise RuntimeError(msg)
        return list(self._histories[0])

    @applied_transforms.setter
    def applied_transforms(self, history: Sequence[Any]) -> None:
        copied = list(history)
        self._histories = [list(copied) for _ in range(self.batch_size)]

    def _set_histories(self, histories: Sequence[Sequence[Any]]) -> None:
        """Replace every element history."""
        if len(histories) != self.batch_size:
            msg = f"Expected {self.batch_size} histories, got {len(histories)}"
            raise ValueError(msg)
        self._histories = [list(history) for history in histories]

    def _append_history(self, traces: Sequence[Any | None]) -> None:
        """Append one optional trace per batch element."""
        if len(traces) != self.batch_size:
            msg = f"Expected {self.batch_size} traces, got {len(traces)}"
            raise ValueError(msg)
        for history, trace in zip(self._histories, traces, strict=True):
            if trace is not None:
                history.append(trace)

    def clear_history(self) -> None:
        """Remove every element history."""
        self._histories = [[] for _ in range(self.batch_size)]

    def get_inverse_transform(self, **kwargs: Any) -> Any:
        """Build a vectorized inverse for a uniform batch history.

        Args:
            **kwargs: Forwarded to `Invertible.get_inverse_transform`.

        Raises:
            RuntimeError: If element histories differ.
        """
        if self.has_divergent_history:
            msg = (
                "This batch has divergent element histories, so one vectorized"
                " inverse is ambiguous. Use `apply_inverse_transform()`."
            )
            raise RuntimeError(msg)
        return super().get_inverse_transform(**kwargs)

    def apply_inverse_transform(self, **kwargs: Any) -> Self:
        """Apply vectorized or per-element inverse transforms.

        Args:
            **kwargs: Forwarded to `get_inverse_transform`.

        Returns:
            A batch with transforms undone.
        """
        if not self.has_divergent_history:
            return super().apply_inverse_transform(**kwargs)
        inverted = [item.apply_inverse_transform(**kwargs) for item in self.unbatch()]
        result = self._batch_items(inverted)
        result.clear_history()
        return result

    def _batch_items(self, items: Sequence[Any]) -> Self:
        """Rebuild the concrete batch type from unbatched items."""
        raise NotImplementedError


def _histories_equal(first: Sequence[Any], second: Sequence[Any]) -> bool:
    """Compare histories, treating ambiguous value equality as unequal."""
    try:
        return bool(first == second)
    except (RuntimeError, TypeError, ValueError):
        return False
