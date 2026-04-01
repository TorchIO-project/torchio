"""To transform: move data to a device and/or cast dtype."""

from __future__ import annotations

from typing import Any

from ..data.batch import SubjectsBatch
from .transform import Transform


class To(Transform):
    """Move all data to a device and/or cast to a dtype.

    Wraps the ``to()`` method as a transform so it can be used inside
    [`Compose`][torchio.Compose] pipelines.

    Args:
        *to_args: Positional arguments forwarded to
            [`torch.Tensor.to()`](https://pytorch.org/docs/stable/generated/torch.Tensor.to.html).
            Typically a device string (``"cpu"``, ``"cuda"``,
            ``"mps"``) or a ``torch.dtype`` (``torch.float16``).
        **to_kwargs: Keyword arguments forwarded to
            ``torch.Tensor.to()``.

    Examples:
        >>> import torchio as tio
        >>> transform = tio.To(torch.float16)
        >>> transform = tio.To("cuda")
        >>> pipeline = tio.Compose([
        ...     tio.To("cuda"),
        ...     tio.Noise(std=0.1),
        ... ])
    """

    def __init__(self, *to_args: Any, **to_kwargs: Any) -> None:
        super().__init__()
        self.to_args = to_args
        self.to_kwargs = to_kwargs

    def make_params(self, batch: SubjectsBatch) -> dict[str, Any]:
        return {"to_args": self.to_args, "to_kwargs": self.to_kwargs}

    def apply_transform(
        self,
        batch: SubjectsBatch,
        params: dict[str, Any],
    ) -> SubjectsBatch:
        batch.to(*params["to_args"], **params["to_kwargs"])
        return batch
