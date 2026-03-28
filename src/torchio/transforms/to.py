"""To transform: move data to a device and/or cast dtype."""

from __future__ import annotations

from typing import Any

import attrs

from ..data.subject import Subject
from .transform import Transform


@attrs.define(slots=False, eq=False, kw_only=True, repr=False)
class To(Transform):
    """Move all data to a device and/or cast to a dtype.

    Wraps ``Subject.to()`` as a transform so it can be used inside
    ``Compose`` pipelines.

    Args:
        *to_args: Positional arguments forwarded to ``torch.Tensor.to()``.
            Typically a device string (``"cpu"``, ``"cuda"``) or a
            ``torch.dtype`` (``torch.float16``).
        **to_kwargs: Keyword arguments forwarded to ``torch.Tensor.to()``.

    Examples:
        >>> transform = tio.To(torch.float16)
        >>> transform = tio.To("cuda")
        >>> transform = tio.To("cuda", dtype=torch.float16)
    """

    to_args: tuple[Any, ...] = attrs.field(factory=tuple, alias="to_args")
    to_kwargs: dict[str, Any] = attrs.field(factory=dict, alias="to_kwargs")

    def __init__(self, *to_args: Any, **to_kwargs: Any) -> None:
        self.__attrs_init__(to_args=to_args, to_kwargs=to_kwargs)

    def make_params(self, subject: Subject) -> dict[str, Any]:
        return {"to_args": self.to_args, "to_kwargs": self.to_kwargs}

    def apply_transform(self, subject: Subject, params: dict[str, Any]) -> Subject:
        subject.to(*params["to_args"], **params["to_kwargs"])
        return subject
