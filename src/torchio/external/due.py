"""Stub for a guaranteed safe import of duecredit constructs.

If duecredit is not available, a no-op collector is used instead.

Origin:     Originally a part of the duecredit
Copyright:  2015-2019  DueCredit developers
License:    BSD-2

See https://github.com/duecredit/duecredit/blob/master/README.md for examples.
"""

from __future__ import annotations

import importlib
import logging
from typing import Any


class InactiveDueCreditCollector:
    """Stub Collector that does nothing."""

    def _donothing(self, *args: Any, **kwargs: Any) -> None:
        pass

    def dcite(self, *args: Any, **kwargs: Any) -> Any:
        def nondecorating_decorator(func: Any) -> Any:
            return func

        return nondecorating_decorator

    active = False
    activate = add = cite = dump = load = _donothing

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"


def _donothing_func(*args: Any, **kwargs: Any) -> None:
    pass


try:
    _duecredit = importlib.import_module("duecredit")
    BibTeX = _duecredit.BibTeX
    Doi = _duecredit.Doi
    Text = _duecredit.Text
    Url = _duecredit.Url
    due = _duecredit.due

    if "due" in locals() and not hasattr(due, "cite"):
        msg = "Imported due lacks .cite. DueCredit is now disabled"
        raise RuntimeError(msg)
except Exception as _exc:
    if not isinstance(_exc, ImportError):
        logging.getLogger("duecredit").error(
            "Failed to import duecredit due to %s",
            _exc,
        )
    due = InactiveDueCreditCollector()
    BibTeX = Doi = Url = Text = _donothing_func
