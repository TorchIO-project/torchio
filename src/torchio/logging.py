"""Logging configuration for TorchIO.

Logging is **disabled by default**.  Call :func:`enable_logging` to opt in.

Example::

    import torchio as tio
    tio.enable_logging("DEBUG")  # rich-formatted debug output
"""

from __future__ import annotations

import sys

from loguru import logger
from rich.logging import RichHandler

# Libraries must not emit logs unless the user opts in
logger.disable("torchio")


def enable_logging(level: str = "INFO", *, rich: bool = True) -> None:
    """Enable TorchIO logging.

    Args:
        level: Minimum log level (``DEBUG``, ``INFO``, ``WARNING``,
            ``ERROR``).
        rich: If ``True`` (default), use :class:`rich.logging.RichHandler`
            for colourful, markup-enabled output with pretty tracebacks.
            Set to ``False`` for plain stderr output.
    """
    logger.enable("torchio")
    logger.remove()
    if rich:
        logger.add(
            RichHandler(markup=True, rich_tracebacks=True),
            format="{message}",
            level=level,
        )
    else:
        logger.add(sys.stderr, level=level)
