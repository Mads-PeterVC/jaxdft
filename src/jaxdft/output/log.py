"""The ``jaxdft`` logger. The library only emits records; the application decides
where they go. ``jaxdft/__init__.py`` attaches a ``NullHandler``, so nothing is
printed unless logging is configured, e.g. with ``log_to_file``."""

from __future__ import annotations

import logging
import sys
from pathlib import Path

LOGGER_NAME = "jaxdft"

_FORMAT = "%(message)s"


def _attach(handler: logging.Handler, level) -> logging.Handler:
    handler.setFormatter(logging.Formatter(_FORMAT))
    handler.setLevel(level)
    logger = logging.getLogger(LOGGER_NAME)
    logger.addHandler(handler)
    if logger.level == logging.NOTSET or logger.level > handler.level:
        logger.setLevel(handler.level)
    return handler


def log_to_file(path, level="INFO", mode: str = "w") -> logging.FileHandler:
    """Send ``jaxdft`` log records at ``level`` and above to the text file ``path``
    (overwritten by default). Returns the handler; pass it to ``logging.getLogger(
    "jaxdft").removeHandler`` (and close it) to stop."""
    return _attach(logging.FileHandler(Path(path), mode=mode, encoding="utf-8"), level)


def log_to_stream(stream=None, level="INFO") -> logging.StreamHandler:
    """Send ``jaxdft`` log records to ``stream`` (default: standard output)."""
    return _attach(logging.StreamHandler(sys.stdout if stream is None else stream), level)
