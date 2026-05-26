"""Small utility helpers used across the project."""

from __future__ import annotations

import logging
import sys
import time
from contextlib import contextmanager
from typing import Iterator


def get_logger(name: str) -> logging.Logger:
    """Return a console logger that doesn't double-attach handlers."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(
            logging.Formatter(
                "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
                datefmt="%H:%M:%S",
            )
        )
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
        logger.propagate = False
    return logger


@contextmanager
def timer(label: str) -> Iterator[None]:
    """Log how long a block takes."""
    log = get_logger("timer")
    start = time.perf_counter()
    log.info("▶ %s …", label)
    try:
        yield
    finally:
        elapsed = time.perf_counter() - start
        log.info("✔ %s done in %.2fs", label, elapsed)
