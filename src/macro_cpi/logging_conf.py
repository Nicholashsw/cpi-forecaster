"""Project-wide logging configuration."""
from __future__ import annotations

import logging
import sys


def get_logger(name: str) -> logging.Logger:
    """Return a configured logger that writes structured lines to stderr once."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stderr)
        handler.setFormatter(
            logging.Formatter("%(asctime)s | %(levelname)-7s | %(name)s | %(message)s")
        )
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
        logger.propagate = False
    return logger
