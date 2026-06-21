"""Thin HTTP wrapper with exponential backoff. Raises on exhaustion; never fails silently."""
from __future__ import annotations

import time
from typing import Any

import requests

from macro_cpi import config
from macro_cpi.logging_conf import get_logger

logger = get_logger(__name__)


class FetchError(RuntimeError):
    """Raised when an HTTP fetch fails after all retries."""


def get_json(url: str, params: dict[str, Any] | None = None) -> dict:
    """GET a URL with retries and exponential backoff; return parsed JSON or raise FetchError."""
    last_exc: Exception | None = None
    for attempt in range(1, config.HTTP_MAX_RETRIES + 1):
        try:
            resp = requests.get(url, params=params, timeout=config.HTTP_TIMEOUT_SECS)
            resp.raise_for_status()
            return resp.json()
        except (requests.RequestException, ValueError) as exc:
            last_exc = exc
            wait = config.HTTP_BACKOFF_BASE_SECS * (2 ** (attempt - 1))
            logger.warning(
                "fetch failed (attempt %d/%d) url=%s err=%s; retrying in %.1fs",
                attempt,
                config.HTTP_MAX_RETRIES,
                url,
                exc,
                wait,
            )
            if attempt < config.HTTP_MAX_RETRIES:
                time.sleep(wait)
    raise FetchError(f"GET {url} failed after {config.HTTP_MAX_RETRIES} attempts") from last_exc
