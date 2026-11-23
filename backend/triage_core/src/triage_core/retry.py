"""Retries with exponential backoff and jitter, without third-party dependencies."""

from __future__ import annotations

import random
import time
from collections.abc import Callable


class RetryableError(RuntimeError):
    """A transient failure (throttling, 5xx, timeout) worth retrying."""


def call_with_retries[T](
    fn: Callable[[], T],
    *,
    is_retryable: Callable[[Exception], bool],
    attempts: int = 5,
    base_delay: float = 1.0,
    sleep: Callable[[float], None] = time.sleep,
    jitter: Callable[[], float] = random.random,  # noqa: S311 - backoff jitter, not crypto
) -> T:
    for attempt in range(attempts):
        try:
            return fn()
        except Exception as err:
            if attempt == attempts - 1 or not is_retryable(err):
                raise
            sleep(base_delay * 2**attempt + base_delay * jitter())
    raise AssertionError("unreachable")
