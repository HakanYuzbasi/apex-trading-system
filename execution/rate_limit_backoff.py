"""
execution/rate_limit_backoff.py — HTTP 429 exponential backoff (Round 7 / GAP-10D)
==================================================================================

Every HTTP broker client eventually hits a 429 ("Too Many Requests") response.
The default naive ``2 ** attempt`` backoff quickly becomes correlated across
retries in a live fleet — every client retries at the same boundary and piles
into the next window together. This module centralises the backoff so every
connector gets the same jittered, capped exponential schedule and surfaces a
process-wide "submission halted" flag when the retry budget is exhausted.

Design notes
------------

- Delays are clamped to ``API_RATE_LIMIT_MAX_DELAY_SECONDS`` so a deep retry
  never sleeps for minutes and then blows past the caller's own timeout.
- Jitter is uniform in ``[0, 0.5]`` seconds, matching the audit spec exactly.
- The halt flag is opt-in — callers probe it before submitting an order and
  may clear it via :func:`reset_submission_halt` once the upstream API
  recovers. This sidesteps process-wide deadlocks.
"""

from __future__ import annotations

import logging
import random
import threading
from typing import Optional

from config import ApexConfig

logger = logging.getLogger(__name__)


_HALT_LOCK = threading.Lock()
_HALT_ACTIVE: bool = False
_HALT_REASON: Optional[str] = None


def compute_backoff_seconds(
    attempt: int,
    *,
    base_delay: Optional[float] = None,
    max_delay: Optional[float] = None,
) -> float:
    """
    Compute the sleep duration (seconds) for retry ``attempt``.

    Args:
        attempt: Zero-based retry index (0 for the first backoff, 1 for the
            second, …). Must be ≥ 0.
        base_delay: Override for ``ApexConfig.API_RATE_LIMIT_BASE_DELAY_SECONDS``.
        max_delay: Override for ``ApexConfig.API_RATE_LIMIT_MAX_DELAY_SECONDS``.

    Returns:
        The sleep duration in seconds, clamped to ``max_delay`` and with a
        uniform ``[0, 0.5]`` jitter applied.

    Raises:
        ValueError: If ``attempt`` is negative.
    """
    if attempt < 0:
        raise ValueError(f"attempt must be >= 0, got {attempt}")

    base = float(
        base_delay
        if base_delay is not None
        else getattr(ApexConfig, "API_RATE_LIMIT_BASE_DELAY_SECONDS", 1.0)
    )
    ceiling = float(
        max_delay
        if max_delay is not None
        else getattr(ApexConfig, "API_RATE_LIMIT_MAX_DELAY_SECONDS", 30.0)
    )
    if base <= 0.0:
        raise ValueError(f"base_delay must be > 0, got {base}")
    if ceiling < base:
        raise ValueError(
            f"max_delay ({ceiling}) must be >= base_delay ({base})"
        )

    raw = base * (2.0 ** attempt) + random.uniform(0.0, 0.5)
    return min(raw, ceiling)


def max_retries() -> int:
    """Return the configured retry budget (``API_RATE_LIMIT_MAX_RETRIES``)."""
    return int(getattr(ApexConfig, "API_RATE_LIMIT_MAX_RETRIES", 5))


def submission_halted() -> bool:
    """Return ``True`` while the process-wide order-submission halt is active."""
    with _HALT_LOCK:
        return _HALT_ACTIVE


def halt_submission(reason: str) -> None:
    """
    Raise the order-submission halt flag and log CRITICAL.

    The spec for GAP-10D requires that we halt *new order submission* when the
    retry budget is exhausted but that we *do not* crash the process — the
    monitoring loops and existing position management must remain running.

    Args:
        reason: Operator-facing explanation for the halt (source connector,
            request path, last status code, …).
    """
    global _HALT_ACTIVE, _HALT_REASON
    with _HALT_LOCK:
        if not _HALT_ACTIVE:
            logger.critical(
                "🛑 Order submission HALTED (rate-limit/backoff exhausted): %s",
                reason,
            )
        _HALT_ACTIVE = True
        _HALT_REASON = reason


def reset_submission_halt(reason: str = "manual_reset") -> None:
    """
    Clear the halt flag. Call this once the upstream API has recovered —
    typically after a successful probe request from the caller.

    Args:
        reason: Operator-facing explanation logged alongside the reset.
    """
    global _HALT_ACTIVE, _HALT_REASON
    with _HALT_LOCK:
        was_active = _HALT_ACTIVE
        _HALT_ACTIVE = False
        _HALT_REASON = None
    if was_active:
        logger.info("✅ Order submission halt cleared (%s)", reason)


def current_halt_reason() -> Optional[str]:
    """Return the reason string set by the most recent :func:`halt_submission`."""
    with _HALT_LOCK:
        return _HALT_REASON
