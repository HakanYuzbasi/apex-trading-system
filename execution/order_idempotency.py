"""
execution/order_idempotency.py — Duplicate-order suppression guard
==================================================================

Broker connectors reconnect under load; when they do, pending-order retries can
re-send the same (symbol, side, qty) pair twice and produce double-fills. This
module exposes :class:`OrderIdempotencyGuard`, a per-process cache keyed by the
tuple of ``(symbol, side, quantized_qty)`` with a configurable time-to-live.

Flow:
    1. Before submission, call :meth:`register` with the tentative order. If
       the key is already live (within TTL), ``register`` returns an existing
       key and the caller **must not** submit. On a fresh key a new token is
       returned and the submission proceeds.
    2. When the broker accepts the order, call :meth:`mark_submitted` with the
       broker-assigned order id so we can reconcile fills. If the broker
       rejects / errors, call :meth:`forget` to release the slot immediately.

All thresholds are read from :class:`ApexConfig`:

- ``ORDER_DEDUP_TTL_SECONDS`` — dedup window (seconds).
- ``ORDER_DEDUP_MAX_ENTRIES`` — memory cap; oldest entries evicted first.

A TTL of zero disables the guard completely; callers always receive a fresh
key and are never blocked.
"""
from __future__ import annotations

import hashlib
import logging
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass
from typing import Optional, Tuple

from config import ApexConfig

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class DedupKey:
    """Canonical order-identity tuple used as cache key."""

    symbol: str
    side: str
    qty_bucket: float

    def hexdigest(self) -> str:
        """Stable hex digest suitable for use as a broker client-order id."""
        payload = f"{self.symbol}|{self.side}|{self.qty_bucket:.10f}"
        return hashlib.sha1(payload.encode("utf-8")).hexdigest()[:20]


class OrderIdempotencyGuard:
    """
    Thread-safe TTL cache of in-flight and recently-submitted orders.

    Attributes:
        ttl_seconds: Dedup window. Orders submitted within ``ttl`` seconds of
            an earlier identical submission are suppressed as duplicates.
        max_entries: Memory cap; once exceeded the oldest entry is evicted.
    """

    _QTY_EPS: float = 1e-6

    def __init__(
        self,
        ttl_seconds: Optional[float] = None,
        max_entries: Optional[int] = None,
    ) -> None:
        self.ttl_seconds: float = (
            float(ttl_seconds)
            if ttl_seconds is not None
            else float(getattr(ApexConfig, "ORDER_DEDUP_TTL_SECONDS", 0.0))
        )
        self.max_entries: int = (
            int(max_entries)
            if max_entries is not None
            else int(getattr(ApexConfig, "ORDER_DEDUP_MAX_ENTRIES", 512))
        )
        if self.max_entries < 1:
            raise ValueError(
                f"ORDER_DEDUP_MAX_ENTRIES must be >= 1, got {self.max_entries}"
            )

        self._entries: "OrderedDict[str, Tuple[float, Optional[str]]]" = OrderedDict()
        self._lock = threading.Lock()

    # ── Key construction ─────────────────────────────────────────────────────

    @staticmethod
    def _normalize_side(side: str) -> str:
        if not isinstance(side, str):
            raise TypeError(f"side must be str, got {type(side).__name__}")
        s = side.strip().upper()
        if s not in {"BUY", "SELL"}:
            raise ValueError(f"side must be BUY or SELL, got {side!r}")
        return s

    @staticmethod
    def _normalize_symbol(symbol: str) -> str:
        if not isinstance(symbol, str):
            raise TypeError(f"symbol must be str, got {type(symbol).__name__}")
        s = symbol.strip().upper()
        if not s:
            raise ValueError("symbol must be non-empty")
        return s

    @classmethod
    def _quantize_qty(cls, quantity: float) -> float:
        """
        Round quantity to a bucket. Fractional crypto sizes differ bar-to-bar
        by tiny amounts (e.g. 0.00123451 vs 0.00123449) yet represent the same
        economic order — without bucketing those collide only accidentally.
        """
        try:
            q = float(quantity)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"quantity not numeric: {quantity!r}") from exc
        if not (q > 0.0):
            raise ValueError(f"quantity must be positive, got {q}")
        if q >= 1.0:
            return round(q, 2)
        return round(q, 6)

    def _build_key(
        self, symbol: str, side: str, quantity: float
    ) -> DedupKey:
        return DedupKey(
            symbol=self._normalize_symbol(symbol),
            side=self._normalize_side(side),
            qty_bucket=self._quantize_qty(quantity),
        )

    # ── Internal housekeeping ────────────────────────────────────────────────

    def _evict_expired(self, now: float) -> None:
        """Drop entries whose ``ttl`` has elapsed. Caller must hold lock."""
        if self.ttl_seconds <= 0.0:
            return
        cutoff = now - self.ttl_seconds
        stale = [k for k, (ts, _oid) in self._entries.items() if ts < cutoff]
        for k in stale:
            self._entries.pop(k, None)

    def _cap_size(self) -> None:
        """Evict oldest entries until under ``max_entries``. Caller holds lock."""
        while len(self._entries) > self.max_entries:
            self._entries.popitem(last=False)

    # ── Public API ───────────────────────────────────────────────────────────

    def register(
        self, symbol: str, side: str, quantity: float
    ) -> Tuple[str, bool]:
        """
        Reserve the slot for a pending order.

        Args:
            symbol: Trading symbol.
            side: ``"BUY"`` or ``"SELL"`` (case-insensitive).
            quantity: Positive order quantity.

        Returns:
            Tuple ``(client_order_id, is_new)``:

            - ``client_order_id``: A stable, deterministic token derived from
              the order identity. Safe to pass to brokers that accept client
              order ids.
            - ``is_new``: ``True`` when this order is fresh (caller should
              submit); ``False`` when an identical order is still live in the
              dedup window (caller must suppress the submission).

        Raises:
            TypeError: If ``symbol`` or ``side`` are not strings.
            ValueError: If ``quantity`` is non-positive or ``side`` unknown.
        """
        key = self._build_key(symbol, side, quantity)
        token = key.hexdigest()
        now = time.monotonic()

        with self._lock:
            self._evict_expired(now)
            existing = self._entries.get(token)
            if existing is not None and self.ttl_seconds > 0.0:
                ts, _ = existing
                if (now - ts) < self.ttl_seconds:
                    logger.warning(
                        "Order dedup HIT: %s %s qty=%s (age=%.1fs < ttl=%.1fs) "
                        "— suppressing duplicate submission",
                        key.symbol, key.side, key.qty_bucket,
                        now - ts, self.ttl_seconds,
                    )
                    return token, False

            # Fresh or stale-expired entry — reserve a new slot.
            self._entries[token] = (now, None)
            self._entries.move_to_end(token, last=True)
            self._cap_size()
        return token, True

    def mark_submitted(self, token: str, broker_order_id: str) -> None:
        """
        Attach the broker-assigned order id once the submission succeeds. Keeps
        the slot live for the full TTL so burst retries are still suppressed.
        """
        if not token:
            return
        with self._lock:
            entry = self._entries.get(token)
            if entry is None:
                return
            ts, _ = entry
            self._entries[token] = (ts, str(broker_order_id))

    def forget(self, token: str) -> None:
        """
        Release the slot immediately (broker rejected / caller aborted). A
        follow-up resubmission is then permitted without waiting for the TTL.
        """
        if not token:
            return
        with self._lock:
            self._entries.pop(token, None)

    def is_live(self, symbol: str, side: str, quantity: float) -> bool:
        """
        Return ``True`` when an order with this identity is still inside its
        dedup window.
        """
        if self.ttl_seconds <= 0.0:
            return False
        key = self._build_key(symbol, side, quantity)
        token = key.hexdigest()
        now = time.monotonic()
        with self._lock:
            self._evict_expired(now)
            entry = self._entries.get(token)
            if entry is None:
                return False
            ts, _ = entry
            return (now - ts) < self.ttl_seconds


# ── Module-level singleton ──────────────────────────────────────────────────
_GLOBAL_GUARD: Optional[OrderIdempotencyGuard] = None
_GLOBAL_LOCK = threading.Lock()


def get_order_guard() -> OrderIdempotencyGuard:
    """
    Return the process-wide :class:`OrderIdempotencyGuard`. All broker
    connectors should share the same instance so a stray IBKR retry after an
    Alpaca submission (same symbol / side / qty) is caught.
    """
    global _GLOBAL_GUARD
    if _GLOBAL_GUARD is None:
        with _GLOBAL_LOCK:
            if _GLOBAL_GUARD is None:
                _GLOBAL_GUARD = OrderIdempotencyGuard()
    return _GLOBAL_GUARD
