from __future__ import annotations

import logging
import time
from datetime import datetime, timezone
from typing import Any

from alpaca.trading.client import TradingClient

logger = logging.getLogger(__name__)

class AlpacaMarketClock:
    """Thin cached wrapper around Alpaca GET /v2/clock."""

    def __init__(self, trading_client: TradingClient, cache_seconds: int = 30):
        self._client = trading_client
        self._cache_seconds = cache_seconds
        self._cached_at: float = 0.0
        self._cached: dict[str, Any] = {}

    def _refresh(self) -> dict[str, Any]:
        now = time.monotonic()
        if now - self._cached_at < self._cache_seconds and self._cached:
            return self._cached
            
        try:
            clock = self._client.get_clock()
            self._cached = {
                "is_open": bool(clock.is_open),
                "next_open": clock.next_open,
                "next_close": clock.next_close,
            }
            self._cached_at = now
        except Exception as e:
            logger.warning("AlpacaMarketClock: failed to fetch clock from Alpaca: %s", e)
            if not self._cached:
                # If we've never fetched, fail closed
                self._cached = {
                    "is_open": False,
                    "next_open": datetime.now(timezone.utc),
                    "next_close": datetime.now(timezone.utc),
                }
        return self._cached

    def is_open(self) -> bool:
        try:
            return self._refresh()["is_open"]
        except Exception:
            return False  # fail-closed

    def minutes_to_open(self) -> float:
        try:
            next_open = self._refresh()["next_open"]
            if isinstance(next_open, str):
                next_open = datetime.fromisoformat(next_open.replace("Z", "+00:00"))
            if next_open.tzinfo is None:
                next_open = next_open.replace(tzinfo=timezone.utc)
            delta = (next_open - datetime.now(timezone.utc)).total_seconds()
            return max(0.0, delta / 60.0)
        except Exception:
            return 999.0
