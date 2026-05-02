"""
quant_system/risk/btc_dominance_monitor.py

BTC dominance signal for crypto alt/alt pairs.

CoinGecko /api/v3/global returns BTC dominance percentage (no API key needed).
We track the rolling trend: if BTC.D has risen > BTC_DOM_RISING_THRESHOLD pp
over the last BTC_DOM_WINDOW_COUNT readings, alt/alt pair entries are blocked.

Usage:
    from quant_system.risk.btc_dominance_monitor import get_btc_dominance_monitor
    mon = get_btc_dominance_monitor()
    if mon.is_btc_dominance_rising():
        # skip entry for altcoin pairs
"""
from __future__ import annotations

import logging
import time
from collections import deque

logger = logging.getLogger(__name__)

_BTC_DOM_TTL        = 1_800.0   # refresh every 30 min
_BTC_DOM_WINDOW     = 3         # compare current vs 3 readings ago
_BTC_DOM_THRESHOLD  = 0.8       # percentage-point rise to trigger filter
_COINGECKO_URL      = "https://api.coingecko.com/api/v3/global"


class BtcDominanceMonitor:
    """
    Tracks BTC market dominance and detects rising-dominance regimes.
    Rising BTC.D means capital is rotating from alts back to BTC — bad for
    alt/alt pairs that assume correlated altcoin momentum.
    """

    def __init__(self) -> None:
        self._history: deque[float] = deque(maxlen=_BTC_DOM_WINDOW + 1)
        self._last_fetch: float = 0.0
        self._last_dom: float   = 0.0
        self._fetch_failed:bool = False

    def _fetch(self) -> None:
        try:
            import urllib.request as _ur
            import json as _json
            with _ur.urlopen(_COINGECKO_URL, timeout=5) as resp:
                data = _json.loads(resp.read())
            dom = float(data["data"]["bitcoin_dominance_percentage"])
            self._history.append(dom)
            self._last_dom  = dom
            self._last_fetch = time.monotonic()
            self._fetch_failed = False
            logger.debug("BTC dominance: %.2f%%", dom)
        except Exception as exc:
            logger.debug("BtcDominanceMonitor: fetch failed — %s", exc)
            self._last_fetch = time.monotonic()   # suppress retry until TTL
            self._fetch_failed = True

    def _maybe_refresh(self) -> None:
        if time.monotonic() - self._last_fetch >= _BTC_DOM_TTL:
            self._fetch()

    def current_dominance(self) -> float:
        """Latest BTC dominance percentage (0–100)."""
        self._maybe_refresh()
        return self._last_dom

    def force_refresh(self) -> None:
        """Proactively fetch latest BTC dominance (called on a schedule from harness)."""
        self._fetch()

    def is_btc_dominance_rising(self) -> bool:
        """
        True if BTC.D has risen by more than _BTC_DOM_THRESHOLD pp since
        _BTC_DOM_WINDOW readings ago.  Returns False when data is unavailable
        or insufficient history exists (give benefit of the doubt).
        """
        self._maybe_refresh()
        if self._fetch_failed or len(self._history) < _BTC_DOM_WINDOW + 1:
            return False
        oldest = self._history[0]
        newest = self._history[-1]
        return (newest - oldest) > _BTC_DOM_THRESHOLD


_INSTANCE: BtcDominanceMonitor | None = None


def get_btc_dominance_monitor() -> BtcDominanceMonitor:
    global _INSTANCE
    if _INSTANCE is None:
        _INSTANCE = BtcDominanceMonitor()
    return _INSTANCE
