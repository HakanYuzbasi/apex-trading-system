"""
monitoring/signal_decay_shield.py - Signal Time-Decay & Data Staleness Guard

Prevents trading on stale data by tracking freshness of every data source
and applying exponential decay to signals based on data age.

Three data sources monitored:
1. Price data   — max 120s, linear decay to 0 at 300s
2. Sentiment    — max 30 min, 0.5x after 30 min
3. Features     — max 4 hours, 0.8x after 4h

If ANY source is beyond max age, signal is blocked entirely.
"""

from typing import Dict, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


@dataclass
class FreshnessReport:
    """Report on data freshness for a symbol."""
    is_fresh: bool
    staleness_seconds: Dict[str, float]
    decay_factor: float
    stale_components: list
    timestamp: datetime = field(default_factory=datetime.now)


class SignalDecayShield:
    """
    Time-decay guard that prevents trading on stale data.

    Tracks timestamps of price data, sentiment, and features per symbol.
    Applies multiplicative decay to signal strength based on data age.
    """

    def __init__(
        self,
        max_price_age_seconds: int = 120,
        max_sentiment_age_seconds: int = 1800,
        max_feature_age_seconds: int = 14400,
        price_decay_limit_seconds: int = 300,
    ):
        self.max_price_age = max_price_age_seconds
        self.max_sentiment_age = max_sentiment_age_seconds
        self.max_feature_age = max_feature_age_seconds
        self.price_decay_limit = price_decay_limit_seconds

        # Per-symbol data timestamps: symbol -> {source -> timestamp}
        self._timestamps: Dict[str, Dict[str, datetime]] = {}

        logger.info(
            f"SignalDecayShield initialized: max_price={max_price_age_seconds}s, "
            f"max_sentiment={max_sentiment_age_seconds}s, max_feature={max_feature_age_seconds}s"
        )

    def record_data_timestamp(
        self,
        symbol: str,
        source: str,
        timestamp: Optional[datetime] = None,
    ):
        """
        Record when data was fetched for a symbol.

        Args:
            symbol: Stock symbol
            source: One of "price", "sentiment", "features"
            timestamp: When the data was fetched (default: now)
        """
        if symbol not in self._timestamps:
            self._timestamps[symbol] = {}
        self._timestamps[symbol][source] = timestamp or datetime.now()

    def check_freshness(
        self, symbol: str, now: Optional[datetime] = None
    ) -> FreshnessReport:
        """
        Check data freshness for a symbol across all sources.

        Returns:
            FreshnessReport with staleness details and decay factor
        """
        now = now or datetime.now()
        sym_ts = self._timestamps.get(symbol, {})

        staleness = {}
        stale_components = []
        decay_factor = 1.0

        # Check price data
        price_ts = sym_ts.get("price")
        if price_ts is None:
            staleness["price"] = float("inf")
            stale_components.append("price")
            decay_factor = 0.0
        else:
            age = (now - price_ts).total_seconds()
            staleness["price"] = age
            if age > self.price_decay_limit:
                stale_components.append("price")
                decay_factor = 0.0
            elif age > self.max_price_age:
                # Linear decay from 1.0 at max_price_age to 0.0 at price_decay_limit
                remaining = self.price_decay_limit - self.max_price_age
                if remaining > 0:
                    decay_factor *= max(0.0, 1.0 - (age - self.max_price_age) / remaining)
                stale_components.append("price")

        # Check sentiment data
        sentiment_ts = sym_ts.get("sentiment")
        if sentiment_ts is not None:
            age = (now - sentiment_ts).total_seconds()
            staleness["sentiment"] = age
            if age > self.max_sentiment_age:
                stale_components.append("sentiment")
                decay_factor *= 0.5  # 50% reduction for stale sentiment

        # Check feature data
        feature_ts = sym_ts.get("features")
        if feature_ts is not None:
            age = (now - feature_ts).total_seconds()
            staleness["features"] = age
            if age > self.max_feature_age:
                stale_components.append("features")
                decay_factor *= 0.8  # 20% reduction for stale features

        is_fresh = len(stale_components) == 0

        return FreshnessReport(
            is_fresh=is_fresh,
            staleness_seconds=staleness,
            decay_factor=max(0.0, decay_factor),
            stale_components=stale_components,
        )

    def apply_decay(
        self,
        signal: float,
        confidence: float,
        freshness: FreshnessReport,
    ) -> Tuple[float, float]:
        """
        Apply time-decay to signal and confidence based on freshness.

        Args:
            signal: Raw signal value
            confidence: Raw confidence value
            freshness: FreshnessReport from check_freshness()

        Returns:
            (decayed_signal, decayed_confidence)
        """
        factor = freshness.decay_factor
        return signal * factor, confidence * factor

    def is_data_tradeable(self, symbol: str) -> bool:
        """
        Quick gate: can we trade this symbol given current data freshness?

        Returns False if price data is missing or beyond max age.
        """
        sym_ts = self._timestamps.get(symbol)
        if not sym_ts:
            return False

        price_ts = sym_ts.get("price")
        if price_ts is None:
            return False

        age = (datetime.now() - price_ts).total_seconds()
        return age <= self.price_decay_limit

    def get_diagnostics(self) -> Dict:
        """Return shield state for monitoring."""
        now = datetime.now()
        symbol_status = {}
        for sym, sources in self._timestamps.items():
            ages = {}
            for source, ts in sources.items():
                ages[source] = round((now - ts).total_seconds(), 1)
            symbol_status[sym] = {
                "ages_seconds": ages,
                "tradeable": self.is_data_tradeable(sym),
            }

        return {
            "tracked_symbols": len(self._timestamps),
            "symbol_status": symbol_status,
        }
