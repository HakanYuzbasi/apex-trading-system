"""
risk/liquidity_guard.py - Liquidity Crisis Detection

Monitors market liquidity conditions and blocks trading when
liquidity is deteriorating. Illiquid conditions cause:
- Wide bid-ask spreads (execution slippage)
- Market impact (moving price against you)
- Inability to exit positions

Liquidity indicators:
- Bid-ask spread vs historical
- Volume vs 20-day average
- Order book depth (if available)
- Time of day patterns

Liquidity regimes:
NORMAL:   Spread < 0.1%, Volume > 80% avg
THIN:     Spread 0.1-0.3%, Volume 50-80% avg
STRESSED: Spread 0.3-0.5%, Volume 30-50% avg
CRISIS:   Spread > 0.5%, Volume < 30% avg

Actions:
NORMAL:   Full trading
THIN:     Reduce position size 25%
STRESSED: Reduce 50%, no new entries
CRISIS:   Emergency exits only
"""

import logging
from typing import Dict, List, Optional, Set, Tuple
from dataclasses import dataclass, field
from datetime import datetime, time, timedelta
from enum import IntEnum
from collections import deque
import numpy as np

logger = logging.getLogger(__name__)


class LiquidityRegime(IntEnum):
    NORMAL = 0
    THIN = 1
    STRESSED = 2
    CRISIS = 3


@dataclass
class LiquidityMetrics:
    """Liquidity metrics for a symbol."""
    symbol: str
    bid_ask_spread_pct: float
    volume_ratio: float  # Today's volume / 20-day avg
    avg_spread_20d: float
    spread_z_score: float  # How many std devs above normal
    regime: LiquidityRegime
    is_tradeable: bool
    position_size_multiplier: float
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class MarketLiquidityState:
    """Overall market liquidity assessment."""
    regime: LiquidityRegime
    avg_spread_pct: float
    avg_volume_ratio: float
    stressed_symbols: List[str]
    crisis_symbols: List[str]
    entry_allowed: bool
    market_wide_size_mult: float


class LiquidityGuard:
    """
    Market liquidity monitoring and protection.

    Tracks bid-ask spreads and volume for all traded symbols,
    detecting when liquidity is deteriorating and adjusting
    position sizing accordingly.
    """

    def __init__(
        self,
        thin_spread_threshold: float = 0.001,      # 0.1%
        stressed_spread_threshold: float = 0.003,  # 0.3%
        crisis_spread_threshold: float = 0.005,    # 0.5%
        thin_volume_threshold: float = 0.80,       # 80% of avg
        stressed_volume_threshold: float = 0.50,   # 50% of avg
        crisis_volume_threshold: float = 0.30,     # 30% of avg
        spread_history_size: int = 100,
        min_dollar_volume: float = 1_000_000,      # $1M min daily volume
    ):
        self.thin_spread = thin_spread_threshold
        self.stressed_spread = stressed_spread_threshold
        self.crisis_spread = crisis_spread_threshold
        self.thin_volume = thin_volume_threshold
        self.stressed_volume = stressed_volume_threshold
        self.crisis_volume = crisis_volume_threshold
        self.spread_history_size = spread_history_size
        self.min_dollar_volume = min_dollar_volume

        # Per-symbol tracking
        self._spread_history: Dict[str, deque] = {}
        self._volume_history: Dict[str, deque] = {}
        self._avg_volume_20d: Dict[str, float] = {}
        self._last_metrics: Dict[str, LiquidityMetrics] = {}

        # Illiquid symbol blacklist
        self._blacklist: Set[str] = set()
        self._blacklist_reasons: Dict[str, str] = {}

        logger.info(
            f"LiquidityGuard initialized: "
            f"crisis_spread={crisis_spread_threshold:.2%}, "
            f"crisis_volume={crisis_volume_threshold:.0%}"
        )

    def update_quote(
        self,
        symbol: str,
        bid: float,
        ask: float,
        volume: float,
        avg_volume_20d: Optional[float] = None,
    ) -> LiquidityMetrics:
        """
        Update liquidity metrics with new quote data.

        Args:
            symbol: Stock symbol
            bid: Current bid price
            ask: Current ask price
            volume: Current day's volume
            avg_volume_20d: 20-day average volume (if known)

        Returns:
            LiquidityMetrics with current assessment
        """
        # Calculate spread
        mid = (bid + ask) / 2 if bid > 0 and ask > 0 else 0
        spread_pct = (ask - bid) / mid if mid > 0 else 0

        # Initialize history if needed
        if symbol not in self._spread_history:
            self._spread_history[symbol] = deque(maxlen=self.spread_history_size)
            self._volume_history[symbol] = deque(maxlen=20)

        # Update history
        self._spread_history[symbol].append(spread_pct)
        self._volume_history[symbol].append(volume)

        # Store 20d avg volume
        if avg_volume_20d is not None:
            self._avg_volume_20d[symbol] = avg_volume_20d

        # Calculate metrics
        avg_vol = self._avg_volume_20d.get(symbol, volume)
        volume_ratio = volume / avg_vol if avg_vol > 0 else 1.0

        # Spread statistics
        spread_list = list(self._spread_history[symbol])
        if len(spread_list) >= 5:
            avg_spread = np.mean(spread_list)
            std_spread = np.std(spread_list) if len(spread_list) >= 10 else avg_spread * 0.3
            z_score = (spread_pct - avg_spread) / std_spread if std_spread > 0 else 0
        else:
            avg_spread = spread_pct
            z_score = 0

        # Determine regime
        regime = self._determine_regime(spread_pct, volume_ratio)

        # Determine tradability
        is_tradeable = regime < LiquidityRegime.CRISIS
        if symbol in self._blacklist:
            is_tradeable = False

        # Position size multiplier
        size_mult = {
            LiquidityRegime.NORMAL: 1.0,
            LiquidityRegime.THIN: 0.75,
            LiquidityRegime.STRESSED: 0.50,
            LiquidityRegime.CRISIS: 0.0,
        }[regime]

        metrics = LiquidityMetrics(
            symbol=symbol,
            bid_ask_spread_pct=spread_pct,
            volume_ratio=volume_ratio,
            avg_spread_20d=avg_spread,
            spread_z_score=z_score,
            regime=regime,
            is_tradeable=is_tradeable,
            position_size_multiplier=size_mult,
        )

        self._last_metrics[symbol] = metrics

        if regime >= LiquidityRegime.STRESSED:
            logger.warning(
                f"Liquidity {regime.name} for {symbol}: "
                f"spread={spread_pct:.3%}, vol_ratio={volume_ratio:.1%}"
            )

        return metrics

    def _determine_regime(self, spread_pct: float, volume_ratio: float) -> LiquidityRegime:
        """Determine liquidity regime from spread and volume."""
        # Spread-based classification
        if spread_pct >= self.crisis_spread:
            spread_regime = LiquidityRegime.CRISIS
        elif spread_pct >= self.stressed_spread:
            spread_regime = LiquidityRegime.STRESSED
        elif spread_pct >= self.thin_spread:
            spread_regime = LiquidityRegime.THIN
        else:
            spread_regime = LiquidityRegime.NORMAL

        # Volume-based classification
        if volume_ratio <= self.crisis_volume:
            volume_regime = LiquidityRegime.CRISIS
        elif volume_ratio <= self.stressed_volume:
            volume_regime = LiquidityRegime.STRESSED
        elif volume_ratio <= self.thin_volume:
            volume_regime = LiquidityRegime.THIN
        else:
            volume_regime = LiquidityRegime.NORMAL

        # Take the worse of the two
        return max(spread_regime, volume_regime)

    def assess_market_liquidity(
        self,
        symbols: Optional[List[str]] = None,
    ) -> MarketLiquidityState:
        """
        Assess overall market liquidity.

        Args:
            symbols: Symbols to include (default: all tracked)

        Returns:
            MarketLiquidityState with aggregate metrics
        """
        if symbols is None:
            symbols = list(self._last_metrics.keys())

        if not symbols:
            return MarketLiquidityState(
                regime=LiquidityRegime.NORMAL,
                avg_spread_pct=0.0,
                avg_volume_ratio=1.0,
                stressed_symbols=[],
                crisis_symbols=[],
                entry_allowed=True,
                market_wide_size_mult=1.0,
            )

        spreads = []
        volumes = []
        stressed = []
        crisis = []

        for symbol in symbols:
            metrics = self._last_metrics.get(symbol)
            if metrics:
                spreads.append(metrics.bid_ask_spread_pct)
                volumes.append(metrics.volume_ratio)
                if metrics.regime == LiquidityRegime.STRESSED:
                    stressed.append(symbol)
                elif metrics.regime == LiquidityRegime.CRISIS:
                    crisis.append(symbol)

        avg_spread = np.mean(spreads) if spreads else 0.0
        avg_volume = np.mean(volumes) if volumes else 1.0

        # Market-wide regime (if >20% in crisis, market is in crisis)
        crisis_pct = len(crisis) / len(symbols) if symbols else 0
        stressed_pct = (len(stressed) + len(crisis)) / len(symbols) if symbols else 0

        if crisis_pct >= 0.20:
            market_regime = LiquidityRegime.CRISIS
        elif stressed_pct >= 0.30:
            market_regime = LiquidityRegime.STRESSED
        elif stressed_pct >= 0.15:
            market_regime = LiquidityRegime.THIN
        else:
            market_regime = LiquidityRegime.NORMAL

        # Entry allowed if not in crisis
        entry_allowed = market_regime < LiquidityRegime.STRESSED

        # Market-wide size multiplier
        size_mult = {
            LiquidityRegime.NORMAL: 1.0,
            LiquidityRegime.THIN: 0.85,
            LiquidityRegime.STRESSED: 0.60,
            LiquidityRegime.CRISIS: 0.25,
        }[market_regime]

        return MarketLiquidityState(
            regime=market_regime,
            avg_spread_pct=avg_spread,
            avg_volume_ratio=avg_volume,
            stressed_symbols=stressed,
            crisis_symbols=crisis,
            entry_allowed=entry_allowed,
            market_wide_size_mult=size_mult,
        )

    def get_position_size_multiplier(self, symbol: str) -> float:
        """Get position size multiplier for a symbol."""
        metrics = self._last_metrics.get(symbol)
        if metrics:
            return metrics.position_size_multiplier
        return 1.0  # Default to full size if not tracked

    def should_block_entry(self, symbol: str) -> Tuple[bool, str]:
        """Check if entry should be blocked for a symbol."""
        if symbol in self._blacklist:
            reason = self._blacklist_reasons.get(symbol, "Blacklisted")
            return True, f"Symbol blacklisted: {reason}"

        metrics = self._last_metrics.get(symbol)
        if metrics:
            if metrics.regime >= LiquidityRegime.CRISIS:
                return True, f"Liquidity crisis: spread={metrics.bid_ask_spread_pct:.2%}"
            if metrics.regime >= LiquidityRegime.STRESSED:
                return True, f"Stressed liquidity: spread={metrics.bid_ask_spread_pct:.2%}"

        return False, ""

    def blacklist_symbol(self, symbol: str, reason: str = "Manual blacklist"):
        """Add symbol to illiquid blacklist."""
        self._blacklist.add(symbol)
        self._blacklist_reasons[symbol] = reason
        logger.warning(f"Symbol {symbol} blacklisted: {reason}")

    def remove_from_blacklist(self, symbol: str):
        """Remove symbol from blacklist."""
        self._blacklist.discard(symbol)
        self._blacklist_reasons.pop(symbol, None)

    def get_illiquid_symbols(self) -> List[str]:
        """Get list of currently illiquid symbols."""
        illiquid = list(self._blacklist)
        for symbol, metrics in self._last_metrics.items():
            if metrics.regime >= LiquidityRegime.STRESSED and symbol not in illiquid:
                illiquid.append(symbol)
        return illiquid

    def set_avg_volume(self, symbol: str, avg_volume_20d: float):
        """Set 20-day average volume for a symbol."""
        self._avg_volume_20d[symbol] = avg_volume_20d

    def get_metrics(self, symbol: str) -> Optional[LiquidityMetrics]:
        """Get last metrics for a symbol."""
        return self._last_metrics.get(symbol)

    def get_diagnostics(self) -> Dict:
        """Return guard state for monitoring."""
        market_state = self.assess_market_liquidity()

        regime_counts = {regime.name: 0 for regime in LiquidityRegime}
        for metrics in self._last_metrics.values():
            regime_counts[metrics.regime.name] += 1

        return {
            "symbols_tracked": len(self._last_metrics),
            "market_regime": market_state.regime.name,
            "avg_spread_pct": round(market_state.avg_spread_pct * 100, 3),
            "avg_volume_ratio": round(market_state.avg_volume_ratio * 100, 1),
            "regime_distribution": regime_counts,
            "stressed_count": len(market_state.stressed_symbols),
            "crisis_count": len(market_state.crisis_symbols),
            "blacklisted": list(self._blacklist),
            "entry_allowed": market_state.entry_allowed,
        }
