"""
risk/overnight_risk_guard.py - Overnight Gap Risk Protection

Manages overnight exposure to protect against gap risk:
- Reduces position sizes in final hour of trading
- Blocks new entries in last 30 minutes
- Calculates overnight VaR and adjusts exposure
- Forces partial exits if overnight risk too high

Gap risk metrics:
- Average overnight gap: ~0.3% for S&P 500
- 95th percentile gap: ~1.5%
- Tail gap (black swan): 5%+

Protection rules:
- Last 30 min: No new entries
- Last 15 min: Reduce to 70% max exposure
- High VIX (>25): Additional 20% reduction overnight
- Earnings next day: Exit that position before close
"""

import logging
from typing import Dict, List, Optional, Set, Tuple
from dataclasses import dataclass
from datetime import datetime, time
from enum import Enum
import numpy as np

logger = logging.getLogger(__name__)


class MarketPhase(Enum):
    PRE_MARKET = "pre_market"
    OPEN = "open"
    REGULAR = "regular"
    POWER_HOUR = "power_hour"  # Last hour
    CLOSE_PREP = "close_prep"  # Last 30 min
    FINAL_MINUTES = "final_minutes"  # Last 15 min
    AFTER_HOURS = "after_hours"
    CLOSED = "closed"


@dataclass
class OvernightRiskAssessment:
    """Assessment of overnight gap risk."""
    market_phase: MarketPhase
    minutes_to_close: Optional[int]
    overnight_var_pct: float  # Estimated overnight VaR
    recommended_exposure_pct: float  # 0-100
    entry_allowed: bool
    exit_urgency: str  # "none", "low", "medium", "high"
    positions_to_reduce: List[Tuple[str, float]]  # (symbol, reduction_pct)
    force_exit_symbols: Set[str]  # Must exit before close


class OvernightRiskGuard:
    """
    Overnight gap risk protection.

    Monitors time to market close and adjusts exposure to
    minimize overnight gap risk while preserving positions
    with strong momentum.
    """

    def __init__(
        self,
        market_open: time = time(9, 30),
        market_close: time = time(16, 0),
        no_entry_minutes: int = 30,
        reduce_exposure_minutes: int = 60,
        final_reduction_minutes: int = 15,
        max_overnight_var_pct: float = 2.0,
        high_vix_threshold: float = 25.0,
        high_vix_reduction: float = 0.20,
    ):
        self.market_open = market_open
        self.market_close = market_close
        self.no_entry_minutes = no_entry_minutes
        self.reduce_exposure_minutes = reduce_exposure_minutes
        self.final_reduction_minutes = final_reduction_minutes
        self.max_overnight_var = max_overnight_var_pct / 100
        self.high_vix_threshold = high_vix_threshold
        self.high_vix_reduction = high_vix_reduction

        # Historical gap data (per-symbol volatility)
        self._symbol_overnight_vol: Dict[str, float] = {}
        self._default_overnight_vol = 0.015  # 1.5% default

        # Earnings calendar (symbols reporting after close or before open)
        self._earnings_tonight: Set[str] = set()
        self._earnings_tomorrow_premarket: Set[str] = set()

        logger.info(
            f"OvernightRiskGuard initialized: "
            f"no_entry={no_entry_minutes}min, "
            f"max_overnight_var={max_overnight_var_pct}%"
        )

    def set_symbol_overnight_vol(self, symbol: str, vol: float):
        """Set historical overnight volatility for a symbol."""
        self._symbol_overnight_vol[symbol] = vol

    def add_earnings_tonight(self, symbol: str):
        """Mark symbol as having earnings after today's close."""
        self._earnings_tonight.add(symbol)

    def add_earnings_tomorrow(self, symbol: str):
        """Mark symbol as having earnings before tomorrow's open."""
        self._earnings_tomorrow_premarket.add(symbol)

    def clear_earnings(self):
        """Clear earnings calendar (call at start of each day)."""
        self._earnings_tonight.clear()
        self._earnings_tomorrow_premarket.clear()

    def get_market_phase(self, current_time: Optional[datetime] = None) -> MarketPhase:
        """Determine current market phase."""
        now = current_time or datetime.now()
        t = now.time()

        # Check if weekend
        if now.weekday() >= 5:
            return MarketPhase.CLOSED

        # Pre-market
        if t < self.market_open:
            if t >= time(4, 0):
                return MarketPhase.PRE_MARKET
            return MarketPhase.CLOSED

        # After hours
        if t >= self.market_close:
            if t <= time(20, 0):
                return MarketPhase.AFTER_HOURS
            return MarketPhase.CLOSED

        # Regular hours - calculate minutes to close
        close_dt = datetime.combine(now.date(), self.market_close)
        minutes_to_close = (close_dt - now).total_seconds() / 60

        if minutes_to_close <= self.final_reduction_minutes:
            return MarketPhase.FINAL_MINUTES
        elif minutes_to_close <= self.no_entry_minutes:
            return MarketPhase.CLOSE_PREP
        elif minutes_to_close <= self.reduce_exposure_minutes:
            return MarketPhase.POWER_HOUR
        elif minutes_to_close <= 60:  # First 30 min
            return MarketPhase.OPEN
        else:
            return MarketPhase.REGULAR

    def assess_overnight_risk(
        self,
        positions: Dict[str, Dict],
        capital: float,
        vix_level: float = 20.0,
        current_time: Optional[datetime] = None,
    ) -> OvernightRiskAssessment:
        """
        Assess overnight risk and provide recommendations.

        Args:
            positions: Dict of symbol -> position info
            capital: Total portfolio capital
            vix_level: Current VIX level
            current_time: Current time (default: now)

        Returns:
            OvernightRiskAssessment with recommendations
        """
        now = current_time or datetime.now()
        phase = self.get_market_phase(now)

        # Calculate minutes to close
        minutes_to_close = None
        if phase not in [MarketPhase.CLOSED, MarketPhase.AFTER_HOURS, MarketPhase.PRE_MARKET]:
            close_dt = datetime.combine(now.date(), self.market_close)
            minutes_to_close = max(0, int((close_dt - now).total_seconds() / 60))

        # Calculate overnight VaR
        overnight_var = self._calculate_overnight_var(positions, capital, vix_level)

        # Determine entry allowance
        entry_allowed = phase in [MarketPhase.OPEN, MarketPhase.REGULAR, MarketPhase.POWER_HOUR]

        # Determine recommended exposure
        if phase == MarketPhase.FINAL_MINUTES:
            base_exposure = 70.0
        elif phase == MarketPhase.CLOSE_PREP:
            base_exposure = 85.0
        elif phase == MarketPhase.POWER_HOUR:
            base_exposure = 95.0
        else:
            base_exposure = 100.0

        # VIX adjustment
        if vix_level > self.high_vix_threshold:
            base_exposure *= (1 - self.high_vix_reduction)

        # Overnight VaR adjustment
        if overnight_var > self.max_overnight_var:
            var_ratio = self.max_overnight_var / overnight_var
            base_exposure = min(base_exposure, var_ratio * 100)

        # Determine exit urgency
        if phase == MarketPhase.FINAL_MINUTES:
            exit_urgency = "high"
        elif phase == MarketPhase.CLOSE_PREP:
            exit_urgency = "medium"
        elif phase == MarketPhase.POWER_HOUR and overnight_var > self.max_overnight_var:
            exit_urgency = "medium"
        else:
            exit_urgency = "none"

        # Find positions to reduce and force exits
        positions_to_reduce = []
        force_exit = set()

        # Must exit: positions with earnings tonight/tomorrow
        force_exit.update(self._earnings_tonight & set(positions.keys()))
        force_exit.update(self._earnings_tomorrow_premarket & set(positions.keys()))

        # Reduce: highest overnight vol positions
        if phase in [MarketPhase.CLOSE_PREP, MarketPhase.FINAL_MINUTES]:
            if overnight_var > self.max_overnight_var * 0.8:
                # Reduce highest-vol positions
                vol_positions = []
                for symbol, pos in positions.items():
                    if symbol in force_exit:
                        continue
                    vol = self._symbol_overnight_vol.get(symbol, self._default_overnight_vol)
                    value = abs(pos.get("market_value", pos.get("value", 0)))
                    vol_positions.append((symbol, vol, value))

                vol_positions.sort(key=lambda x: x[1], reverse=True)

                # Reduce top 3 highest vol positions
                for symbol, vol, value in vol_positions[:3]:
                    if vol > 0.02:  # 2% overnight vol
                        reduction = min(0.5, vol / 0.04)  # Up to 50% reduction
                        positions_to_reduce.append((symbol, reduction))

        return OvernightRiskAssessment(
            market_phase=phase,
            minutes_to_close=minutes_to_close,
            overnight_var_pct=overnight_var * 100,
            recommended_exposure_pct=base_exposure,
            entry_allowed=entry_allowed,
            exit_urgency=exit_urgency,
            positions_to_reduce=positions_to_reduce,
            force_exit_symbols=force_exit,
        )

    def _calculate_overnight_var(
        self,
        positions: Dict[str, Dict],
        capital: float,
        vix_level: float,
    ) -> float:
        """
        Calculate portfolio overnight VaR.

        Uses per-symbol overnight volatility and VIX scaling.
        """
        if not positions or capital <= 0:
            return 0.0

        # VIX scaling factor (VIX 20 = baseline, scales linearly)
        vix_mult = vix_level / 20.0

        total_var = 0.0
        for symbol, pos in positions.items():
            value = abs(pos.get("market_value", pos.get("value", 0)))
            vol = self._symbol_overnight_vol.get(symbol, self._default_overnight_vol)

            # 95% VaR = 1.65 * vol * value
            symbol_var = 1.65 * vol * vix_mult * value
            total_var += symbol_var ** 2  # Sum of variances (simplified)

        # Portfolio VaR (simplified - ignores correlations)
        portfolio_var = np.sqrt(total_var)
        return portfolio_var / capital

    def should_block_entry(self, current_time: Optional[datetime] = None) -> Tuple[bool, str]:
        """Check if entries should be blocked due to market phase."""
        phase = self.get_market_phase(current_time)

        if phase == MarketPhase.FINAL_MINUTES:
            return True, "Final 15 minutes - no new entries"
        elif phase == MarketPhase.CLOSE_PREP:
            return True, "Last 30 minutes - no new entries"
        elif phase in [MarketPhase.AFTER_HOURS, MarketPhase.CLOSED]:
            return True, f"Market {phase.value}"
        elif phase == MarketPhase.PRE_MARKET:
            return True, "Pre-market - no entries until open"

        return False, ""

    def get_position_size_multiplier(
        self,
        vix_level: float = 20.0,
        current_time: Optional[datetime] = None,
    ) -> float:
        """Get position sizing multiplier based on time and VIX."""
        phase = self.get_market_phase(current_time)

        if phase == MarketPhase.FINAL_MINUTES:
            return 0.0  # No new positions
        elif phase == MarketPhase.CLOSE_PREP:
            return 0.0  # No new positions
        elif phase == MarketPhase.POWER_HOUR:
            base = 0.7
        else:
            base = 1.0

        # VIX adjustment
        if vix_level > self.high_vix_threshold:
            base *= (1 - self.high_vix_reduction)

        return base

    def get_symbols_to_exit_before_close(self) -> Set[str]:
        """Get symbols that must be exited before today's close."""
        return self._earnings_tonight | self._earnings_tomorrow_premarket

    def is_near_close(self, minutes: int = 30, current_time: Optional[datetime] = None) -> bool:
        """Check if within N minutes of market close."""
        now = current_time or datetime.now()
        phase = self.get_market_phase(now)

        if phase in [MarketPhase.CLOSED, MarketPhase.AFTER_HOURS, MarketPhase.PRE_MARKET]:
            return False

        close_dt = datetime.combine(now.date(), self.market_close)
        minutes_to_close = (close_dt - now).total_seconds() / 60
        return minutes_to_close <= minutes

    def get_diagnostics(self) -> Dict:
        """Return guard state for monitoring."""
        assessment = self.assess_overnight_risk({}, 1000000)
        return {
            "market_phase": assessment.market_phase.value,
            "minutes_to_close": assessment.minutes_to_close,
            "entry_allowed": assessment.entry_allowed,
            "earnings_tonight": list(self._earnings_tonight),
            "earnings_tomorrow": list(self._earnings_tomorrow_premarket),
            "symbols_tracked_vol": len(self._symbol_overnight_vol),
        }
