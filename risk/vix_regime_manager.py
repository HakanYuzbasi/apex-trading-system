"""
risk/vix_regime_manager.py - VIX-Based Adaptive Risk Management

Uses free VIX data (via Yahoo Finance) to:
- Detect volatility regimes
- Adjust position sizing based on market fear
- Implement tail risk hedging signals
- Scale risk during volatility spikes

VIX Regimes:
- Complacency: VIX < 12 (reduce positions, buy cheap puts)
- Normal: VIX 12-20 (standard sizing)
- Elevated: VIX 20-30 (reduce by 30%)
- Fear: VIX 30-40 (reduce by 50%)
- Panic: VIX > 40 (reduce by 75% or close)
"""

import numpy as np
import pandas as pd
from typing import Dict, Optional, Tuple, List
from datetime import datetime, timedelta
from dataclasses import dataclass
from enum import Enum
import logging

from config import ApexConfig

logger = logging.getLogger(__name__)

# Try to import yfinance for free VIX data
try:
    import yfinance as yf
    YFINANCE_AVAILABLE = True
except ImportError:
    YFINANCE_AVAILABLE = False
    logger.warning("yfinance not available. Install with: pip install yfinance")


class VIXRegime(Enum):
    """VIX-based volatility regime classification."""
    COMPLACENCY = "complacency"  # VIX < 12
    NORMAL = "normal"            # VIX 12-20
    ELEVATED = "elevated"        # VIX 20-30
    FEAR = "fear"                # VIX 30-40
    PANIC = "panic"              # VIX > 40


@dataclass
class VIXState:
    """Current VIX state and derived metrics."""
    current_vix: float
    vix_percentile: float  # Percentile vs 1-year history
    vix_change_1d: float   # 1-day change
    vix_change_5d: float   # 5-day change
    regime: VIXRegime
    risk_multiplier: float
    timestamp: datetime
    
    def to_dict(self) -> Dict:
        return {
            'current_vix': self.current_vix,
            'vix_percentile': self.vix_percentile,
            'vix_change_1d': self.vix_change_1d,
            'vix_change_5d': self.vix_change_5d,
            'regime': self.regime.value,
            'risk_multiplier': self.risk_multiplier,
            'timestamp': self.timestamp.isoformat()
        }


class VIXRegimeManager:
    """
    VIX-based adaptive risk management.
    
    Features:
    - Real-time VIX regime detection
    - Dynamic position sizing based on VIX
    - Tail risk hedging signals
    - VIX term structure analysis
    """
    
    # VIX regime thresholds
    # Dynamic Z-Score Thresholds (std devs above 60d mean)
    # Complacency: Z < -1.0
    # Normal: -1.0 <= Z < 0.5
    # Elevated: 0.5 <= Z < 1.5
    # Fear: 1.5 <= Z < 3.0
    # Panic: Z >= 3.0
    # Tuned 2026-03-11: raised FEAR boundary from 1.5 → 2.0σ
    # At 1.5σ the FEAR regime fired ~13% of the time (far too often).
    # At 2.0σ it fires only ~2-5% — correctly reserved for genuine extremes.
    Z_SCORE_THRESHOLDS = {
        VIXRegime.COMPLACENCY: (-float('inf'), -1.0),
        VIXRegime.NORMAL: (-1.0, 0.5),
        VIXRegime.ELEVATED: (0.5, 2.0),
        VIXRegime.FEAR: (2.0, 3.5),
        VIXRegime.PANIC: (3.5, float('inf'))
    }

    # Fallback static thresholds — sourced from ApexConfig so every module in
    # the risk stack classifies VIX against the same boundaries.
    # Buckets:
    #   COMPLACENCY : vix <  VIX_REGIME_LOW
    #   NORMAL      : VIX_REGIME_LOW  <= vix < VIX_REGIME_MID
    #   ELEVATED    : VIX_REGIME_MID  <= vix < VIX_REGIME_HIGH
    #   FEAR        : VIX_REGIME_HIGH <= vix < VIX_REGIME_CRISIS
    #   PANIC       : vix >= VIX_REGIME_CRISIS
    STATIC_THRESHOLDS = {
        VIXRegime.COMPLACENCY: (0.0, float(ApexConfig.VIX_REGIME_LOW)),
        VIXRegime.NORMAL:      (float(ApexConfig.VIX_REGIME_LOW),   float(ApexConfig.VIX_REGIME_MID)),
        VIXRegime.ELEVATED:    (float(ApexConfig.VIX_REGIME_MID),   float(ApexConfig.VIX_REGIME_HIGH)),
        VIXRegime.FEAR:        (float(ApexConfig.VIX_REGIME_HIGH),  float(ApexConfig.VIX_REGIME_CRISIS)),
        VIXRegime.PANIC:       (float(ApexConfig.VIX_REGIME_CRISIS), float('inf')),
    }
    
    # Risk multipliers by regime — tuned 2026-03-11
    # Goal: preserve participation in stressed-but-trending markets.
    # Combined effect at VIX=25.73 (post-tuning):
    #   VIXRegimeManager × EnhancedSignalFilter × DynamicConfigAdjuster
    #   ≈ 0.90 × 0.86 × 0.95 ≈ 0.74  (was 0.65 × 0.71 × 0.95 ≈ 0.44)
    RISK_MULTIPLIERS = {
        VIXRegime.COMPLACENCY: 0.8,  # Slightly reduce (complacency = hidden risk)
        VIXRegime.NORMAL: 1.0,
        VIXRegime.ELEVATED: 0.90,    # Raised from 0.85: stressed market, keep most size
        VIXRegime.FEAR: 0.75,        # Raised from 0.65: meaningful reduction but still tradeable
        VIXRegime.PANIC: 0.45        # Raised from 0.40: Z≥3.5σ is a true crash event
    }
    
    def __init__(
        self,
        cache_minutes: int = 5,
        use_term_structure: bool = True
    ):
        """
        Initialize VIX regime manager.
        
        Args:
            cache_minutes: Minutes to cache VIX data
            use_term_structure: Whether to use VIX term structure
        """
        self.cache_minutes = cache_minutes
        self.use_term_structure = use_term_structure

        # Cache
        self._vix_cache: Optional[pd.DataFrame] = None
        self._last_fetch: Optional[datetime] = None
        self._current_state: Optional[VIXState] = None

        # History for percentile calculations
        self._vix_history: List[float] = []

        # Rate-limit VIX spike warning to once per hour
        self._last_vix_spike_log: Optional[datetime] = None
        
        logger.info("VIXRegimeManager initialized")
        if not YFINANCE_AVAILABLE:
            logger.warning("yfinance not available - using simulated VIX")
    
    def fetch_vix_data(self, lookback_days: int = 365) -> Optional[pd.DataFrame]:
        """
        Fetch VIX historical data from Yahoo Finance (free).
        
        Args:
            lookback_days: Days of history to fetch
        
        Returns:
            DataFrame with VIX data
        """
        if not YFINANCE_AVAILABLE:
            return self._simulate_vix_data(lookback_days)
        
        try:
            # yfinance 1.0: history(start=, end=) raises KeyError('chart').
            # Use period= strings instead; map lookback_days to the closest period.
            if lookback_days <= 30:
                period = "1mo"
            elif lookback_days <= 90:
                period = "3mo"
            elif lookback_days <= 180:
                period = "6mo"
            elif lookback_days <= 365:
                period = "1y"
            elif lookback_days <= 730:
                period = "2y"
            else:
                period = "5y"

            vix = yf.Ticker("^VIX")
            data = vix.history(period=period)

            if data.empty:
                logger.warning("No VIX data received from yfinance")
                return self._simulate_vix_data(lookback_days)

            self._vix_cache = data
            self._last_fetch = datetime.now()

            logger.debug(f"Fetched {len(data)} days of VIX data (period={period})")
            return data

        except Exception as e:
            logger.error(f"Error fetching VIX data: {e}")
            return self._simulate_vix_data(lookback_days)
    
    def _simulate_vix_data(self, lookback_days: int) -> pd.DataFrame:
        """Simulate VIX data when yfinance unavailable."""
        dates = pd.date_range(end=datetime.now(), periods=lookback_days, freq='D')
        
        # Simulate mean-reverting VIX around 18
        np.random.seed(42)
        vix_values = 18 + np.cumsum(np.random.randn(lookback_days) * 0.5)
        vix_values = np.clip(vix_values, 10, 50)
        
        return pd.DataFrame({
            'Close': vix_values,
            'Open': vix_values * (1 + np.random.randn(lookback_days) * 0.02),
            'High': vix_values * 1.05,
            'Low': vix_values * 0.95
        }, index=dates)
    
    def get_current_state(self, force_refresh: bool = False) -> VIXState:
        """
        Get current VIX state with regime classification.
        
        Args:
            force_refresh: Force refresh even if cache valid
        
        Returns:
            VIXState with current metrics
        """
        # Check cache
        if not force_refresh and self._current_state:
            if self._last_fetch:
                cache_age = (datetime.now() - self._last_fetch).total_seconds() / 60
                if cache_age < self.cache_minutes:
                    return self._current_state
        
        # Fetch fresh data
        data = self.fetch_vix_data()
        
        if data is None or len(data) < 5:
            return self._default_state()
        
        # Current VIX
        current_vix = float(data['Close'].iloc[-1])
        
        # Changes
        vix_change_1d = 0.0
        vix_change_5d = 0.0
        if len(data) >= 2:
            vix_change_1d = (current_vix / data['Close'].iloc[-2]) - 1
        if len(data) >= 6:
            vix_change_5d = (current_vix / data['Close'].iloc[-6]) - 1
        
        # Percentile
        vix_percentile = self._calculate_percentile(current_vix, data['Close'])
        
        # Regime
        regime = self._classify_regime(current_vix)
        
        # Risk multiplier (with spike adjustment)
        risk_mult = self._calculate_risk_multiplier(current_vix, vix_change_1d, regime)
        
        state = VIXState(
            current_vix=current_vix,
            vix_percentile=vix_percentile,
            vix_change_1d=vix_change_1d,
            vix_change_5d=vix_change_5d,
            regime=regime,
            risk_multiplier=risk_mult,
            timestamp=datetime.now()
        )
        
        self._current_state = state
        
        # Log regime changes
        logger.info(f"VIX: {current_vix:.1f} ({regime.value}), Risk mult: {risk_mult:.2f}")
        
        return state
    
    def _classify_regime(self, vix: float) -> VIXRegime:
        """
        Classify VIX regime using Z-scores (adaptive) or static fallback.
        """
        # Try dynamic Z-score classification
        if self._vix_cache is not None and len(self._vix_cache) > 60:
            try:
                # Calculate 60-day rolling stats
                history = self._vix_cache['Close'].tail(65)
                rolling_mean = history.rolling(window=60).mean().iloc[-1]
                rolling_std = history.rolling(window=60).std().iloc[-1]
                
                if rolling_std > 0:
                    z_score = (vix - rolling_mean) / rolling_std
                    
                    for regime, (low, high) in self.Z_SCORE_THRESHOLDS.items():
                        if low <= z_score < high:
                             # Log occasionally
                             if np.random.random() < 0.05:
                                 logger.debug(f"VIX Dynamic: Z={z_score:.2f} (Regime={regime.name})")
                             return regime
            except Exception as e:
                logger.debug(f"Dynamic regime calc failed: {e}")

        # Fallback to static
        for regime, (low, high) in self.STATIC_THRESHOLDS.items():
            if low <= vix < high:
                return regime
        return VIXRegime.PANIC
    
    def _calculate_percentile(self, current: float, history: pd.Series) -> float:
        """Calculate percentile rank of current VIX."""
        return float((history < current).mean() * 100)
    
    def _calculate_risk_multiplier(
        self,
        vix: float,
        vix_change_1d: float,
        regime: VIXRegime
    ) -> float:
        """
        Calculate risk multiplier with spike adjustment.
        
        Large VIX spikes warrant additional risk reduction.
        """
        base_mult = self.RISK_MULTIPLIERS.get(regime, 1.0)
        
        # Additional reduction for VIX spikes
        if vix_change_1d > 0.20:  # 20% spike
            base_mult *= 0.7
            _now = datetime.now()
            if (self._last_vix_spike_log is None or
                    (_now - self._last_vix_spike_log).total_seconds() >= 3600):
                logger.warning(f"VIX spike detected: {vix_change_1d*100:.1f}% - reducing risk")
                self._last_vix_spike_log = _now
        elif vix_change_1d > 0.10:  # 10% spike
            base_mult *= 0.85
        
        return float(max(0.1, base_mult))
    
    def _default_state(self) -> VIXState:
        """Return default state when data unavailable."""
        return VIXState(
            current_vix=18.0,
            vix_percentile=50.0,
            vix_change_1d=0.0,
            vix_change_5d=0.0,
            regime=VIXRegime.NORMAL,
            risk_multiplier=1.0,
            timestamp=datetime.now()
        )
    
    def should_hedge(self) -> Tuple[bool, str]:
        """
        Determine if portfolio should be hedged.
        
        Returns:
            Tuple of (should_hedge, reason)
        """
        state = self.get_current_state()
        
        # Hedge in complacency (cheap insurance)
        if state.regime == VIXRegime.COMPLACENCY:
            return True, f"VIX low at {state.current_vix:.1f} - cheap put protection"
        
        # Don't chase expensive hedges in panic
        if state.regime == VIXRegime.PANIC:
            return False, f"VIX too high at {state.current_vix:.1f} - hedges expensive"
        
        # VIX above 90th percentile but not extreme
        if state.vix_percentile > 90 and state.regime != VIXRegime.PANIC:
            return True, f"VIX at {state.vix_percentile:.0f} percentile"
        
        return False, "Normal conditions"
    
    def get_position_size_adjustment(self, base_size: float) -> float:
        """
        Adjust position size based on VIX regime.
        
        Args:
            base_size: Base position size
        
        Returns:
            Adjusted position size
        """
        state = self.get_current_state()
        return base_size * state.risk_multiplier
    
    def analyze_term_structure(self) -> Dict[str, float]:
        """
        Analyze VIX futures term structure (if data available).
        
        Contango (futures > spot) = normal
        Backwardation (futures < spot) = fear, often precedes rallies
        
        Returns:
            Dict with term structure metrics
        """
        # Would require VIX futures data
        # For now, use VIX spot vs 3-month average as proxy
        
        if self._vix_cache is None or len(self._vix_cache) < 60:
            return {'term_structure': 'unknown', 'contango': 0.0}
        
        current = self._vix_cache['Close'].iloc[-1]
        avg_3m = self._vix_cache['Close'].iloc[-60:].mean()
        
        contango = (avg_3m - current) / current
        
        if contango > 0.1:
            structure = 'backwardation'  # Current > average (fear)
        elif contango < -0.1:
            structure = 'contango'       # Current < average (normal)
        else:
            structure = 'flat'
        
        return {
            'term_structure': structure,
            'contango': float(contango),
            'current_vs_avg': float(current / avg_3m)
        }
    
    def get_report(self) -> Dict:
        """Generate VIX regime report."""
        state = self.get_current_state()
        term = self.analyze_term_structure()
        should_hedge, hedge_reason = self.should_hedge()
        
        return {
            'state': state.to_dict(),
            'term_structure': term,
            'hedging': {
                'should_hedge': should_hedge,
                'reason': hedge_reason
            },
            'recommendations': self._get_recommendations(state)
        }
    
    def _get_recommendations(self, state: VIXState) -> List[str]:
        """Get trading recommendations based on VIX state."""
        recs = []
        
        if state.regime == VIXRegime.COMPLACENCY:
            recs.append("Consider buying put protection (cheap)")
            recs.append("Reduce overall exposure slightly")
            
        elif state.regime == VIXRegime.ELEVATED:
            recs.append("Reduce position sizes by 30%")
            recs.append("Tighten stop losses")
            
        elif state.regime == VIXRegime.FEAR:
            recs.append("Significantly reduce risk exposure")
            recs.append("Consider closing high-beta positions")
            
        elif state.regime == VIXRegime.PANIC:
            recs.append("Minimize new trades")
            recs.append("Focus on capital preservation")
            recs.append("Watch for mean reversion opportunities")

        if state.vix_change_1d > 0.15:
            recs.append("VIX spike detected - extra caution")

        return recs


# ─────────────────────────────────────────────────────────────────────────────
# Process-wide singleton access (Round 7 / GAP-8D)
# ─────────────────────────────────────────────────────────────────────────────
#
# ``VIXRegimeManager`` is stateful (cache + optional yfinance handles), so
# multiple instances would duplicate network calls and diverge on regime
# detection. Expose a thread-safe singleton for callers — chiefly the ML
# signal generator — that only need read-only regime information.

import threading as _threading  # noqa: E402 — late import keeps header tidy

_GLOBAL_VIX_MANAGER: Optional["VIXRegimeManager"] = None
_GLOBAL_VIX_LOCK = _threading.Lock()


def get_global_vix_manager() -> "VIXRegimeManager":
    """
    Return the process-wide :class:`VIXRegimeManager` singleton.

    The instance uses the default cache window (5 minutes) and re-uses
    whichever data source is available. Safe to call from any thread.
    """
    global _GLOBAL_VIX_MANAGER
    if _GLOBAL_VIX_MANAGER is None:
        with _GLOBAL_VIX_LOCK:
            if _GLOBAL_VIX_MANAGER is None:
                _GLOBAL_VIX_MANAGER = VIXRegimeManager()
    return _GLOBAL_VIX_MANAGER


def is_crisis_regime() -> bool:
    """
    Return ``True`` when the current VIX state is classified as PANIC.

    Catches every exception — a crisis check must never take down a caller.
    In the failure path we return ``False`` (fail-open) so a transient data
    outage does not silently halt all ML-driven trading.
    """
    try:
        state = get_global_vix_manager().get_current_state()
    except Exception:  # pragma: no cover — defensive: yfinance etc. failures
        return False
    return state.regime is VIXRegime.PANIC
