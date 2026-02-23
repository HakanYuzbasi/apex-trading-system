"""
risk/correlation_cascade_breaker.py - Portfolio-Wide Correlation Shield

Continuously monitors pairwise correlations across the entire portfolio.
Detects when diversification collapses (all positions move together) and
auto-deleverages to prevent cascading losses.

Correlation regimes:
NORMAL   (avg < 0.40) → Full trading
ELEVATED (0.40-0.60)  → Block new correlated entries
HERDING  (0.60-0.80)  → Reduce positions to 70% of normal
CRISIS   (> 0.80)     → Reduce to 50%, close most-correlated pair

Effective positions formula:
    effective_N = N² / sum(all_pairwise_correlations)
    If 10 positions with avg corr 0.85 → effective_N ≈ 2.4
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime
from enum import IntEnum
import logging

logger = logging.getLogger(__name__)


class CorrelationRegime(IntEnum):
    NORMAL = 0
    ELEVATED = 1
    HERDING = 2
    CRISIS = 3


@dataclass
class CorrelationState:
    """Current portfolio correlation assessment."""
    regime: CorrelationRegime
    avg_correlation: float
    max_pairwise: float
    effective_positions: float
    concentration_risk: float  # 0-1, higher = more concentrated
    most_correlated_pair: Optional[Tuple[str, str]] = None
    timestamp: datetime = field(default_factory=datetime.now)


class CorrelationCascadeBreaker:
    """
    Portfolio-wide correlation monitoring with automatic deleveraging.

    Computes rolling pairwise correlations across all positions and detects
    when the portfolio transitions into herding/crisis regimes where
    diversification fails.
    """

    
    def _get_dynamic_thresholds(self, vix_level: float = None):
        from config import ApexConfig
        if not getattr(ApexConfig, 'CORRELATION_DYNAMIC_ENABLED', False) or vix_level is None:
            return (0.40, 0.60, 0.80)  # Fallback to static
            
        if vix_level > 30:
            return (0.35, 0.55, 0.75)  # High VIX: tighten
        elif vix_level > 20:
            return (0.50, 0.70, 0.85)  # Normal
        else:
            return (0.60, 0.80, 0.90)  # Low VIX: relax

    def __init__(
        self,
        elevated_threshold: float = 0.40,
        herding_threshold: float = 0.60,
        crisis_threshold: float = 0.80,
        lookback_days: int = 20,
        min_positions_for_check: int = 3,
    ):
        self.elevated_threshold = elevated_threshold
        self.herding_threshold = herding_threshold
        self.crisis_threshold = crisis_threshold
        self.lookback_days = lookback_days
        self.min_positions = min_positions_for_check

        self._last_state: Optional[CorrelationState] = None

        logger.info(
            f"CorrelationCascadeBreaker initialized: "
            f"elevated={elevated_threshold}, herding={herding_threshold}, "
            f"crisis={crisis_threshold}"
        )

    def assess_correlation_state(
        self,
        positions: List[str],
        historical_data: Dict[str, pd.DataFrame],
    ) -> CorrelationState:
        """
        Assess portfolio-wide correlation state.

        Args:
            positions: List of currently held symbols
            historical_data: Dict of symbol -> DataFrame with 'Close' column

        Returns:
            CorrelationState with regime classification
        """
        if len(positions) < self.min_positions:
            state = CorrelationState(
                regime=CorrelationRegime.NORMAL,
                avg_correlation=0.0,
                max_pairwise=0.0,
                effective_positions=float(len(positions)),
                concentration_risk=0.0,
            )
            self._last_state = state
            return state

        # Build return matrix
        returns_dict = {}
        for sym in positions:
            df = historical_data.get(sym)
            if df is None or not isinstance(df, pd.DataFrame):
                continue

            close_col = "Close" if "Close" in df.columns else "close"
            if close_col not in df.columns:
                continue

            close = df[close_col].tail(self.lookback_days + 1)
            if len(close) < 5:
                continue

            ret = close.pct_change().dropna()
            if len(ret) >= 3:
                returns_dict[sym] = ret.values[-min(self.lookback_days, len(ret)):]

        if len(returns_dict) < self.min_positions:
            state = CorrelationState(
                regime=CorrelationRegime.NORMAL,
                avg_correlation=0.0,
                max_pairwise=0.0,
                effective_positions=float(len(positions)),
                concentration_risk=0.0,
            )
            self._last_state = state
            return state

        # Compute pairwise correlations
        symbols = list(returns_dict.keys())
        n = len(symbols)
        # Align lengths
        min_len = min(len(v) for v in returns_dict.values())
        aligned = {s: v[-min_len:] for s, v in returns_dict.items()}

        corr_matrix = np.corrcoef([aligned[s] for s in symbols])
        # Handle NaN from constant series
        corr_matrix = np.nan_to_num(corr_matrix, nan=0.0)

        # Extract upper triangle (pairwise)
        pairwise = []
        max_corr = 0.0
        max_pair = None
        for i in range(n):
            for j in range(i + 1, n):
                c = abs(corr_matrix[i, j])
                pairwise.append(c)
                if c > max_corr:
                    max_corr = c
                    max_pair = (symbols[i], symbols[j])

        avg_corr = float(np.mean(pairwise)) if pairwise else 0.0
        max_pairwise = float(max_corr)

        # Effective positions: N² / sum(correlation_matrix)
        total_corr = float(np.sum(np.abs(corr_matrix)))
        effective_n = (n * n) / max(total_corr, 1.0)

        # Concentration risk: 1 - (effective_N / actual_N)
        concentration = max(0.0, 1.0 - effective_n / max(n, 1))

        # Determine regime
        if avg_corr >= self.crisis_threshold:
            regime = CorrelationRegime.CRISIS
        elif avg_corr >= self.herding_threshold:
            regime = CorrelationRegime.HERDING
        elif avg_corr >= self.elevated_threshold:
            regime = CorrelationRegime.ELEVATED
        else:
            regime = CorrelationRegime.NORMAL

        state = CorrelationState(
            regime=regime,
            avg_correlation=avg_corr,
            max_pairwise=max_pairwise,
            effective_positions=effective_n,
            concentration_risk=concentration,
            most_correlated_pair=max_pair,
        )
        self._last_state = state

        if regime >= CorrelationRegime.HERDING:
            logger.warning(
                f"Correlation regime: {regime.name} "
                f"(avg={avg_corr:.2f}, max_pair={max_pair}, "
                f"effective_N={effective_n:.1f})"
            )

        return state

    def get_max_position_count(self, state: Optional[CorrelationState] = None) -> int:
        """Get maximum recommended position count for current regime."""
        state = state or self._last_state
        if state is None:
            return 20  # default max

        base_max = 20
        return {
            CorrelationRegime.NORMAL: base_max,
            CorrelationRegime.ELEVATED: int(base_max * 0.8),
            CorrelationRegime.HERDING: int(base_max * 0.6),
            CorrelationRegime.CRISIS: int(base_max * 0.4),
        }[state.regime]

    def get_positions_to_reduce(
        self,
        positions: List[str],
        historical_data: Dict[str, pd.DataFrame],
    ) -> List[Tuple[str, float]]:
        """
        Get positions that should be reduced due to high correlation.

        Returns list of (symbol, reduction_fraction) sorted by urgency.
        """
        state = self._last_state
        if state is None or state.regime < CorrelationRegime.HERDING:
            return []

        if not positions or len(positions) < 2:
            return []

        # For CRISIS: reduce the most-correlated pair
        reductions = []
        if state.most_correlated_pair:
            sym1, sym2 = state.most_correlated_pair
            if state.regime == CorrelationRegime.CRISIS:
                reductions.append((sym1, 0.5))
                reductions.append((sym2, 0.5))
            elif state.regime == CorrelationRegime.HERDING:
                reductions.append((sym1, 0.3))
                reductions.append((sym2, 0.3))

        return reductions

    def should_block_entry(
        self,
        symbol: str,
        existing_positions: List[str],
        historical_data: Dict[str, pd.DataFrame],
    ) -> bool:
        """
        Check if a new entry should be blocked due to correlation.

        Blocks if:
        - Portfolio already in HERDING/CRISIS regime
        - New symbol is highly correlated with existing positions
        """
        state = self._last_state

        if state is not None and state.regime >= CorrelationRegime.HERDING:
            return True

        if state is not None and state.regime >= CorrelationRegime.ELEVATED:
            # Check if new symbol would increase concentration
            if len(existing_positions) >= self.get_max_position_count(state):
                return True

        return False

    def get_effective_diversification(self) -> float:
        """
        Get 0-1 diversification score.
        1.0 = perfectly diversified, 0.0 = all positions identical.
        """
        if self._last_state is None:
            return 1.0
        return max(0.0, 1.0 - self._last_state.concentration_risk)

    def get_diagnostics(self) -> Dict:
        """Return breaker state for monitoring."""
        if self._last_state is None:
            return {"state": "not_initialized"}

        s = self._last_state
        return {
            "regime": s.regime.name,
            "avg_correlation": round(s.avg_correlation, 3),
            "max_pairwise": round(s.max_pairwise, 3),
            "effective_positions": round(s.effective_positions, 1),
            "concentration_risk": round(s.concentration_risk, 3),
            "most_correlated_pair": s.most_correlated_pair,
            "diversification_score": round(self.get_effective_diversification(), 3),
        }
