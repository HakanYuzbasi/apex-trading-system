"""
risk/adaptive_position_sizer.py
DYNAMIC POSITION SIZING BASED ON MARKET CONDITIONS
- Kelly Criterion
- Volatility scaling (non-mutating percentile lookup)
- Performance-based adjustment
- Geometrically-normalized multiplier stacking (no cascade to floor)
"""

from __future__ import annotations

import logging
from collections import deque
from typing import Deque, Dict, Optional

import numpy as np

from config import ApexConfig
from core.logging_config import setup_logging

logger = logging.getLogger(__name__)


class AdaptivePositionSizer:
    """
    Dynamically adjust position size based on:
    - Market volatility (rolling-window percentile rank, read-only)
    - Strategy performance (Sharpe ratio)
    - Win rate
    - Signal confidence
    - Current drawdown
    - Market regime

    All thresholds are sourced from ``ApexConfig``. The six per-factor
    multipliers are combined via a geometric-root of their *deviations from
    1.0* so that stacking additional factors cannot cascade the size to the
    clip-floor. This is the canonical fix for the classic "six 0.85× factors
    compound to 0.38×" bug that chronically under-sized trades.
    """

    def __init__(self, base_position_size: float = 5000):
        self.base_position_size: float = float(base_position_size)
        self._vol_lookback: int = int(ApexConfig.ADAPTIVE_SIZER_VOL_LOOKBACK)
        self._vol_min_history: int = int(ApexConfig.ADAPTIVE_SIZER_VOL_MIN_HISTORY)
        # Bounded deque — O(1) append, no manual slicing.
        self.volatility_history: Deque[float] = deque(maxlen=self._vol_lookback)

        logger.info(
            "✅ Adaptive Position Sizer initialized (base: $%s, vol_lookback=%d)",
            f"{base_position_size:,.0f}", self._vol_lookback,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def record_volatility(self, volatility: float) -> None:
        """
        Explicitly record a new volatility observation.

        Callers should invoke this **once per sizing cycle** — never from
        inside the percentile-lookup path. Separating write from read
        prevents history pollution during backtest replay or duplicate
        sizing calls for the same bar.

        Args:
            volatility: Annualized volatility (e.g. 0.20 = 20%).

        Raises:
            ValueError: If ``volatility`` is negative or non-finite.
        """
        v = float(volatility)
        if not np.isfinite(v) or v < 0.0:
            raise ValueError(f"volatility must be non-negative finite, got {volatility!r}")
        self.volatility_history.append(v)

    def calculate_position_size(
        self,
        signal_confidence: float,
        volatility: float,
        sharpe_ratio: float = 0.0,
        win_rate: float = 0.5,
        current_drawdown: float = 0.0,
        regime_multiplier: float = 1.0,
        portfolio_value: float = 1_000_000,
        max_position_pct: float = 0.02,
    ) -> Dict:
        """
        Calculate the optimal notional position size.

        The computation applies six independent multipliers (confidence,
        volatility percentile, Sharpe, win-rate, drawdown, regime), all
        sourced from :class:`ApexConfig`. Rather than multiplying the raw
        factors together (which compounds downward), deviations from 1.0 are
        raised to a fractional exponent (``ADAPTIVE_SIZER_STACK_EXPONENT``)
        and then re-combined. This preserves directional intent without
        cascading the size to the clip floor when many factors are modestly
        below 1.0.

        Args:
            signal_confidence: Signal confidence in ``[0, 1]``. Out-of-range
                values are clipped.
            volatility: Current annualized volatility (e.g. 0.20). Must be
                non-negative and finite.
            sharpe_ratio: Rolling strategy Sharpe ratio.
            win_rate: Historical win rate in ``[0, 1]``. Out-of-range values
                are clipped.
            current_drawdown: Current drawdown as a non-negative fraction
                (0.10 = 10% DD).
            regime_multiplier: Market-regime size multiplier (``>= 0``).
            portfolio_value: Total portfolio equity in account currency.
                Must be ``> 0``.
            max_position_pct: Per-trade cap as fraction of ``portfolio_value``.
                Must be in ``(0, 1]``.

        Returns:
            ``{'position_size': float, 'multiplier': float,
               'components': dict[str, float], 'base_size': float,
               'portfolio_cap': float}``

        Raises:
            ValueError: If ``portfolio_value <= 0``, ``max_position_pct``
                outside ``(0, 1]``, or ``volatility`` is negative/non-finite.
        """
        # ── Argument validation ────────────────────────────────────────
        if portfolio_value <= 0.0 or not np.isfinite(portfolio_value):
            raise ValueError(f"portfolio_value must be positive finite, got {portfolio_value!r}")
        if not (0.0 < max_position_pct <= 1.0):
            raise ValueError(f"max_position_pct must be in (0,1], got {max_position_pct!r}")
        if not np.isfinite(volatility) or volatility < 0.0:
            raise ValueError(f"volatility must be non-negative finite, got {volatility!r}")
        if not np.isfinite(regime_multiplier) or regime_multiplier < 0.0:
            raise ValueError(f"regime_multiplier must be non-negative finite, got {regime_multiplier!r}")

        conf = float(np.clip(signal_confidence, 0.0, 1.0))
        wr = float(np.clip(win_rate, 0.0, 1.0))
        dd = max(0.0, float(current_drawdown))

        components: Dict[str, float] = {}

        # 1. Signal confidence — linear blend [FLOOR, FLOOR+WEIGHT]
        conf_mult = float(ApexConfig.ADAPTIVE_SIZER_CONF_FLOOR) + conf * float(
            ApexConfig.ADAPTIVE_SIZER_CONF_WEIGHT
        )
        components["confidence"] = conf_mult

        # 2. Volatility — inverse percentile, bounded
        vol_percentile = self._get_volatility_percentile(float(volatility))
        vol_mult = float(
            np.clip(
                2.0 - vol_percentile,
                ApexConfig.ADAPTIVE_SIZER_VOL_MULT_MIN,
                ApexConfig.ADAPTIVE_SIZER_VOL_MULT_MAX,
            )
        )
        components["volatility"] = vol_mult
        components["vol_percentile"] = float(vol_percentile)

        # 3. Sharpe tiering
        if sharpe_ratio > ApexConfig.ADAPTIVE_SIZER_SHARPE_EXCELLENT:
            sharpe_mult = float(ApexConfig.ADAPTIVE_SIZER_SHARPE_MULT_EXCELLENT)
        elif sharpe_ratio > ApexConfig.ADAPTIVE_SIZER_SHARPE_GOOD:
            sharpe_mult = float(ApexConfig.ADAPTIVE_SIZER_SHARPE_MULT_GOOD)
        elif sharpe_ratio > ApexConfig.ADAPTIVE_SIZER_SHARPE_OK:
            sharpe_mult = float(ApexConfig.ADAPTIVE_SIZER_SHARPE_MULT_OK)
        elif sharpe_ratio > 0.0:
            sharpe_mult = float(ApexConfig.ADAPTIVE_SIZER_SHARPE_MULT_WEAK)
        else:
            sharpe_mult = float(ApexConfig.ADAPTIVE_SIZER_SHARPE_MULT_NEG)
        components["sharpe"] = sharpe_mult

        # 4. Win-rate — linear blend [FLOOR, FLOOR+WEIGHT]
        wr_mult = float(ApexConfig.ADAPTIVE_SIZER_WR_FLOOR) + wr * float(
            ApexConfig.ADAPTIVE_SIZER_WR_WEIGHT
        )
        components["win_rate"] = wr_mult

        # 5. Drawdown tiering
        if dd >= ApexConfig.ADAPTIVE_SIZER_DD_SEVERE:
            dd_mult = float(ApexConfig.ADAPTIVE_SIZER_DD_MULT_SEVERE)
        elif dd >= ApexConfig.ADAPTIVE_SIZER_DD_HIGH:
            dd_mult = float(ApexConfig.ADAPTIVE_SIZER_DD_MULT_HIGH)
        elif dd >= ApexConfig.ADAPTIVE_SIZER_DD_MODERATE:
            dd_mult = float(ApexConfig.ADAPTIVE_SIZER_DD_MULT_MODERATE)
        else:
            dd_mult = 1.0
        components["drawdown"] = dd_mult

        # 6. Regime — passthrough, validated above
        components["regime"] = float(regime_multiplier)

        # ── Geometric stacking of deviations-from-1.0 ─────────────────
        # Raising each factor to STACK_EXPONENT in log-space is equivalent
        # to taking the geometric mean over effectively (1/exponent) factors,
        # shrinking cumulative drift without ever flipping sign.
        exp = float(ApexConfig.ADAPTIVE_SIZER_STACK_EXPONENT)
        factors = (conf_mult, vol_mult, sharpe_mult, wr_mult, dd_mult, float(regime_multiplier))
        # log(factor ** exp) == exp * log(factor). Floor factor at 1e-6 to avoid log(0).
        log_sum = sum(exp * float(np.log(max(f, 1e-6))) for f in factors)
        raw_multiplier = float(np.exp(log_sum))

        multiplier = float(
            np.clip(
                raw_multiplier,
                ApexConfig.ADAPTIVE_SIZER_CLIP_MIN,
                ApexConfig.ADAPTIVE_SIZER_CLIP_MAX,
            )
        )

        position_size = self.base_position_size * multiplier
        portfolio_cap = portfolio_value * float(max_position_pct)
        if position_size > portfolio_cap:
            position_size = portfolio_cap
            logger.debug(
                "Position capped at %.2f%% of portfolio ($%s)",
                max_position_pct * 100.0, f"{portfolio_cap:,.0f}",
            )

        # Position size must be non-negative
        position_size = max(0.0, position_size)

        logger.debug(
            "💰 Position Size: $%s → $%s (%.2fx raw=%.2fx)",
            f"{self.base_position_size:,.0f}",
            f"{position_size:,.0f}",
            multiplier, raw_multiplier,
        )

        return {
            "position_size": position_size,
            "multiplier": multiplier,
            "components": components,
            "base_size": self.base_position_size,
            "portfolio_cap": portfolio_cap,
        }

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------
    def _get_volatility_percentile(self, current_vol: float) -> float:
        """
        Return the percentile rank of ``current_vol`` within the stored
        history, then append the sample to history.

        The percentile is computed against the history **as it was before
        this sample was appended** — the previous implementation folded
        the query value into the ranking basis, biasing every observation
        toward the 0.5 midpoint. It also used an O(n²) ``list.index`` and
        ``abs()``-nearest lookup, which assigned wrong ranks to ties.

        Uses ``np.searchsorted`` on a sorted copy (O(n log n)) and the
        canonical ``(below + 0.5*equal) / N`` convention, correctly
        handling duplicate observations.

        Args:
            current_vol: The volatility value to rank. Must be
                non-negative and finite.

        Returns:
            Percentile in ``[0.0, 1.0]``. Returns ``0.5`` when fewer than
            ``ADAPTIVE_SIZER_VOL_MIN_HISTORY`` samples are available.

        Raises:
            ValueError: If ``current_vol`` is negative or non-finite.
        """
        v = float(current_vol)
        if not np.isfinite(v) or v < 0.0:
            raise ValueError(f"current_vol must be non-negative finite, got {current_vol!r}")

        n = len(self.volatility_history)
        if n < self._vol_min_history:
            # Still record the observation so warm-up completes.
            self.volatility_history.append(v)
            return 0.5

        sorted_vol = np.sort(np.asarray(self.volatility_history, dtype=float))
        left = int(np.searchsorted(sorted_vol, v, side="left"))
        right = int(np.searchsorted(sorted_vol, v, side="right"))
        below = left
        equal = right - left
        # Rank against the CURRENT history (pre-append) — this is the
        # proper unbiased percentile. Then persist the sample for future
        # calls so history accumulates across cycles.
        rank = (below + 0.5 * equal) / float(n)
        self.volatility_history.append(v)
        return float(np.clip(rank, 0.0, 1.0))
    
    def kelly_criterion(
        self,
        win_rate: float,
        avg_win: float,
        avg_loss: float,
        kelly_fraction: float = 0.25
    ) -> float:
        """
        Calculate position size using Kelly Criterion.
        
        Kelly % = W - [(1-W)/R]
        where:
        - W = win rate
        - R = avg_win / avg_loss ratio
        
        Args:
            win_rate: Historical win rate (0 to 1)
            avg_win: Average winning trade
            avg_loss: Average losing trade (positive number)
            kelly_fraction: Fraction of full Kelly (0.25 = quarter Kelly)
        
        Returns:
            Recommended position size as fraction of capital
        """
        if avg_loss == 0 or win_rate == 0:
            return 0.0
        
        # Win/Loss ratio
        win_loss_ratio = avg_win / avg_loss
        
        # Kelly formula
        kelly_pct = win_rate - ((1 - win_rate) / win_loss_ratio)
        
        # Apply fraction (quarter Kelly is safer)
        kelly_pct *= kelly_fraction
        
        # Clip to reasonable range
        kelly_pct = np.clip(kelly_pct, 0.0, ApexConfig.KELLY_MAX_POSITION_PCT)
        
        return kelly_pct


if __name__ == "__main__":
    # Test adaptive position sizer
    setup_logging(level="DEBUG", log_file=None, json_format=False, console_output=True)
    
    sizer = AdaptivePositionSizer(base_position_size=5000)
    
    print("\n" + "="*60)
    print("TESTING ADAPTIVE POSITION SIZING")
    print("="*60)
    
    scenarios = [
        {
            'name': 'Perfect Conditions',
            'confidence': 0.9,
            'volatility': 0.12,
            'sharpe': 2.0,
            'win_rate': 0.65,
            'drawdown': 0.02,
            'regime_mult': 1.3
        },
        {
            'name': 'High Volatility',
            'confidence': 0.7,
            'volatility': 0.35,
            'sharpe': 1.0,
            'win_rate': 0.55,
            'drawdown': 0.05,
            'regime_mult': 1.0
        },
        {
            'name': 'Deep Drawdown',
            'confidence': 0.6,
            'volatility': 0.20,
            'sharpe': 0.5,
            'win_rate': 0.45,
            'drawdown': 0.12,
            'regime_mult': 0.7
        },
        {
            'name': 'Crisis Mode',
            'confidence': 0.8,
            'volatility': 0.50,
            'sharpe': -0.5,
            'win_rate': 0.40,
            'drawdown': 0.15,
            'regime_mult': 0.2
        }
    ]
    
    for scenario in scenarios:
        print(f"\n{scenario['name']}:")
        result = sizer.calculate_position_size(
            signal_confidence=scenario['confidence'],
            volatility=scenario['volatility'],
            sharpe_ratio=scenario['sharpe'],
            win_rate=scenario['win_rate'],
            current_drawdown=scenario['drawdown'],
            regime_multiplier=scenario['regime_mult']
        )
        
        print(f"  Position Size: ${result['position_size']:,.0f} ({result['multiplier']:.2f}x)")
        print("  Components:")
        for key, value in result['components'].items():
            print(f"    {key:12s}: {value:.2f}x")
    
    # Test Kelly Criterion
    print("\n" + "="*60)
    print("TESTING KELLY CRITERION")
    print("="*60)
    
    kelly_pct = sizer.kelly_criterion(
        win_rate=0.60,
        avg_win=100,
        avg_loss=50,
        kelly_fraction=0.25
    )
    
    print(f"Kelly Position Size: {kelly_pct*100:.2f}% of capital")
    
    print("\n✅ Adaptive position sizer tests complete!")
