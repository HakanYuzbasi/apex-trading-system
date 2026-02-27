"""
models/adaptive_regime_detector.py - Probability-Based Regime Detection

Replaces simplistic MA-crossover + volatility threshold regime detection
with multi-indicator weighted voting, EMA-smoothed probability distributions,
and fine-grained sub-regime classification.

Key improvements over the old RegimeDetector:
- 8 independent indicators vote on regime probabilities
- Smooth transitions via EMA smoothing (no hard cutoffs)
- Sub-regime detection (early_bull, late_bull, capitulation, etc.)
- Leading indicators for early regime change detection
- Per-symbol regime overlay
"""

import numpy as np
import pandas as pd
from typing import Dict, Optional
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from collections import deque
import logging

logger = logging.getLogger(__name__)

REGIMES = ["bull", "bear", "neutral", "volatile"]


class SubRegime(Enum):
    EARLY_BULL = "early_bull"
    MID_BULL = "mid_bull"
    LATE_BULL = "late_bull"
    EARLY_BEAR = "early_bear"
    MID_BEAR = "mid_bear"
    CAPITULATION = "capitulation"
    RECOVERY = "recovery"
    RANGE_BOUND = "range_bound"
    VOLATILE_UP = "volatile_up"
    VOLATILE_DOWN = "volatile_down"


@dataclass
class RegimeAssessment:
    """Probability-based regime output."""
    primary_regime: str
    sub_regime: SubRegime
    regime_probabilities: Dict[str, float]
    transition_probability: float
    regime_age_days: int
    regime_strength: float
    leading_indicators: Dict[str, float]
    timestamp: datetime = field(default_factory=datetime.now)


class AdaptiveRegimeDetector:
    """
    Multi-indicator regime detection with smooth probability transitions.

    Uses 8 independent indicators that each vote on regime probabilities.
    Votes are combined via weighted average and EMA-smoothed to prevent
    whipsaw transitions.
    """

    # Indicator weights for combining votes
    INDICATOR_WEIGHTS = {
        "trend_ma": 0.20,
        "trend_slope": 0.10,
        "volatility": 0.20,
        "vol_clustering": 0.10,
        "breadth": 0.15,
        "momentum_diffusion": 0.10,
        "vix": 0.10,
        "correlation": 0.05,
    }

    def __init__(
        self,
        smoothing_alpha: float = 0.15,
        min_regime_duration: int = 3,
        min_transition_gap: float = 0.02,
        transition_cooldown_steps: int = 3,
    ):
        self.smoothing_alpha = smoothing_alpha
        self.min_regime_duration = min_regime_duration
        self.min_transition_gap = max(0.0, float(min_transition_gap))
        self.transition_cooldown_steps = max(1, int(transition_cooldown_steps))

        # State tracking
        self._smoothed_probs: Dict[str, float] = {r: 0.25 for r in REGIMES}
        self._current_regime: str = "neutral"
        self._regime_start: datetime = datetime.now()
        self._regime_age_days: int = 0
        self._step_counter: int = 0
        self._last_transition_step: int = 0  # Replaced -10_000 initialization
        self._history: deque = deque(maxlen=500)
        self._previous_assessment: Optional[RegimeAssessment] = None

        self._load_state()

        logger.info(
            f"AdaptiveRegimeDetector initialized: alpha={smoothing_alpha}, "
            f"min_duration={min_regime_duration}"
        )

    def _load_state(self):
        import json
        from config import ApexConfig
        try:
            state_file = ApexConfig.DATA_DIR / "regime_state.json"
            if state_file.exists():
                with open(state_file, 'r') as f:
                    data = json.load(f)
                    self._current_regime = data.get("current_regime", "neutral")
                    self._step_counter = data.get("step_counter", 0)
                    self._last_transition_step = data.get("last_transition_step", 0)
                    logger.info(f"Loaded regime state: {self._current_regime} at step {self._step_counter}")
        except Exception as e:
            logger.warning(f"Failed to load regime state: {e}")

    def save_state(self):
        import json
        from config import ApexConfig
        try:
            state_file = ApexConfig.DATA_DIR / "regime_state.json"
            ApexConfig.DATA_DIR.mkdir(parents=True, exist_ok=True)
            data = {
                "current_regime": self._current_regime,
                "step_counter": self._step_counter,
                "last_transition_step": self._last_transition_step,
            }
            with open(state_file, 'w') as f:
                json.dump(data, f)
        except Exception as e:
            logger.debug(f"Failed to save regime state: {e}")

    def assess_regime(
        self,
        prices: pd.Series,
        universe_data: Optional[Dict[str, pd.DataFrame]] = None,
        vix_level: Optional[float] = None,
        emit_transition_logs: bool = True,
    ) -> RegimeAssessment:
        """
        Assess market regime using multi-indicator voting.

        Args:
            prices: Market index price series (e.g., SPY close prices)
            universe_data: Optional dict of symbol -> DataFrame for breadth calc
            vix_level: Current VIX level if available

        Returns:
            RegimeAssessment with probabilities, sub-regime, and leading indicators
        """
        if len(prices) < 60:
            return self._neutral_assessment()
        self._step_counter += 1

        # Compute all indicator votes
        indicator_scores = self._compute_indicator_scores(
            prices, universe_data, vix_level
        )

        # Combine into raw probabilities
        raw_probs = self._combine_votes(indicator_scores)

        # Smooth with EMA
        smoothed = self._smooth_probabilities(raw_probs)
        self._smoothed_probs = smoothed

        # Determine primary regime
        primary = max(smoothed, key=smoothed.get)

        # Check min duration before switching
        if primary != self._current_regime:
            sorted_probs = sorted(smoothed.items(), key=lambda kv: kv[1], reverse=True)
            prob_gap = (
                float(sorted_probs[0][1] - sorted_probs[1][1])
                if len(sorted_probs) > 1 else float(sorted_probs[0][1])
            )
            cooldown_ready = (self._step_counter - self._last_transition_step) >= self.transition_cooldown_steps
            cooldown_remaining = max(0, self.transition_cooldown_steps - (self._step_counter - self._last_transition_step))
            
            from config import ApexConfig
            margin_req = float(getattr(ApexConfig, "REGIME_MIN_MARGIN", 0.10))
            eff_gap_req = max(self.min_transition_gap, margin_req)

            allowed = (
                self._regime_age_days >= self.min_regime_duration
                and prob_gap >= eff_gap_req
                and cooldown_ready
            )
            
            if emit_transition_logs:
                logger.debug(
                    f"🕵️ Regime Transition Attempt {self._current_regime}->{primary} | step={self._step_counter} | "
                    f"probs={{k: round(v, 2) for k, v in smoothed.items()}} | margin={prob_gap:.2f} (req={eff_gap_req:.2f}) | "
                    f"cooldown_rem={cooldown_remaining} | {'ALLOWED' if allowed else 'BLOCKED'}"
                )

            if allowed:
                self._current_regime = primary
                self._regime_start = datetime.now()
                self._regime_age_days = 0
                self._last_transition_step = self._step_counter
                if emit_transition_logs:
                    logger.info(
                        f"Regime transition: -> {primary} (probs: "
                        f"{', '.join(f'{r}={p:.2f}' for r, p in smoothed.items())})"
                    )
            else:
                primary = self._current_regime

        self._regime_age_days += 1

        # Compute transition probability
        transition_prob = self._compute_transition_probability(smoothed, primary)

        # Compute regime strength
        regime_strength = smoothed.get(primary, 0.25)

        # Detect sub-regime
        sub = self._detect_sub_regime(primary, self._regime_age_days, prices)

        # Leading indicators
        leading = self._compute_leading_indicators(prices, indicator_scores, vix_level)

        assessment = RegimeAssessment(
            primary_regime=primary,
            sub_regime=sub,
            regime_probabilities=smoothed,
            transition_probability=transition_prob,
            regime_age_days=self._regime_age_days,
            regime_strength=regime_strength,
            leading_indicators=leading,
        )

        self._previous_assessment = assessment
        self._history.append(assessment)
        return assessment

    def get_regime_for_symbol(
        self,
        symbol: str,
        symbol_prices: pd.Series,
        market_regime: RegimeAssessment,
    ) -> RegimeAssessment:
        """
        Get symbol-specific regime overlay.

        A symbol may be in its own bear market while the broad market is bull.
        """
        if len(symbol_prices) < 60:
            return market_regime

        # Simple symbol-specific trend check
        returns = symbol_prices.pct_change().dropna()
        if len(returns) < 20:
            return market_regime

        ret_20 = returns.tail(20).mean() * 252
        vol_20 = returns.tail(20).std() * np.sqrt(252)

        # If symbol diverges strongly from market, create overlay
        if vol_20 > 0.50 and market_regime.primary_regime != "volatile":
            probs = dict(market_regime.regime_probabilities)
            probs["volatile"] = min(1.0, probs.get("volatile", 0) + 0.3)
            total = sum(probs.values())
            probs = {k: v / total for k, v in probs.items()}
            return RegimeAssessment(
                primary_regime="volatile" if probs["volatile"] > 0.4 else market_regime.primary_regime,
                sub_regime=SubRegime.VOLATILE_DOWN if ret_20 < 0 else SubRegime.VOLATILE_UP,
                regime_probabilities=probs,
                transition_probability=market_regime.transition_probability,
                regime_age_days=market_regime.regime_age_days,
                regime_strength=probs.get(max(probs, key=probs.get), 0.25),
                leading_indicators=market_regime.leading_indicators,
            )

        if ret_20 < -0.15 and market_regime.primary_regime in ("bull", "neutral"):
            probs = dict(market_regime.regime_probabilities)
            probs["bear"] = min(1.0, probs.get("bear", 0) + 0.25)
            total = sum(probs.values())
            probs = {k: v / total for k, v in probs.items()}
            primary = "bear" if probs["bear"] > 0.4 else market_regime.primary_regime
            return RegimeAssessment(
                primary_regime=primary,
                sub_regime=SubRegime.EARLY_BEAR,
                regime_probabilities=probs,
                transition_probability=market_regime.transition_probability,
                regime_age_days=market_regime.regime_age_days,
                regime_strength=probs.get(primary, 0.25),
                leading_indicators=market_regime.leading_indicators,
            )

        return market_regime

    def classify_history(
        self,
        prices: pd.Series,
        vix_series: Optional[pd.Series] = None
    ) -> pd.Series:
        """
        Classify regime for an entire price history.
        Non-vectorized (uses internal state) but reliable for training data generation.
        """
        regimes = pd.Series(index=prices.index, dtype=str)
        
        # Reset state for fresh run
        self._smoothed_probs = {r: 0.25 for r in REGIMES}
        self._current_regime = "neutral"
        self._regime_age_days = 0
        
        # Process loops (indicators need rolling windows)
        for i in range(len(prices)):
            if i < 60:
                regimes.iloc[i] = "neutral"
                continue
                
            p_slice = prices.iloc[:i+1]
            v_val = vix_series.iloc[i] if (vix_series is not None and i < len(vix_series)) else None
            
            # Use assess_regime to update internal state and get prediction
            assessment = self.assess_regime(
                p_slice,
                vix_level=v_val,
                emit_transition_logs=False,
            )
            regimes.iloc[i] = assessment.primary_regime
            
        return regimes

    # ─── Indicator Computation ─────────────────────────────────────

    def _compute_indicator_scores(
        self,
        prices: pd.Series,
        universe_data: Optional[Dict[str, pd.DataFrame]],
        vix_level: Optional[float],
    ) -> Dict[str, Dict[str, float]]:
        """Compute probability votes from each indicator."""
        scores = {}

        returns = prices.pct_change().dropna()

        # 1. Trend MA crossovers
        scores["trend_ma"] = self._indicator_trend_ma(prices)

        # 2. Trend slope (linear regression)
        scores["trend_slope"] = self._indicator_trend_slope(prices)

        # 3. Realized volatility
        scores["volatility"] = self._indicator_volatility(returns)

        # 4. Volatility clustering
        scores["vol_clustering"] = self._indicator_vol_clustering(returns)

        # 5. Market breadth
        scores["breadth"] = self._indicator_breadth(universe_data)

        # 6. Momentum diffusion
        scores["momentum_diffusion"] = self._indicator_momentum_diffusion(returns)

        # 7. VIX
        scores["vix"] = self._indicator_vix(vix_level)

        # 8. Correlation
        scores["correlation"] = self._indicator_correlation(universe_data)

        return scores

    def _indicator_trend_ma(self, prices: pd.Series) -> Dict[str, float]:
        """MA crossover indicator: multiple timeframe crosses."""
        ma5 = prices.rolling(5).mean().iloc[-1]
        ma20 = prices.rolling(20).mean().iloc[-1]
        ma50 = prices.rolling(50).mean().iloc[-1]
        ma200 = prices.rolling(200).mean().iloc[-1] if len(prices) >= 200 else ma50
        current = prices.iloc[-1]

        # Score based on alignment
        bull_score = 0.0
        if current > ma20:
            bull_score += 0.25
        if ma5 > ma20:
            bull_score += 0.25
        if ma20 > ma50:
            bull_score += 0.25
        if ma50 > ma200:
            bull_score += 0.25

        bear_score = 1.0 - bull_score

        # Neutral when signals are mixed
        neutral_score = 1.0 - abs(bull_score - bear_score)

        total = bull_score + bear_score + neutral_score + 0.1  # small vol base
        return {
            "bull": bull_score / total,
            "bear": bear_score / total,
            "neutral": neutral_score / total,
            "volatile": 0.1 / total,
        }

    def _indicator_trend_slope(self, prices: pd.Series) -> Dict[str, float]:
        """Linear regression slope over 20 days."""
        recent = prices.tail(20).values
        if len(recent) < 10:
            return {r: 0.25 for r in REGIMES}

        x = np.arange(len(recent))
        try:
            slope = np.polyfit(x, recent, 1)[0]
        except (np.linalg.LinAlgError, ValueError):
            return {r: 0.25 for r in REGIMES}

        normalized_slope = slope / (recent.mean() + 1e-10) * 100  # % per day

        if normalized_slope > 0.15:
            return {"bull": 0.60, "bear": 0.05, "neutral": 0.25, "volatile": 0.10}
        elif normalized_slope < -0.15:
            return {"bull": 0.05, "bear": 0.60, "neutral": 0.25, "volatile": 0.10}
        else:
            return {"bull": 0.20, "bear": 0.20, "neutral": 0.50, "volatile": 0.10}

    def _indicator_volatility(self, returns: pd.Series) -> Dict[str, float]:
        """Realized volatility vs historical."""
        if len(returns) < 60:
            return {r: 0.25 for r in REGIMES}

        vol_20 = returns.tail(20).std() * np.sqrt(252)
        vol_60 = returns.tail(60).std() * np.sqrt(252)

        # Classify volatility
        if vol_20 > 0.35:
            return {"bull": 0.05, "bear": 0.20, "neutral": 0.05, "volatile": 0.70}
        elif vol_20 > 0.25:
            return {"bull": 0.10, "bear": 0.25, "neutral": 0.15, "volatile": 0.50}
        elif vol_20 < 0.10:
            # Very low vol = complacency (could precede crash)
            return {"bull": 0.30, "bear": 0.10, "neutral": 0.50, "volatile": 0.10}
        else:
            # Normal vol
            vol_ratio = vol_20 / (vol_60 + 1e-10)
            if vol_ratio > 1.5:
                return {"bull": 0.10, "bear": 0.20, "neutral": 0.20, "volatile": 0.50}
            else:
                return {"bull": 0.25, "bear": 0.15, "neutral": 0.45, "volatile": 0.15}

    def _indicator_vol_clustering(self, returns: pd.Series) -> Dict[str, float]:
        """Detect volatility clustering (GARCH-like persistence)."""
        if len(returns) < 30:
            return {r: 0.25 for r in REGIMES}

        abs_returns = returns.abs().tail(30)
        # Autocorrelation of absolute returns (vol clustering proxy)
        autocorr = abs_returns.autocorr(lag=1)
        if autocorr is None or np.isnan(autocorr):
            autocorr = 0.0

        if autocorr > 0.3:
            # High persistence = volatile regime likely continuing
            return {"bull": 0.10, "bear": 0.20, "neutral": 0.10, "volatile": 0.60}
        elif autocorr > 0.15:
            return {"bull": 0.15, "bear": 0.20, "neutral": 0.25, "volatile": 0.40}
        else:
            return {"bull": 0.25, "bear": 0.15, "neutral": 0.45, "volatile": 0.15}

    def _indicator_breadth(
        self, universe_data: Optional[Dict[str, pd.DataFrame]]
    ) -> Dict[str, float]:
        """Market breadth: % of stocks above their 50-day MA."""
        if not universe_data or len(universe_data) < 10:
            return {r: 0.25 for r in REGIMES}

        above_50ma = 0
        total = 0
        for sym, df in universe_data.items():
            if isinstance(df, pd.DataFrame) and "Close" in df.columns and len(df) >= 50:
                close = df["Close"]
                ma50 = close.rolling(50).mean()
                if not ma50.empty and not np.isnan(ma50.iloc[-1]):
                    total += 1
                    if close.iloc[-1] > ma50.iloc[-1]:
                        above_50ma += 1

        if total < 10:
            return {r: 0.25 for r in REGIMES}

        breadth_pct = above_50ma / total

        if breadth_pct > 0.70:
            return {"bull": 0.60, "bear": 0.05, "neutral": 0.25, "volatile": 0.10}
        elif breadth_pct > 0.50:
            return {"bull": 0.35, "bear": 0.10, "neutral": 0.45, "volatile": 0.10}
        elif breadth_pct > 0.30:
            return {"bull": 0.10, "bear": 0.35, "neutral": 0.40, "volatile": 0.15}
        else:
            return {"bull": 0.05, "bear": 0.60, "neutral": 0.15, "volatile": 0.20}

    def _indicator_momentum_diffusion(self, returns: pd.Series) -> Dict[str, float]:
        """Momentum diffusion: consistency of recent returns."""
        if len(returns) < 20:
            return {r: 0.25 for r in REGIMES}

        recent = returns.tail(20)
        pct_positive = (recent > 0).mean()
        cum_return = (1 + recent).prod() - 1

        if pct_positive > 0.65 and cum_return > 0.03:
            return {"bull": 0.60, "bear": 0.05, "neutral": 0.25, "volatile": 0.10}
        elif pct_positive < 0.35 and cum_return < -0.03:
            return {"bull": 0.05, "bear": 0.60, "neutral": 0.20, "volatile": 0.15}
        elif abs(cum_return) < 0.01:
            return {"bull": 0.15, "bear": 0.15, "neutral": 0.60, "volatile": 0.10}
        else:
            return {"bull": 0.25, "bear": 0.20, "neutral": 0.35, "volatile": 0.20}

    def _indicator_vix(self, vix_level: Optional[float]) -> Dict[str, float]:
        """VIX-based regime indicator."""
        if vix_level is None:
            return {r: 0.25 for r in REGIMES}

        if vix_level > 35:
            return {"bull": 0.05, "bear": 0.30, "neutral": 0.05, "volatile": 0.60}
        elif vix_level > 25:
            return {"bull": 0.10, "bear": 0.25, "neutral": 0.15, "volatile": 0.50}
        elif vix_level > 20:
            return {"bull": 0.15, "bear": 0.20, "neutral": 0.35, "volatile": 0.30}
        elif vix_level > 12:
            return {"bull": 0.35, "bear": 0.10, "neutral": 0.45, "volatile": 0.10}
        else:
            # Very low VIX = complacency
            return {"bull": 0.30, "bear": 0.15, "neutral": 0.40, "volatile": 0.15}

    def _indicator_correlation(
        self, universe_data: Optional[Dict[str, pd.DataFrame]]
    ) -> Dict[str, float]:
        """Average pairwise correlation spike detection."""
        if not universe_data or len(universe_data) < 10:
            return {r: 0.25 for r in REGIMES}

        # Sample up to 20 symbols for correlation
        symbols = list(universe_data.keys())[:20]
        returns_dict = {}
        for sym in symbols:
            df = universe_data[sym]
            if isinstance(df, pd.DataFrame) and "Close" in df.columns and len(df) >= 30:
                ret = df["Close"].pct_change().dropna().tail(20)
                if len(ret) >= 15:
                    returns_dict[sym] = ret

        if len(returns_dict) < 5:
            return {r: 0.25 for r in REGIMES}

        # Build return matrix and compute average correlation
        ret_df = pd.DataFrame(returns_dict)
        corr_matrix = ret_df.corr()
        n = len(corr_matrix)
        if n < 2:
            return {r: 0.25 for r in REGIMES}

        # Mean off-diagonal correlation
        mask = np.ones_like(corr_matrix.values, dtype=bool)
        np.fill_diagonal(mask, False)
        avg_corr = corr_matrix.values[mask].mean()

        if np.isnan(avg_corr):
            return {r: 0.25 for r in REGIMES}

        if avg_corr > 0.6:
            # High correlation = risk-off / panic
            return {"bull": 0.05, "bear": 0.35, "neutral": 0.10, "volatile": 0.50}
        elif avg_corr > 0.4:
            return {"bull": 0.10, "bear": 0.25, "neutral": 0.25, "volatile": 0.40}
        elif avg_corr < 0.15:
            # Low correlation = healthy, diversified
            return {"bull": 0.30, "bear": 0.10, "neutral": 0.50, "volatile": 0.10}
        else:
            return {"bull": 0.25, "bear": 0.15, "neutral": 0.45, "volatile": 0.15}

    # ─── Probability Combination & Smoothing ───────────────────────

    def _combine_votes(
        self, indicator_scores: Dict[str, Dict[str, float]]
    ) -> Dict[str, float]:
        """Combine indicator votes via weighted average."""
        combined = {r: 0.0 for r in REGIMES}

        for indicator_name, votes in indicator_scores.items():
            weight = self.INDICATOR_WEIGHTS.get(indicator_name, 0.1)
            for regime in REGIMES:
                combined[regime] += weight * votes.get(regime, 0.25)

        # Normalize
        total = sum(combined.values())
        if total > 0:
            combined = {r: v / total for r, v in combined.items()}
        else:
            combined = {r: 0.25 for r in REGIMES}

        return combined

    def _smooth_probabilities(
        self, raw_probs: Dict[str, float]
    ) -> Dict[str, float]:
        """Apply EMA smoothing to prevent hard regime transitions."""
        alpha = self.smoothing_alpha
        smoothed = {}
        for regime in REGIMES:
            prev = self._smoothed_probs.get(regime, 0.25)
            new = raw_probs.get(regime, 0.25)
            smoothed[regime] = alpha * new + (1 - alpha) * prev

        # Renormalize
        total = sum(smoothed.values())
        if total > 0:
            smoothed = {r: v / total for r, v in smoothed.items()}

        return smoothed

    def _compute_transition_probability(
        self, smoothed: Dict[str, float], current: str
    ) -> float:
        """Compute probability of regime transition."""
        current_prob = smoothed.get(current, 0.25)
        # Transition probability = 1 - current regime probability
        # High when current regime is losing dominance
        return max(0.0, min(1.0, 1.0 - current_prob * 2))

    # ─── Sub-Regime Detection ──────────────────────────────────────

    def _detect_sub_regime(
        self, primary: str, age_days: int, prices: pd.Series
    ) -> SubRegime:
        """Detect sub-regime based on phase and characteristics."""
        returns = prices.pct_change().dropna()
        if len(returns) < 20:
            return SubRegime.RANGE_BOUND

        ret_20 = returns.tail(20).mean() * 252
        momentum_accel = 0.0
        if len(returns) >= 10:
            ret_recent = returns.tail(5).mean()
            ret_prior = returns.tail(10).head(5).mean()
            momentum_accel = ret_recent - ret_prior

        if primary == "bull":
            if age_days <= 10:
                return SubRegime.EARLY_BULL
            elif momentum_accel < -0.001:
                return SubRegime.LATE_BULL
            else:
                return SubRegime.MID_BULL

        elif primary == "bear":
            if age_days <= 10:
                return SubRegime.EARLY_BEAR
            elif ret_20 < -0.40:
                return SubRegime.CAPITULATION
            else:
                return SubRegime.MID_BEAR

        elif primary == "volatile":
            if ret_20 > 0:
                return SubRegime.VOLATILE_UP
            else:
                return SubRegime.VOLATILE_DOWN

        else:  # neutral
            if len(returns) >= 60 and returns.tail(60).std() * np.sqrt(252) < 0.12:
                return SubRegime.RANGE_BOUND
            elif momentum_accel > 0.001:
                return SubRegime.RECOVERY
            else:
                return SubRegime.RANGE_BOUND

    # ─── Leading Indicators ────────────────────────────────────────

    def _compute_leading_indicators(
        self,
        prices: pd.Series,
        indicator_scores: Dict[str, Dict[str, float]],
        vix_level: Optional[float],
    ) -> Dict[str, float]:
        """Compute leading indicators that detect regime changes early."""
        indicators = {}

        returns = prices.pct_change().dropna()
        if len(returns) < 60:
            return {"breadth_divergence": 0, "vol_compression": 0,
                    "momentum_divergence": 0, "correlation_spike": 0}

        # 1. Breadth divergence: price making highs while breadth declining
        price_at_high = prices.iloc[-1] >= prices.tail(20).quantile(0.9)
        breadth_score = indicator_scores.get("breadth", {}).get("bull", 0.25)
        indicators["breadth_divergence"] = (
            1.0 if (price_at_high and breadth_score < 0.3) else 0.0
        )

        # 2. Vol compression: extended low vol often precedes breakout
        vol_5 = returns.tail(5).std() * np.sqrt(252)
        vol_60 = returns.tail(60).std() * np.sqrt(252)
        vol_ratio = vol_5 / (vol_60 + 1e-10)
        indicators["vol_compression"] = max(0, 1.0 - vol_ratio) if vol_ratio < 0.5 else 0.0

        # 3. Momentum divergence: price trend vs momentum
        price_trend = (prices.iloc[-1] / prices.iloc[-20] - 1) if len(prices) >= 20 else 0
        momentum_score = indicator_scores.get("momentum_diffusion", {}).get("bull", 0.25)
        if price_trend > 0.03 and momentum_score < 0.2:
            indicators["momentum_divergence"] = 0.8
        elif price_trend < -0.03 and momentum_score > 0.5:
            indicators["momentum_divergence"] = 0.8
        else:
            indicators["momentum_divergence"] = 0.0

        # 4. Correlation spike
        corr_vol_score = indicator_scores.get("correlation", {}).get("volatile", 0.15)
        indicators["correlation_spike"] = min(1.0, corr_vol_score * 2) if corr_vol_score > 0.35 else 0.0

        return indicators

    # ─── Helpers ───────────────────────────────────────────────────

    def _neutral_assessment(self) -> RegimeAssessment:
        """Return neutral assessment when insufficient data."""
        return RegimeAssessment(
            primary_regime="neutral",
            sub_regime=SubRegime.RANGE_BOUND,
            regime_probabilities={r: 0.25 for r in REGIMES},
            transition_probability=0.0,
            regime_age_days=0,
            regime_strength=0.25,
            leading_indicators={
                "breadth_divergence": 0, "vol_compression": 0,
                "momentum_divergence": 0, "correlation_spike": 0,
            },
        )

    def get_diagnostics(self) -> Dict:
        """Return current detector state for monitoring."""
        return {
            "current_regime": self._current_regime,
            "smoothed_probabilities": dict(self._smoothed_probs),
            "regime_age_days": self._regime_age_days,
            "history_length": len(self._history),
        }
