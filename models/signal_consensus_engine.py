"""
models/signal_consensus_engine.py - Multi-Generator Signal Consensus Engine

Runs multiple independent signal generators in parallel and requires
majority agreement before producing a tradeable signal.

Key features:
- Normalizes heterogeneous generator outputs to [-1, 1] + confidence
- Dynamic weighting by rolling performance (not static)
- Majority direction agreement required
- Conviction scoring (0-100) combining agreement, strength, confidence
- Per-generator accuracy tracking with outcome feedback
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from datetime import datetime
from collections import deque
import logging

from config import ApexConfig

logger = logging.getLogger(__name__)


@dataclass
class GeneratorPerformance:
    """Rolling performance metrics for a single generator."""
    generator_name: str
    total_signals: int = 0
    correct_signals: int = 0
    returns_history: List[float] = field(default_factory=list)
    weight: float = 0.333
    last_updated: datetime = field(default_factory=datetime.now)

    @property
    def accuracy_30d(self) -> float:
        if self.total_signals < 5:
            return 0.50
        return self.correct_signals / max(1, self.total_signals)

    @property
    def sharpe_30d(self) -> float:
        if len(self.returns_history) < 5:
            return 0.0
        arr = np.array(self.returns_history[-30:])
        if arr.std() < 1e-10:
            return 0.0
        return float(arr.mean() / arr.std() * np.sqrt(252))


@dataclass
class ConsensusResult:
    """Output of the consensus engine."""
    symbol: str
    timestamp: datetime

    # Consensus signal
    consensus_signal: float
    consensus_confidence: float
    conviction_score: float
    direction_agreement: float

    # Component breakdown
    generator_signals: Dict[str, float]
    generator_confidences: Dict[str, float]
    generator_weights: Dict[str, float]

    # Vetoes
    majority_agrees: bool
    strong_consensus: bool
    vetoed: bool
    veto_reason: str = ""


class SignalConsensusEngine:
    """
    Meta-layer: runs 3+ independent generators and requires consensus.

    Generators are normalized to a common (signal, confidence) output.
    Weights adapt based on rolling performance.
    """

    def __init__(
        self,
        generators: Dict[str, Any],
        min_agreement: float = 0.60,
        min_generators: int = 2,
        min_conviction: float = 30.0,
    ):
        """
        Args:
            generators: Dict of name -> generator instance
            min_agreement: Minimum fraction of generators agreeing on direction
            min_generators: Minimum generators that must produce valid signals
            min_conviction: Minimum conviction score to not veto
        """
        self.generators = generators
        self.min_agreement = min_agreement
        self.min_generators = min_generators
        self.min_conviction = min_conviction

        # Performance tracking per generator
        self.performance: Dict[str, GeneratorPerformance] = {}
        for name in generators:
            self.performance[name] = GeneratorPerformance(
                generator_name=name,
                weight=1.0 / len(generators),
            )

        # Signal history for outcome tracking
        self._signal_history: deque = deque(maxlen=500)

        logger.info(
            f"SignalConsensusEngine initialized with {len(generators)} generators: "
            f"{list(generators.keys())}"
        )

    def generate_consensus(
        self,
        symbol: str,
        prices: pd.Series,
        features: Optional[pd.DataFrame] = None,
    ) -> ConsensusResult:
        """
        Run all generators and produce a consensus signal.

        Args:
            symbol: Stock symbol
            prices: Price series
            features: Optional pre-computed features for EnsembleGenerator

        Returns:
            ConsensusResult with consensus signal and voting details
        """
        signals: Dict[str, float] = {}
        confidences: Dict[str, float] = {}
        now = datetime.now()

        # Run each generator
        for name, gen in self.generators.items():
            try:
                sig, conf = self._run_generator(name, gen, symbol, prices, features)
                if sig is not None and not np.isnan(sig):
                    signals[name] = float(np.clip(sig, -1, 1))
                    confidences[name] = float(np.clip(conf, 0, 1))
            except Exception as e:
                logger.debug(f"Generator {name} failed for {symbol}: {e}")

        # Check minimum generator count
        if len(signals) < self.min_generators:
            return self._vetoed_result(
                symbol, now, signals, confidences,
                f"Only {len(signals)}/{self.min_generators} generators produced signals"
            )

        # Calculate dynamic weights
        regime = self._estimate_regime(prices)
        weights = self._calculate_dynamic_weights(regime)

        # Compute weighted consensus signal
        weighted_signal = 0.0
        weighted_confidence = 0.0
        total_weight = 0.0

        for name in signals:
            w = weights.get(name, 1.0 / len(signals))
            weighted_signal += signals[name] * w
            weighted_confidence += confidences[name] * w
            total_weight += w

        if total_weight > 0:
            weighted_signal /= total_weight
            weighted_confidence /= total_weight

        # Direction agreement
        directions = {name: np.sign(sig) for name, sig in signals.items() if abs(sig) > 0.05}
        if not directions:
            return self._vetoed_result(
                symbol, now, signals, confidences,
                "All signals too weak (< 0.05)"
            )

        majority_dir = np.sign(sum(directions.values()))
        agreeing = sum(1 for d in directions.values() if d == majority_dir)
        agreement = agreeing / len(directions)

        majority_agrees = agreement >= self.min_agreement

        # Strong consensus: all agree with decent strength
        strong_consensus = (
            agreement >= 0.99 and
            all(abs(s) > 0.3 for s in signals.values())
        )

        # Conviction score
        conviction = self._compute_conviction_score(
            signals, weights, agreement, weighted_confidence, strong_consensus
        )

        # Veto check
        vetoed = not majority_agrees or conviction < self.min_conviction
        veto_reason = ""
        if not majority_agrees:
            veto_reason = f"Direction agreement {agreement:.0%} < {self.min_agreement:.0%}"
        elif conviction < self.min_conviction:
            veto_reason = f"Conviction {conviction:.0f} < {self.min_conviction:.0f}"

        # Store for outcome tracking
        self._signal_history.append({
            "symbol": symbol,
            "timestamp": now,
            "signals": dict(signals),
            "consensus_signal": weighted_signal,
            "vetoed": vetoed,
        })

        return ConsensusResult(
            symbol=symbol,
            timestamp=now,
            consensus_signal=float(np.clip(weighted_signal, -1, 1)),
            consensus_confidence=weighted_confidence,
            conviction_score=conviction,
            direction_agreement=agreement,
            generator_signals=signals,
            generator_confidences=confidences,
            generator_weights=weights,
            majority_agrees=majority_agrees,
            strong_consensus=strong_consensus,
            vetoed=vetoed,
            veto_reason=veto_reason,
        )

    def record_outcome(
        self,
        symbol: str,
        generator_signals: Dict[str, float],
        actual_return: float,
    ):
        """
        Record outcome to update per-generator performance.

        Args:
            symbol: Stock symbol
            generator_signals: Dict of generator_name -> signal_value at entry
            actual_return: Actual forward return achieved
        """
        for name, signal in generator_signals.items():
            if name not in self.performance:
                continue

            perf = self.performance[name]
            perf.total_signals += 1

            # Correct if signal direction matches return direction
            if np.sign(signal) == np.sign(actual_return) and abs(actual_return) > 0.001:
                perf.correct_signals += 1

            # Track return magnitude (signal-weighted)
            perf.returns_history.append(actual_return * np.sign(signal))
            if len(perf.returns_history) > 100:
                perf.returns_history = perf.returns_history[-100:]

            perf.last_updated = datetime.now()

    # ─── Internal Methods ──────────────────────────────────────────

    def _run_generator(
        self,
        name: str,
        gen: Any,
        symbol: str,
        prices: pd.Series,
        features: Optional[pd.DataFrame],
    ) -> Tuple[Optional[float], float]:
        """
        Run a single generator and normalize output.

        Returns:
            (signal, confidence) tuple, or (None, 0) on failure
        """
        # InstitutionalSignalGenerator / UltimateSignalGenerator
        if hasattr(gen, "generate_signal"):
            # Check if it's the ensemble generator (needs features + prices)
            import inspect
            sig = inspect.signature(gen.generate_signal)
            params = list(sig.parameters.keys())

            if "features" in params and features is not None:
                result = gen.generate_signal(features=features, prices=prices)
            else:
                result = gen.generate_signal(symbol=symbol, prices=prices)

            return self._normalize_output(result)

        # GodLevel / Advanced generators use generate_ml_signal
        if hasattr(gen, "generate_ml_signal"):
            result = gen.generate_ml_signal(symbol=symbol, prices=prices)
            return self._normalize_output(result)

        logger.warning(f"Generator {name} has no recognized signal method")
        return None, 0.0

    def _normalize_output(self, result: Any) -> Tuple[Optional[float], float]:
        """Normalize heterogeneous generator outputs to (signal, confidence)."""
        if result is None:
            return None, 0.0

        # SignalOutput dataclass
        if hasattr(result, "signal") and hasattr(result, "confidence"):
            return result.signal, result.confidence

        # Dict output
        if isinstance(result, dict):
            signal = result.get("signal", result.get("prediction", None))
            confidence = result.get("confidence", 0.5)
            if signal is not None:
                return float(signal), float(confidence)

        # Tuple (signal, confidence)
        if isinstance(result, (tuple, list)) and len(result) >= 2:
            return float(result[0]), float(result[1])

        return None, 0.0

    def _calculate_dynamic_weights(self, regime: str) -> Dict[str, float]:
        """
        Calculate weights based on rolling performance.

        Weight = f(accuracy - 0.50, Sharpe), clipped to [0.10, 0.60].
        Falls back to equal weights if insufficient data.
        """
        weights = {}
        has_data = False

        regime_weights = getattr(ApexConfig, "CONSENSUS_REGIME_WEIGHTS", {}).get(regime, {})

        for name, perf in self.performance.items():
            if perf.total_signals >= 20:
                has_data = True
                edge = max(0.01, perf.accuracy_30d - 0.45)
                sharpe_factor = max(0.1, 1.0 + perf.sharpe_30d * 0.1)
                raw = edge * sharpe_factor
            else:
                raw = 1.0 / len(self.performance)

            weights[name] = raw * regime_weights.get(name, 1.0)

        # Normalize and clip
        total = sum(weights.values())
        if total > 0:
            weights = {k: v / total for k, v in weights.items()}

        # Clip to [0.10, 0.60]
        if has_data:
            weights = {k: np.clip(v, 0.10, 0.60) for k, v in weights.items()}
            total = sum(weights.values())
            weights = {k: v / total for k, v in weights.items()}

        return weights

    def _estimate_regime(self, prices: pd.Series) -> str:
        """Lightweight regime detection for weighting."""
        if prices is None or len(prices) < 60:
            return "neutral"
        returns = prices.pct_change().dropna()
        vol = returns.std() * np.sqrt(252)
        ma_20 = prices.iloc[-20:].mean()
        ma_60 = prices.iloc[-60:].mean()
        trend = (ma_20 - ma_60) / ma_60 if ma_60 > 0 else 0.0
        if vol > 0.35:
            return "volatile"
        if trend > 0.05:
            return "bull"
        if trend < -0.05:
            return "bear"
        return "neutral"

    def _compute_conviction_score(
        self,
        signals: Dict[str, float],
        weights: Dict[str, float],
        agreement: float,
        avg_confidence: float,
        strong_consensus: bool,
    ) -> float:
        """
        Compute 0-100 conviction score.

        Components:
        - Direction agreement (40 pts)
        - Weighted signal magnitude (30 pts)
        - Average confidence (20 pts)
        - Unanimous strong bonus (10 pts)
        """
        # Agreement component (0-40)
        agreement_pts = agreement * 40

        # Signal strength component (0-30)
        weighted_abs = sum(
            abs(signals[n]) * weights.get(n, 0.33)
            for n in signals
        )
        total_w = sum(weights.get(n, 0.33) for n in signals)
        if total_w > 0:
            weighted_abs /= total_w
        strength_pts = min(1.0, weighted_abs) * 30

        # Confidence component (0-20)
        confidence_pts = min(1.0, avg_confidence) * 20

        # Unanimous bonus (0-10)
        bonus_pts = 10.0 if strong_consensus else 0.0

        return agreement_pts + strength_pts + confidence_pts + bonus_pts

    def _vetoed_result(
        self,
        symbol: str,
        timestamp: datetime,
        signals: Dict[str, float],
        confidences: Dict[str, float],
        reason: str,
    ) -> ConsensusResult:
        """Create a vetoed consensus result."""
        weights = self._calculate_dynamic_weights("neutral")
        return ConsensusResult(
            symbol=symbol,
            timestamp=timestamp,
            consensus_signal=0.0,
            consensus_confidence=0.0,
            conviction_score=0.0,
            direction_agreement=0.0,
            generator_signals=signals,
            generator_confidences=confidences,
            generator_weights=weights,
            majority_agrees=False,
            strong_consensus=False,
            vetoed=True,
            veto_reason=reason,
        )

    def get_diagnostics(self) -> Dict:
        """Return current engine state for monitoring."""
        return {
            "generator_count": len(self.generators),
            "performance": {
                name: {
                    "accuracy": perf.accuracy_30d,
                    "sharpe": perf.sharpe_30d,
                    "total_signals": perf.total_signals,
                    "weight": perf.weight,
                }
                for name, perf in self.performance.items()
            },
            "signal_history_size": len(self._signal_history),
            "current_weights": self._calculate_dynamic_weights("neutral"),
        }
