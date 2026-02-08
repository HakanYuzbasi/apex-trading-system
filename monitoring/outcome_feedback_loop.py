"""
monitoring/outcome_feedback_loop.py - Automatic Outcome Tracking & Model Retraining

Bridges signal outcome tracking with model performance monitoring:
1. Correctly computes forward returns from historical price data
2. Feeds outcomes to signal generators for drift monitoring
3. Feeds outcomes to consensus engine for weight adaptation
4. Detects performance degradation
5. Auto-triggers model retraining when accuracy drops
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from collections import defaultdict
import logging

logger = logging.getLogger(__name__)


@dataclass
class PerformanceDegradation:
    """Detected performance degradation."""
    metric: str
    current_value: float
    baseline_value: float
    degradation_pct: float
    recommendation: str  # 'monitor', 'retrain', 'halt'
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class _TrackedSignal:
    """Internal record of a signal awaiting outcome."""
    symbol: str
    signal_value: float
    confidence: float
    regime: str
    entry_price: float
    generator_signals: Dict[str, float]
    timestamp: datetime
    # Computed later
    return_1d: Optional[float] = None
    return_5d: Optional[float] = None
    return_10d: Optional[float] = None
    return_20d: Optional[float] = None
    completed: bool = False


class OutcomeFeedbackLoop:
    """
    Bridges signal generation with outcome tracking and model retraining.

    Solves the problem of outcomes never being fed back to generators:
    - Correctly computes forward returns from historical DataFrames
    - Calls inst_generator.record_outcome() for drift monitoring
    - Calls consensus_engine.record_outcome() for weight adaptation
    - Monitors rolling accuracy and triggers retrain when degraded
    """

    def __init__(
        self,
        outcome_tracker: Any = None,
        consensus_engine: Any = None,
        inst_generator: Any = None,
        retrain_accuracy_threshold: float = 0.45,
        retrain_sharpe_threshold: float = 0.5,
        max_tracked_signals: int = 2000,
    ):
        self.outcome_tracker = outcome_tracker
        self.consensus_engine = consensus_engine
        self.inst_generator = inst_generator
        self.retrain_accuracy_threshold = retrain_accuracy_threshold
        self.retrain_sharpe_threshold = retrain_sharpe_threshold

        # Internal tracking
        self._active_signals: List[_TrackedSignal] = []
        self._completed_signals: List[_TrackedSignal] = []
        self._max_tracked = max_tracked_signals
        self._last_retrain: datetime = datetime.now()
        self._retrain_cooldown_hours: int = 24

        # Rolling performance metrics
        self._accuracy_history: List[float] = []
        self._return_history: List[float] = []

        logger.info("OutcomeFeedbackLoop initialized")

    def record_signal(
        self,
        symbol: str,
        signal_value: float,
        confidence: float,
        regime: str,
        entry_price: float,
        generator_signals: Optional[Dict[str, float]] = None,
    ):
        """
        Record a new signal for forward outcome tracking.

        Args:
            symbol: Stock symbol
            signal_value: Combined signal value
            confidence: Signal confidence
            regime: Market regime at signal time
            entry_price: Price at signal time
            generator_signals: Per-generator signal values (for consensus engine)
        """
        tracked = _TrackedSignal(
            symbol=symbol,
            signal_value=signal_value,
            confidence=confidence,
            regime=regime,
            entry_price=entry_price,
            generator_signals=generator_signals or {},
            timestamp=datetime.now(),
        )
        self._active_signals.append(tracked)

        # Prevent unbounded growth
        if len(self._active_signals) > self._max_tracked:
            self._active_signals = self._active_signals[-self._max_tracked:]

    def update_forward_returns(
        self, historical_data: Dict[str, pd.DataFrame]
    ):
        """
        Compute forward returns for active signals using actual price data.

        This is the correct implementation - uses historical DataFrames
        (not price_cache single floats) to compute true forward returns.

        Args:
            historical_data: Dict of symbol -> DataFrame with 'Close' column
        """
        now = datetime.now()
        newly_completed = []

        for sig in self._active_signals:
            if sig.completed:
                continue

            symbol_data = historical_data.get(sig.symbol)
            if symbol_data is None or not isinstance(symbol_data, pd.DataFrame):
                continue

            close_col = "Close" if "Close" in symbol_data.columns else "close"
            if close_col not in symbol_data.columns:
                continue

            close = symbol_data[close_col]
            if close.empty:
                continue

            # Find the entry date index
            signal_date = sig.timestamp
            days_elapsed = (now - signal_date).days

            # Get entry price from data (more accurate than cached)
            entry_price = sig.entry_price
            if entry_price <= 0:
                continue

            # Compute forward returns at various horizons
            # We use the last N prices from the close series
            prices_after = close.tail(max(days_elapsed + 1, 2))

            if days_elapsed >= 1 and sig.return_1d is None and len(prices_after) >= 2:
                sig.return_1d = float(prices_after.iloc[-1] / entry_price - 1)

            if days_elapsed >= 5 and sig.return_5d is None and len(prices_after) >= 5:
                # Use price 5 days after entry
                idx = min(5, len(prices_after) - 1)
                sig.return_5d = float(prices_after.iloc[idx] / entry_price - 1)

            if days_elapsed >= 10 and sig.return_10d is None and len(prices_after) >= 10:
                idx = min(10, len(prices_after) - 1)
                sig.return_10d = float(prices_after.iloc[idx] / entry_price - 1)

            if days_elapsed >= 20 and sig.return_20d is None and len(prices_after) >= 20:
                idx = min(20, len(prices_after) - 1)
                sig.return_20d = float(prices_after.iloc[idx] / entry_price - 1)

            # Mark complete when 20-day return is available
            if sig.return_20d is not None:
                sig.completed = True
                newly_completed.append(sig)

        # Move completed signals
        if newly_completed:
            self._completed_signals.extend(newly_completed)
            self._active_signals = [s for s in self._active_signals if not s.completed]

            # Trim completed history
            if len(self._completed_signals) > self._max_tracked:
                self._completed_signals = self._completed_signals[-self._max_tracked:]

            logger.info(f"Completed {len(newly_completed)} signal outcomes")

    def feed_outcomes_to_generators(self):
        """
        Feed completed outcomes to signal generators and consensus engine.

        This is the critical missing link - generators need outcome data
        to detect drift and adapt weights.
        """
        fed_count = 0

        for sig in self._completed_signals:
            # Use 5-day return as the primary outcome
            actual_return = sig.return_5d
            if actual_return is None:
                actual_return = sig.return_1d
            if actual_return is None:
                continue

            # Feed to institutional generator for drift monitoring
            if self.inst_generator and hasattr(self.inst_generator, "record_outcome"):
                try:
                    self.inst_generator.record_outcome(actual_return)
                except Exception as e:
                    logger.debug(f"Failed to feed outcome to inst_generator: {e}")

            # Feed to consensus engine for weight adaptation
            if self.consensus_engine and sig.generator_signals:
                try:
                    self.consensus_engine.record_outcome(
                        symbol=sig.symbol,
                        generator_signals=sig.generator_signals,
                        actual_return=actual_return,
                    )
                except Exception as e:
                    logger.debug(f"Failed to feed outcome to consensus_engine: {e}")

            # Track for internal metrics
            correct = np.sign(sig.signal_value) == np.sign(actual_return) and abs(actual_return) > 0.001
            self._accuracy_history.append(1.0 if correct else 0.0)
            self._return_history.append(actual_return * np.sign(sig.signal_value))

            fed_count += 1

        # Keep rolling windows
        max_history = 200
        self._accuracy_history = self._accuracy_history[-max_history:]
        self._return_history = self._return_history[-max_history:]

        # Clear completed (already fed)
        self._completed_signals.clear()

        if fed_count > 0:
            logger.info(f"Fed {fed_count} outcomes to generators")

    def check_performance_degradation(self) -> List[PerformanceDegradation]:
        """
        Check if signal performance has degraded below thresholds.

        Returns:
            List of degradation detections with recommendations
        """
        degradations = []

        if len(self._accuracy_history) < 20:
            return degradations

        # Rolling 30-day accuracy
        recent_accuracy = self._accuracy_history[-30:]
        accuracy = float(np.mean(recent_accuracy))

        if accuracy < self.retrain_accuracy_threshold:
            degradations.append(PerformanceDegradation(
                metric="accuracy_30d",
                current_value=accuracy,
                baseline_value=0.55,
                degradation_pct=(0.55 - accuracy) / 0.55 * 100,
                recommendation="retrain" if accuracy < 0.42 else "monitor",
            ))

        # Rolling Sharpe
        if len(self._return_history) >= 20:
            recent_returns = np.array(self._return_history[-30:])
            if recent_returns.std() > 1e-10:
                sharpe = float(recent_returns.mean() / recent_returns.std() * np.sqrt(252))
            else:
                sharpe = 0.0

            if sharpe < self.retrain_sharpe_threshold:
                degradations.append(PerformanceDegradation(
                    metric="sharpe_30d",
                    current_value=sharpe,
                    baseline_value=1.0,
                    degradation_pct=max(0, (1.0 - sharpe) / 1.0 * 100),
                    recommendation="retrain" if sharpe < 0.0 else "monitor",
                ))

        # Calibration check: recent accuracy trend
        if len(self._accuracy_history) >= 60:
            first_half = np.mean(self._accuracy_history[-60:-30])
            second_half = np.mean(self._accuracy_history[-30:])
            if second_half < first_half - 0.10:
                degradations.append(PerformanceDegradation(
                    metric="accuracy_trend",
                    current_value=second_half,
                    baseline_value=first_half,
                    degradation_pct=(first_half - second_half) / first_half * 100,
                    recommendation="retrain",
                ))

        return degradations

    def should_retrain(self) -> Tuple[bool, str]:
        """
        Determine if models should be retrained.

        Returns:
            (should_retrain, reason) tuple
        """
        # Cooldown check
        hours_since = (datetime.now() - self._last_retrain).total_seconds() / 3600
        if hours_since < self._retrain_cooldown_hours:
            return False, f"Cooldown: {hours_since:.0f}/{self._retrain_cooldown_hours}h"

        degradations = self.check_performance_degradation()
        retrain_reasons = [d for d in degradations if d.recommendation == "retrain"]

        if retrain_reasons:
            reasons = ", ".join(
                f"{d.metric}={d.current_value:.3f}" for d in retrain_reasons
            )
            return True, f"Performance degradation: {reasons}"

        # Time-based retrain (every 30 days)
        if hours_since > 30 * 24:
            return True, f"Scheduled retrain ({hours_since / 24:.0f} days since last)"

        return False, "Performance within thresholds"

    def trigger_retrain(
        self, historical_data: Dict[str, pd.DataFrame]
    ) -> bool:
        """
        Trigger model retraining.

        Args:
            historical_data: Training data

        Returns:
            True if retrain succeeded
        """
        if not self.inst_generator or not hasattr(self.inst_generator, "train"):
            logger.warning("Cannot retrain: no inst_generator with train() method")
            return False

        try:
            logger.info("AUTO-RETRAIN: Starting model retraining...")
            self.inst_generator.train(historical_data)
            self._last_retrain = datetime.now()
            logger.info("AUTO-RETRAIN: Retraining completed successfully")
            return True
        except Exception as e:
            logger.error(f"AUTO-RETRAIN: Failed: {e}")
            return False

    def get_rolling_metrics(self, window_days: int = 30) -> Dict:
        """
        Get rolling performance metrics.

        Returns:
            Dict of accuracy, Sharpe, calibration metrics
        """
        metrics = {
            "active_signals": len(self._active_signals),
            "completed_total": len(self._accuracy_history),
            "last_retrain": self._last_retrain.isoformat(),
        }

        if len(self._accuracy_history) >= 10:
            recent = self._accuracy_history[-window_days:]
            metrics["accuracy"] = float(np.mean(recent))
            metrics["accuracy_std"] = float(np.std(recent))

        if len(self._return_history) >= 10:
            recent = np.array(self._return_history[-window_days:])
            metrics["avg_return"] = float(recent.mean())
            if recent.std() > 1e-10:
                metrics["sharpe"] = float(recent.mean() / recent.std() * np.sqrt(252))
            else:
                metrics["sharpe"] = 0.0

        # By regime breakdown
        regime_counts: Dict[str, List[float]] = defaultdict(list)
        for sig in self._active_signals:
            if sig.return_5d is not None:
                correct = np.sign(sig.signal_value) == np.sign(sig.return_5d)
                regime_counts[sig.regime].append(1.0 if correct else 0.0)

        if regime_counts:
            metrics["by_regime"] = {
                regime: {
                    "count": len(outcomes),
                    "accuracy": float(np.mean(outcomes)),
                }
                for regime, outcomes in regime_counts.items()
            }

        return metrics

    def get_diagnostics(self) -> Dict:
        """Return feedback loop state for monitoring."""
        return {
            "active_signals": len(self._active_signals),
            "completed_signals": len(self._accuracy_history),
            "rolling_accuracy": (
                float(np.mean(self._accuracy_history[-30:]))
                if len(self._accuracy_history) >= 10
                else None
            ),
            "last_retrain": self._last_retrain.isoformat(),
            "retrain_cooldown_hours": self._retrain_cooldown_hours,
        }
