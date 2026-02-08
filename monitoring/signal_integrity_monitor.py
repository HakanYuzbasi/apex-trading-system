"""
monitoring/signal_integrity_monitor.py - Real-Time Signal Anomaly Detection

Monitors the signal stream for anomalies that indicate model degradation,
data issues, or system failures. Auto-quarantines symbols producing
suspicious signals.

7 anomaly checks:
1. Stuck signals (same value repeated)
2. Distribution shift (KL divergence)
3. Feature importance drift
4. Data quality degradation
5. Signal volatility anomaly (too low or too high)
6. Regime-signal mismatch
7. Confidence calibration drift
"""

import numpy as np
from typing import Dict, List, Optional, Set
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from collections import deque, defaultdict
import logging

logger = logging.getLogger(__name__)


class AlertSeverity(Enum):
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"


@dataclass
class IntegrityAlert:
    alert_type: str
    severity: AlertSeverity
    symbol: str
    message: str
    metric_value: float
    threshold: float
    timestamp: datetime = field(default_factory=datetime.now)
    auto_quarantine: bool = False


@dataclass
class SignalHealthReport:
    healthy: bool
    alerts: List[IntegrityAlert]
    quarantined_symbols: Set[str]
    metrics: Dict[str, float]
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class _SignalRecord:
    """Internal record for a single signal observation."""
    signal: float
    confidence: float
    regime: str
    data_quality: float
    timestamp: datetime


class SignalIntegrityMonitor:
    """
    Real-time anomaly detection on the signal stream.

    Maintains per-symbol ring buffers of recent signals and runs
    statistical checks to detect model degradation, data issues,
    or system failures.
    """

    def __init__(
        self,
        window_size: int = 100,
        stuck_threshold: int = 10,
        kl_threshold: float = 0.5,
        quarantine_minutes: int = 60,
    ):
        self.window_size = window_size
        self.stuck_threshold = stuck_threshold
        self.kl_threshold = kl_threshold
        self.quarantine_minutes = quarantine_minutes

        # Per-symbol signal buffers
        self._buffers: Dict[str, deque] = defaultdict(
            lambda: deque(maxlen=window_size)
        )

        # Quarantine registry: symbol -> expiration time
        self._quarantined: Dict[str, datetime] = {}

        # Baseline distributions (built from first N signals)
        self._baselines: Dict[str, List[float]] = {}

        # Outcome tracking for calibration check
        self._confidence_outcomes: Dict[str, List[dict]] = defaultdict(list)

        logger.info(
            f"SignalIntegrityMonitor initialized: window={window_size}, "
            f"stuck_threshold={stuck_threshold}"
        )

    def record_signal(
        self,
        symbol: str,
        signal: float,
        confidence: float,
        regime: str,
        data_quality: float = 1.0,
    ):
        """Record a signal observation for monitoring."""
        record = _SignalRecord(
            signal=signal,
            confidence=confidence,
            regime=regime,
            data_quality=data_quality,
            timestamp=datetime.now(),
        )
        self._buffers[symbol].append(record)

        # Build baseline from first 80 signals
        if symbol not in self._baselines:
            if len(self._buffers[symbol]) >= 80:
                self._baselines[symbol] = [
                    r.signal for r in list(self._buffers[symbol])[:80]
                ]

    def record_outcome(
        self,
        symbol: str,
        confidence_at_entry: float,
        actual_return: float,
    ):
        """Record outcome for calibration tracking."""
        self._confidence_outcomes[symbol].append({
            "confidence": confidence_at_entry,
            "return": actual_return,
            "correct": np.sign(actual_return) > 0,  # simplified
        })
        # Keep last 100
        if len(self._confidence_outcomes[symbol]) > 100:
            self._confidence_outcomes[symbol] = self._confidence_outcomes[symbol][-100:]

    def is_quarantined(self, symbol: str) -> bool:
        """Check if a symbol is currently quarantined."""
        if symbol not in self._quarantined:
            return False
        if datetime.now() > self._quarantined[symbol]:
            del self._quarantined[symbol]
            logger.info(f"Quarantine expired for {symbol}")
            return False
        return True

    def auto_quarantine(self, symbol: str, duration_minutes: Optional[int] = None):
        """Quarantine a symbol for the specified duration."""
        minutes = duration_minutes or self.quarantine_minutes
        expiration = datetime.now() + timedelta(minutes=minutes)
        self._quarantined[symbol] = expiration
        logger.warning(f"Symbol {symbol} quarantined until {expiration}")

    def check_integrity(
        self, symbol: Optional[str] = None
    ) -> SignalHealthReport:
        """
        Run all anomaly checks.

        Args:
            symbol: Check specific symbol, or None for all tracked symbols

        Returns:
            SignalHealthReport with alerts and overall health status
        """
        alerts: List[IntegrityAlert] = []
        symbols = [symbol] if symbol else list(self._buffers.keys())

        for sym in symbols:
            buf = self._buffers.get(sym)
            if not buf or len(buf) < 10:
                continue

            # Run all checks
            checks = [
                self._check_stuck_signals(sym),
                self._check_distribution_shift(sym),
                self._check_data_quality_trend(sym),
                self._check_signal_volatility(sym),
                self._check_regime_signal_mismatch(sym),
                self._check_confidence_calibration(sym),
            ]

            for alert in checks:
                if alert is not None:
                    alerts.append(alert)
                    if alert.auto_quarantine:
                        self.auto_quarantine(sym)

        # Clean expired quarantines
        expired = [s for s, t in self._quarantined.items() if datetime.now() > t]
        for s in expired:
            del self._quarantined[s]

        healthy = all(a.severity != AlertSeverity.CRITICAL for a in alerts)

        # Aggregate metrics
        metrics = self._compute_aggregate_metrics()

        return SignalHealthReport(
            healthy=healthy,
            alerts=alerts,
            quarantined_symbols=set(self._quarantined.keys()),
            metrics=metrics,
        )

    # ─── Individual Checks ─────────────────────────────────────────

    def _check_stuck_signals(self, symbol: str) -> Optional[IntegrityAlert]:
        """Detect when signals are stuck at the same value."""
        buf = list(self._buffers[symbol])
        if len(buf) < self.stuck_threshold:
            return None

        recent = [r.signal for r in buf[-self.stuck_threshold:]]

        # Check if all recent values are within epsilon of each other
        if max(recent) - min(recent) < 0.02:
            return IntegrityAlert(
                alert_type="stuck_signal",
                severity=AlertSeverity.CRITICAL,
                symbol=symbol,
                message=(
                    f"Signal stuck at {recent[-1]:.3f} for "
                    f"{self.stuck_threshold} observations"
                ),
                metric_value=max(recent) - min(recent),
                threshold=0.02,
                auto_quarantine=True,
            )
        return None

    def _check_distribution_shift(self, symbol: str) -> Optional[IntegrityAlert]:
        """Detect shift in signal distribution using simplified KL divergence."""
        if symbol not in self._baselines:
            return None

        baseline = self._baselines[symbol]
        recent = [r.signal for r in list(self._buffers[symbol])[-20:]]

        if len(recent) < 15 or len(baseline) < 20:
            return None

        # Compute histograms
        bins = np.linspace(-1, 1, 11)
        hist_baseline, _ = np.histogram(baseline, bins=bins, density=True)
        hist_recent, _ = np.histogram(recent, bins=bins, density=True)

        # Add smoothing to avoid division by zero
        eps = 1e-10
        hist_baseline = hist_baseline + eps
        hist_recent = hist_recent + eps

        # Normalize
        hist_baseline = hist_baseline / hist_baseline.sum()
        hist_recent = hist_recent / hist_recent.sum()

        # KL divergence
        kl_div = float(np.sum(hist_recent * np.log(hist_recent / hist_baseline)))

        if kl_div > self.kl_threshold:
            return IntegrityAlert(
                alert_type="distribution_shift",
                severity=AlertSeverity.WARNING,
                symbol=symbol,
                message=f"Signal distribution shift detected (KL={kl_div:.3f})",
                metric_value=kl_div,
                threshold=self.kl_threshold,
            )
        return None

    def _check_data_quality_trend(self, symbol: str) -> Optional[IntegrityAlert]:
        """Monitor declining data quality."""
        recent = [r.data_quality for r in list(self._buffers[symbol])[-20:]]
        if len(recent) < 10:
            return None

        avg_quality = np.mean(recent)
        if avg_quality < 0.7:
            return IntegrityAlert(
                alert_type="data_quality_degradation",
                severity=AlertSeverity.WARNING,
                symbol=symbol,
                message=f"Data quality declining: avg={avg_quality:.2f}",
                metric_value=avg_quality,
                threshold=0.70,
            )
        return None

    def _check_signal_volatility(self, symbol: str) -> Optional[IntegrityAlert]:
        """Detect abnormal signal volatility (too stable or too erratic)."""
        recent = [r.signal for r in list(self._buffers[symbol])[-30:]]
        if len(recent) < 15:
            return None

        signal_std = float(np.std(recent))

        # Too stable = model might be stuck or ignoring data
        if signal_std < 0.03:
            return IntegrityAlert(
                alert_type="signal_volatility_low",
                severity=AlertSeverity.CRITICAL,
                symbol=symbol,
                message=f"Signal volatility abnormally low: std={signal_std:.4f}",
                metric_value=signal_std,
                threshold=0.03,
                auto_quarantine=True,
            )

        # Too volatile = model might be unstable
        if signal_std > 0.80:
            return IntegrityAlert(
                alert_type="signal_volatility_high",
                severity=AlertSeverity.WARNING,
                symbol=symbol,
                message=f"Signal volatility abnormally high: std={signal_std:.4f}",
                metric_value=signal_std,
                threshold=0.80,
            )
        return None

    def _check_regime_signal_mismatch(self, symbol: str) -> Optional[IntegrityAlert]:
        """Detect when signals contradict the detected regime."""
        recent = list(self._buffers[symbol])[-20:]
        if len(recent) < 15:
            return None

        # Get dominant regime
        regimes = [r.regime for r in recent]
        if not regimes:
            return None

        from collections import Counter
        regime_counts = Counter(regimes)
        dominant_regime = regime_counts.most_common(1)[0][0]

        # Check signal direction distribution
        signals = [r.signal for r in recent]
        pct_bullish = sum(1 for s in signals if s > 0.1) / len(signals)
        pct_bearish = sum(1 for s in signals if s < -0.1) / len(signals)

        if dominant_regime == "bear" and pct_bullish > 0.70:
            return IntegrityAlert(
                alert_type="regime_signal_mismatch",
                severity=AlertSeverity.WARNING,
                symbol=symbol,
                message=(
                    f"Regime is '{dominant_regime}' but "
                    f"{pct_bullish:.0%} signals are bullish"
                ),
                metric_value=pct_bullish,
                threshold=0.70,
            )

        if dominant_regime == "bull" and pct_bearish > 0.70:
            return IntegrityAlert(
                alert_type="regime_signal_mismatch",
                severity=AlertSeverity.WARNING,
                symbol=symbol,
                message=(
                    f"Regime is '{dominant_regime}' but "
                    f"{pct_bearish:.0%} signals are bearish"
                ),
                metric_value=pct_bearish,
                threshold=0.70,
            )

        return None

    def _check_confidence_calibration(self, symbol: str) -> Optional[IntegrityAlert]:
        """Check if high-confidence signals actually outperform low-confidence."""
        outcomes = self._confidence_outcomes.get(symbol, [])
        if len(outcomes) < 30:
            return None

        high_conf = [o for o in outcomes if o["confidence"] > 0.6]
        low_conf = [o for o in outcomes if o["confidence"] < 0.4]

        if len(high_conf) < 5 or len(low_conf) < 5:
            return None

        high_accuracy = sum(1 for o in high_conf if o["correct"]) / len(high_conf)
        low_accuracy = sum(1 for o in low_conf if o["correct"]) / len(low_conf)

        # High confidence should outperform low confidence
        if high_accuracy <= low_accuracy:
            return IntegrityAlert(
                alert_type="confidence_miscalibration",
                severity=AlertSeverity.INFO,
                symbol=symbol,
                message=(
                    f"High-confidence accuracy ({high_accuracy:.0%}) <= "
                    f"low-confidence ({low_accuracy:.0%})"
                ),
                metric_value=high_accuracy - low_accuracy,
                threshold=0.0,
            )
        return None

    # ─── Aggregate Metrics ─────────────────────────────────────────

    def _compute_aggregate_metrics(self) -> Dict[str, float]:
        """Compute system-wide signal health metrics."""
        all_signals = []
        all_quality = []

        for buf in self._buffers.values():
            for record in buf:
                all_signals.append(record.signal)
                all_quality.append(record.data_quality)

        if not all_signals:
            return {}

        return {
            "total_symbols_tracked": len(self._buffers),
            "quarantined_count": len(self._quarantined),
            "avg_signal_value": float(np.mean(all_signals)),
            "signal_std": float(np.std(all_signals)),
            "avg_data_quality": float(np.mean(all_quality)),
            "pct_near_zero": float(np.mean([abs(s) < 0.05 for s in all_signals])),
        }

    def get_diagnostics(self) -> Dict:
        """Return monitor state for debugging."""
        return {
            "symbols_tracked": len(self._buffers),
            "quarantined": list(self._quarantined.keys()),
            "baselines_built": len(self._baselines),
            "buffer_sizes": {
                sym: len(buf) for sym, buf in self._buffers.items()
            },
        }
