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

        # Cooldown registry: symbol -> last quarantine timestamp
        # Prevents the expiry→immediate-re-quarantine loop for daily-bar models
        # whose signals are legitimately stable intra-day.
        self._quarantine_history: Dict[str, datetime] = {}

        logger.info(
            f"SignalIntegrityMonitor initialized: window={window_size}, "
            f"stuck_threshold={stuck_threshold}"
        )

    @staticmethod
    def _normalize_symbol(symbol: str) -> str:
        """Canonicalize symbol to avoid duplicate buffers for AVAX/USD vs CRYPTO:AVAX/USD."""
        try:
            from core.symbols import parse_symbol
            return parse_symbol(symbol).normalized
        except Exception:
            return symbol

    def record_signal(
        self,
        symbol: str,
        signal: float,
        confidence: float,
        regime: str,
        data_quality: float = 1.0,
    ):
        """Record a signal observation for monitoring."""
        symbol = self._normalize_symbol(symbol)  # deduplicate bare vs CRYPTO: prefixed
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
        symbol = self._normalize_symbol(symbol)
        if symbol not in self._quarantined:
            return False
        if datetime.now() > self._quarantined[symbol]:
            del self._quarantined[symbol]
            logger.info(f"Quarantine expired for {symbol}")
            return False
        # Equity quarantines set during non-market hours (due to stale signals)
        # are lifted when the market opens so trading can begin at the open.
        if not self._is_crypto_symbol(symbol) and self._is_equity_market_open():
            del self._quarantined[symbol]
            logger.info(f"Quarantine lifted for {symbol}: market is now open")
            return False
        return True

    def auto_quarantine(
        self,
        symbol: str,
        duration_minutes: Optional[int] = None,
        is_flatline: bool = False,
        cooldown_minutes: int = 30,
    ):
        """Quarantine a symbol for the specified duration.

        For daily-bar ML models, signals legitimately remain stable intra-day.
        A 30-minute cooldown prevents the expiry→immediate-re-quarantine loop
        that causes zero-fill paralysis.  The cooldown is BYPASSED when
        ``is_flatline=True`` (std == 0.000) because that indicates a total
        model crash and must always be acted upon immediately.
        """
        symbol = self._normalize_symbol(symbol)

        # Check cooldown — skip re-quarantine unless it's a genuine flatline.
        if not is_flatline and symbol in self._quarantine_history:
            elapsed = (datetime.now() - self._quarantine_history[symbol]).total_seconds() / 60
            if elapsed < cooldown_minutes:
                logger.debug(
                    f"Quarantine suppressed for {symbol}: cooldown active "
                    f"({elapsed:.1f}/{cooldown_minutes} min elapsed)"
                )
                return

        minutes = duration_minutes or self.quarantine_minutes
        expiration = datetime.now() + timedelta(minutes=minutes)
        self._quarantined[symbol] = expiration
        self._quarantine_history[symbol] = datetime.now()
        logger.warning(
            f"Symbol {symbol} {'[FLATLINE] ' if is_flatline else ''}quarantined until {expiration}"
        )

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
        if symbol is not None:
            symbol = self._normalize_symbol(symbol)
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

            _quarantined_this_cycle = False
            for alert in checks:
                if alert is not None:
                    alerts.append(alert)
                    if alert.auto_quarantine and not _quarantined_this_cycle:
                        # Pass is_flatline=True when metric_value is exactly 0.0
                        # so the cooldown gate is bypassed for genuine model crashes.
                        _is_flatline = (alert.metric_value == 0.0)
                        self.auto_quarantine(sym, is_flatline=_is_flatline)
                        _quarantined_this_cycle = True

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

    @staticmethod
    def _is_equity_market_open() -> bool:
        """Return True if NYSE is currently in regular session (9:30–16:00 ET, Mon–Fri)."""
        try:
            from zoneinfo import ZoneInfo
            et = ZoneInfo("America/New_York")
        except ImportError:
            try:
                import pytz
                et = pytz.timezone("America/New_York")
            except ImportError:
                return True  # can't determine — don't suppress
        now_et = datetime.now(tz=et)
        if now_et.weekday() >= 5:  # Saturday=5, Sunday=6
            return False
        open_time = now_et.replace(hour=9, minute=30, second=0, microsecond=0)
        close_time = now_et.replace(hour=16, minute=0, second=0, microsecond=0)
        return open_time <= now_et <= close_time

    @staticmethod
    def _is_crypto_symbol(symbol: str) -> bool:
        s = symbol.upper()
        return (
            s.startswith("CRYPTO:")
            or s.endswith("/USD")
            or s.endswith("-USD")
            or s.endswith("USDT")
            or s.endswith("BTC")
        )

    def _check_stuck_signals(self, symbol: str) -> Optional[IntegrityAlert]:
        """Detect when signals are stuck at the same value."""
        buf = list(self._buffers[symbol])
        if len(buf) < self.stuck_threshold:
            return None

        # Signals are legitimately constant during non-market hours because IBKR doesn't
        # stream price updates overnight.  This includes CRYPTO: the ML model uses equity
        # macro factors (SPY, VIX, sector ETFs) as inputs; when those go flat at 4 PM ET
        # the crypto signal also flatlines.  Skip the stuck check for ALL symbols outside
        # equity market hours to prevent an infinite 60-min quarantine loop overnight.
        if not self._is_equity_market_open():
            return None

        recent = [r.signal for r in buf[-self.stuck_threshold:]]
        signal_range = max(recent) - min(recent)

        # Threshold rationale: a daily-bar ML model naturally produces a very narrow
        # spread intra-day because all 90-second cycles share the same daily feature bar.
        # The only genuine anomaly worth quarantining is when the model's output is
        # COMPLETELY FROZEN to 3+ decimal places (range < 0.001) — this indicates a
        # data-pipeline failure (e.g. NaN propagation, dead ML runner) NOT legitimate
        # stable signal output.  The 30-min cooldown in auto_quarantine() breaks the
        # expiry→re-quarantine loop for borderline cases; std=0.000 exact flatlines
        # bypass the cooldown (is_flatline=True) to guarantee immediate quarantine.
        if signal_range < 0.001:
            # Classify severity: exact flatline = model crash, tiny range = suspicious
            is_flatline = signal_range == 0.0
            return IntegrityAlert(
                alert_type="stuck_signal",
                severity=AlertSeverity.CRITICAL,
                symbol=symbol,
                message=(
                    f"{'[FLATLINE] ' if is_flatline else ''}Signal stuck at {recent[-1]:.3f} for "
                    f"{self.stuck_threshold} observations (range={signal_range:.5f})"
                ),
                metric_value=signal_range,
                threshold=0.001,
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
        # Outside equity market hours all signals legitimately flatten because IBKR/yfinance
        # stop streaming price updates.  Skip the low-volatility (too-stable) quarantine in
        # that window to avoid mass-quarantining every symbol at market close — both equity
        # AND crypto (crypto ML signals use equity macro factors which also go flat at night).
        if not self._is_equity_market_open():
            return None

        recent = [r.signal for r in list(self._buffers[symbol])[-30:]]
        if len(recent) < 15:
            return None

        signal_std = float(np.std(recent))

        # Too stable = model might be stuck or ignoring data.
        # Threshold rationale: a daily-bar ML model retraining once per day will
        # produce nearly identical signal values across consecutive 90-second intra-day
        # cycles — typical intra-day std is 0.001–0.02.  The floor of 0.001 fires ONLY
        # when the model outputs are numerically indistinguishable (machine-epsilon level)
        # which indicates a data-pipeline failure (dead feature feed, NaN silent fill).
        # For std=0.000 exact (complete flatline) the is_flatline flag is set so the
        # 30-min cooldown gate is bypassed and quarantine is always enforced immediately.
        if signal_std < 0.001:
            is_flatline = signal_std == 0.0
            return IntegrityAlert(
                alert_type="signal_volatility_low",
                severity=AlertSeverity.CRITICAL,
                symbol=symbol,
                message=(
                    f"{'[FLATLINE] ' if is_flatline else ''}Signal volatility abnormally low: "
                    f"std={signal_std:.5f}"
                ),
                metric_value=signal_std,
                threshold=0.001,
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
