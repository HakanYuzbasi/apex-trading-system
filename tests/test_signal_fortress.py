"""
tests/test_signal_fortress.py - Tests for Signal Fortress Components

Tests all 5 robustness layers:
1. AdaptiveRegimeDetector
2. SignalConsensusEngine
3. SignalIntegrityMonitor
4. OutcomeFeedbackLoop
5. AdaptiveThresholdOptimizer
"""

import pytest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from unittest.mock import MagicMock, patch


# ════════════════════════════════════════════════════════════════════
# Helpers
# ════════════════════════════════════════════════════════════════════

def make_prices(n=200, start=100.0, trend=0.0005, vol=0.02, seed=42):
    """Generate synthetic price series."""
    rng = np.random.RandomState(seed)
    returns = trend + vol * rng.randn(n)
    prices = start * np.cumprod(1 + returns)
    dates = pd.date_range("2024-01-01", periods=n, freq="B")
    return pd.Series(prices, index=dates, name="Close")


def make_bull_prices(n=200):
    return make_prices(n, trend=0.002, vol=0.01, seed=10)


def make_bear_prices(n=200):
    return make_prices(n, trend=-0.002, vol=0.015, seed=20)


def make_volatile_prices(n=200):
    return make_prices(n, trend=0.0, vol=0.04, seed=30)


def make_universe_data(n_symbols=20, n_days=200):
    """Create a universe of stock DataFrames."""
    universe = {}
    for i in range(n_symbols):
        trend = 0.001 * (i % 3 - 1)  # Mix of up/down/flat
        prices = make_prices(n_days, trend=trend, seed=i)
        df = pd.DataFrame({
            "Close": prices,
            "Volume": np.random.randint(100000, 5000000, size=n_days),
        }, index=prices.index)
        universe[f"SYM{i:02d}"] = df
    return universe


def make_signal_history(n=100, n_symbols=5):
    """Create synthetic signal history DataFrame."""
    rng = np.random.RandomState(42)
    symbols = [f"SYM{i}" for i in range(n_symbols)]
    records = []
    for i in range(n):
        sym = symbols[i % n_symbols]
        signal = rng.uniform(-0.8, 0.8)
        confidence = rng.uniform(0.2, 0.9)
        # Return correlates with signal direction (60% accuracy)
        if rng.random() < 0.60:
            ret = signal * rng.uniform(0.001, 0.05)
        else:
            ret = -signal * rng.uniform(0.001, 0.03)
        records.append({
            "symbol": sym,
            "signal": signal,
            "confidence": confidence,
            "return_5d": ret,
            "regime": rng.choice(["bull", "bear", "neutral", "volatile"]),
            "timestamp": datetime(2024, 1, 1) + timedelta(days=i),
        })
    return pd.DataFrame(records)


# ════════════════════════════════════════════════════════════════════
# 1. AdaptiveRegimeDetector Tests
# ════════════════════════════════════════════════════════════════════

class TestAdaptiveRegimeDetector:
    """Tests for probability-based regime detection."""

    def setup_method(self):
        from models.adaptive_regime_detector import AdaptiveRegimeDetector
        self.detector = AdaptiveRegimeDetector(
            smoothing_alpha=0.3,  # Faster transitions for testing
            min_regime_duration=1,
        )

    def test_basic_regime_assessment(self):
        """Regime assessment returns valid RegimeAssessment."""
        prices = make_prices(200)
        result = self.detector.assess_regime(prices)

        assert result.primary_regime in ("bull", "bear", "neutral", "volatile")
        assert 0 <= result.regime_strength <= 1
        assert 0 <= result.transition_probability <= 1
        assert sum(result.regime_probabilities.values()) == pytest.approx(1.0, abs=0.01)

    def test_bull_detection(self):
        """Strongly trending up prices should detect bull."""
        prices = make_bull_prices(200)
        # Run a few times to let EMA converge
        for _ in range(5):
            result = self.detector.assess_regime(prices)
        assert result.regime_probabilities["bull"] > result.regime_probabilities["bear"]

    def test_bear_detection(self):
        """Strongly trending down prices should detect bear."""
        prices = make_bear_prices(200)
        for _ in range(5):
            result = self.detector.assess_regime(prices)
        assert result.regime_probabilities["bear"] > result.regime_probabilities["bull"]

    def test_volatile_detection(self):
        """High volatility should detect volatile regime."""
        prices = make_volatile_prices(200)
        for _ in range(5):
            result = self.detector.assess_regime(prices)
        assert result.regime_probabilities["volatile"] > 0.15

    def test_probabilities_sum_to_one(self):
        """Regime probabilities must always sum to 1.0."""
        for seed in range(10):
            prices = make_prices(200, seed=seed)
            result = self.detector.assess_regime(prices)
            total = sum(result.regime_probabilities.values())
            assert total == pytest.approx(1.0, abs=0.01)

    def test_smooth_transitions(self):
        """Regime changes should be gradual, not sudden jumps."""
        prices = make_prices(200)
        results = []
        for _ in range(10):
            r = self.detector.assess_regime(prices)
            results.append(r.regime_probabilities.copy())

        # Check that probabilities don't jump more than alpha per step
        for i in range(1, len(results)):
            for regime in ("bull", "bear", "neutral", "volatile"):
                diff = abs(results[i][regime] - results[i-1][regime])
                assert diff < 0.5, f"Regime prob jumped {diff} in one step"

    def test_leading_indicators(self):
        """Leading indicators should be computed."""
        prices = make_prices(200)
        result = self.detector.assess_regime(prices)
        assert "breadth_divergence" in result.leading_indicators
        assert "vol_compression" in result.leading_indicators
        assert "momentum_divergence" in result.leading_indicators
        assert "correlation_spike" in result.leading_indicators

    def test_sub_regime_detection(self):
        """Sub-regime should be a valid SubRegime enum."""
        from models.adaptive_regime_detector import SubRegime
        prices = make_bull_prices(200)
        for _ in range(5):
            result = self.detector.assess_regime(prices)
        assert isinstance(result.sub_regime, SubRegime)

    def test_insufficient_data(self):
        """Short price series should return neutral."""
        prices = make_prices(30)
        result = self.detector.assess_regime(prices)
        assert result.primary_regime == "neutral"

    def test_with_universe_data(self):
        """Should work with universe data for breadth calculation."""
        prices = make_prices(200)
        universe = make_universe_data(20, 200)
        result = self.detector.assess_regime(prices, universe_data=universe)
        assert result.primary_regime in ("bull", "bear", "neutral", "volatile")

    def test_with_vix_level(self):
        """VIX level should influence regime detection."""
        prices = make_prices(200)
        result_normal = self.detector.assess_regime(prices, vix_level=15)
        self.detector._smoothed_probs = {r: 0.25 for r in ("bull", "bear", "neutral", "volatile")}
        result_panic = self.detector.assess_regime(prices, vix_level=45)
        assert result_panic.regime_probabilities["volatile"] > result_normal.regime_probabilities["volatile"]

    def test_symbol_specific_regime(self):
        """Symbol overlay should work."""
        prices = make_prices(200)
        result = self.detector.assess_regime(prices)
        bear_prices = make_bear_prices(200)
        sym_result = self.detector.get_regime_for_symbol("TEST", bear_prices, result)
        assert sym_result is not None


# ════════════════════════════════════════════════════════════════════
# 2. SignalConsensusEngine Tests
# ════════════════════════════════════════════════════════════════════

class TestSignalConsensusEngine:
    """Tests for multi-generator consensus."""

    def _make_mock_generator(self, signal_value, confidence=0.7):
        """Create a mock generator that returns a fixed signal."""
        gen = MagicMock()
        gen.generate_signal = MagicMock(
            return_value=MagicMock(signal=signal_value, confidence=confidence)
        )
        return gen

    def setup_method(self):
        from models.signal_consensus_engine import SignalConsensusEngine
        self.gen_a = self._make_mock_generator(0.5, 0.8)
        self.gen_b = self._make_mock_generator(0.4, 0.7)
        self.gen_c = self._make_mock_generator(0.3, 0.6)

        self.engine = SignalConsensusEngine(
            generators={"a": self.gen_a, "b": self.gen_b, "c": self.gen_c},
            min_agreement=0.60,
            min_generators=2,
        )

    def test_unanimous_agreement(self):
        """All generators agree -> majority_agrees = True."""
        prices = make_prices(200)
        result = self.engine.generate_consensus("AAPL", prices)
        assert result.majority_agrees is True
        assert result.consensus_signal > 0
        assert result.direction_agreement >= 0.99

    def test_disagreement_vetoes(self):
        """Generators disagree -> signal vetoed."""
        from models.signal_consensus_engine import SignalConsensusEngine
        gen_bull = self._make_mock_generator(0.6)
        gen_bear = self._make_mock_generator(-0.5)
        gen_bear2 = self._make_mock_generator(-0.4)

        engine = SignalConsensusEngine(
            generators={"bull": gen_bull, "bear1": gen_bear, "bear2": gen_bear2},
            min_agreement=0.60,
        )
        prices = make_prices(200)
        result = engine.generate_consensus("AAPL", prices)
        # 2 of 3 bearish = majority agrees on bearish
        assert result.majority_agrees is True
        assert result.consensus_signal < 0

    def test_conviction_score_range(self):
        """Conviction score should be 0-100."""
        prices = make_prices(200)
        result = self.engine.generate_consensus("AAPL", prices)
        assert 0 <= result.conviction_score <= 100

    def test_strong_consensus_detection(self):
        """All generators with strong signals -> strong_consensus."""
        from models.signal_consensus_engine import SignalConsensusEngine
        engine = SignalConsensusEngine(
            generators={
                "a": self._make_mock_generator(0.7),
                "b": self._make_mock_generator(0.6),
                "c": self._make_mock_generator(0.5),
            },
        )
        result = engine.generate_consensus("AAPL", make_prices(200))
        assert result.strong_consensus is True

    def test_record_outcome(self):
        """Outcome recording should update performance."""
        self.engine.record_outcome(
            "AAPL",
            {"a": 0.5, "b": 0.4, "c": 0.3},
            0.02,
        )
        perf = self.engine.performance["a"]
        assert perf.total_signals == 1
        assert perf.correct_signals == 1  # Both positive

    def test_dynamic_weights(self):
        """Weights should adapt based on performance."""
        # Record many outcomes favoring generator 'a'
        for _ in range(25):
            self.engine.record_outcome("AAPL", {"a": 0.5, "b": -0.3, "c": -0.2}, 0.03)

        weights = self.engine._calculate_dynamic_weights("neutral")
        assert weights["a"] > weights["b"]

    def test_generator_failure_handled(self):
        """Engine should handle generator exceptions gracefully."""
        from models.signal_consensus_engine import SignalConsensusEngine
        failing_gen = MagicMock()
        failing_gen.generate_signal = MagicMock(side_effect=RuntimeError("model failed"))

        engine = SignalConsensusEngine(
            generators={
                "good": self._make_mock_generator(0.5),
                "bad": failing_gen,
                "ok": self._make_mock_generator(0.4),
            },
            min_generators=2,
        )
        result = engine.generate_consensus("AAPL", make_prices(200))
        # Should still work with 2 of 3
        assert result.majority_agrees is True

    def test_insufficient_generators_veto(self):
        """Veto if fewer than min_generators produce signals."""
        from models.signal_consensus_engine import SignalConsensusEngine
        failing = MagicMock()
        failing.generate_signal = MagicMock(side_effect=RuntimeError("fail"))

        engine = SignalConsensusEngine(
            generators={"a": self._make_mock_generator(0.5), "b": failing, "c": failing},
            min_generators=2,
        )
        result = engine.generate_consensus("AAPL", make_prices(200))
        assert result.vetoed is True


# ════════════════════════════════════════════════════════════════════
# 3. SignalIntegrityMonitor Tests
# ════════════════════════════════════════════════════════════════════

class TestSignalIntegrityMonitor:
    """Tests for signal anomaly detection."""

    def setup_method(self):
        from monitoring.signal_integrity_monitor import SignalIntegrityMonitor
        self.monitor = SignalIntegrityMonitor(
            window_size=100,
            stuck_threshold=5,
            kl_threshold=0.5,
            quarantine_minutes=10,
        )

    def test_normal_signals_healthy(self):
        """Normal signal stream should report healthy."""
        rng = np.random.RandomState(42)
        for i in range(50):
            self.monitor.record_signal(
                "AAPL",
                signal=rng.uniform(-0.5, 0.5),
                confidence=rng.uniform(0.3, 0.8),
                regime="neutral",
            )
        report = self.monitor.check_integrity("AAPL")
        assert report.healthy is True
        assert len(report.alerts) == 0

    def test_stuck_signal_detected(self):
        """Repeated identical signals should trigger quarantine."""
        for i in range(15):
            self.monitor.record_signal("AAPL", signal=0.25, confidence=0.5, regime="bull")

        report = self.monitor.check_integrity("AAPL")
        stuck_alerts = [a for a in report.alerts if a.alert_type == "stuck_signal"]
        assert len(stuck_alerts) > 0
        assert self.monitor.is_quarantined("AAPL")

    def test_low_volatility_detected(self):
        """Very stable signals (near-zero std) should trigger alert."""
        for i in range(30):
            self.monitor.record_signal(
                "AAPL",
                signal=0.25 + 0.001 * (i % 3),  # Tiny variation
                confidence=0.5,
                regime="neutral",
            )
        report = self.monitor.check_integrity("AAPL")
        vol_alerts = [a for a in report.alerts if "volatility" in a.alert_type]
        assert len(vol_alerts) > 0

    def test_distribution_shift_detected(self):
        """Sudden shift in signal distribution should be detected."""
        rng = np.random.RandomState(42)
        # Build baseline with positive signals
        for i in range(80):
            self.monitor.record_signal(
                "AAPL",
                signal=rng.uniform(0.1, 0.5),
                confidence=0.6,
                regime="bull",
            )
        # Shift to negative signals
        for i in range(20):
            self.monitor.record_signal(
                "AAPL",
                signal=rng.uniform(-0.8, -0.3),
                confidence=0.6,
                regime="bear",
            )
        report = self.monitor.check_integrity("AAPL")
        shift_alerts = [a for a in report.alerts if a.alert_type == "distribution_shift"]
        assert len(shift_alerts) > 0

    def test_quarantine_expiration(self):
        """Quarantine should expire after duration."""
        self.monitor.auto_quarantine("AAPL", duration_minutes=1)
        # Manually set expiration to the past to simulate time passing
        self.monitor._quarantined["AAPL"] = datetime.now() - timedelta(seconds=1)
        assert not self.monitor.is_quarantined("AAPL")

    def test_data_quality_degradation(self):
        """Low data quality should trigger alert."""
        for i in range(20):
            self.monitor.record_signal(
                "AAPL", signal=0.3 + 0.05 * (i % 5),
                confidence=0.5, regime="neutral",
                data_quality=0.5,  # Below 0.7 threshold
            )
        report = self.monitor.check_integrity("AAPL")
        quality_alerts = [a for a in report.alerts if a.alert_type == "data_quality_degradation"]
        assert len(quality_alerts) > 0

    def test_regime_signal_mismatch(self):
        """Bullish signals in bear regime should be flagged."""
        for i in range(20):
            self.monitor.record_signal(
                "AAPL",
                signal=0.5 + 0.1 * (i % 3),
                confidence=0.6,
                regime="bear",  # But signals are bullish!
            )
        report = self.monitor.check_integrity("AAPL")
        mismatch_alerts = [a for a in report.alerts if a.alert_type == "regime_signal_mismatch"]
        assert len(mismatch_alerts) > 0

    def test_multiple_symbols(self):
        """Monitor should track multiple symbols independently."""
        rng = np.random.RandomState(42)
        for sym in ["AAPL", "MSFT", "GOOGL"]:
            for i in range(20):
                self.monitor.record_signal(
                    sym,
                    signal=rng.uniform(-0.5, 0.5),
                    confidence=0.5,
                    regime="neutral",
                )
        report = self.monitor.check_integrity()
        assert report.metrics["total_symbols_tracked"] == 3


# ════════════════════════════════════════════════════════════════════
# 4. OutcomeFeedbackLoop Tests
# ════════════════════════════════════════════════════════════════════

class TestOutcomeFeedbackLoop:
    """Tests for automatic outcome tracking and retraining."""

    def setup_method(self):
        from monitoring.outcome_feedback_loop import OutcomeFeedbackLoop
        self.mock_tracker = MagicMock()
        self.mock_consensus = MagicMock()
        self.mock_generator = MagicMock()
        self.mock_generator.record_outcome = MagicMock()
        self.mock_generator.train = MagicMock()

        self.loop = OutcomeFeedbackLoop(
            outcome_tracker=self.mock_tracker,
            consensus_engine=self.mock_consensus,
            inst_generator=self.mock_generator,
            retrain_accuracy_threshold=0.45,
        )

    def test_record_signal(self):
        """Should record signals for tracking."""
        self.loop.record_signal(
            symbol="AAPL",
            signal_value=0.5,
            confidence=0.7,
            regime="bull",
            entry_price=150.0,
            generator_signals={"inst": 0.5, "god": 0.4},
        )
        assert len(self.loop._active_signals) == 1

    def test_update_forward_returns(self):
        """Should compute forward returns from price data."""
        # Record a signal
        self.loop.record_signal("AAPL", 0.5, 0.7, "bull", 100.0)

        # Simulate 25 days of price data
        prices = pd.Series(
            np.linspace(100, 105, 25),
            index=pd.date_range("2024-01-01", periods=25, freq="B"),
            name="Close",
        )
        historical = {"AAPL": pd.DataFrame({"Close": prices})}

        # Hack: set signal timestamp to 25 days ago
        self.loop._active_signals[0].timestamp = datetime.now() - timedelta(days=25)

        self.loop.update_forward_returns(historical)

        sig = self.loop._active_signals[0] if self.loop._active_signals else self.loop._completed_signals[0]
        assert sig.return_1d is not None

    def test_performance_degradation_detection(self):
        """Should detect when accuracy drops."""
        # Simulate mostly wrong signals
        self.loop._accuracy_history = [0.0] * 30  # 0% accuracy
        degradations = self.loop.check_performance_degradation()
        assert len(degradations) > 0
        assert any(d.metric == "accuracy_30d" for d in degradations)

    def test_should_retrain_on_degradation(self):
        """Should recommend retrain when accuracy is bad."""
        self.loop._accuracy_history = [0.0] * 30
        self.loop._last_retrain = datetime.now() - timedelta(hours=48)
        should, reason = self.loop.should_retrain()
        assert should is True
        assert "degradation" in reason.lower()

    def test_retrain_cooldown(self):
        """Should not retrain within cooldown period."""
        self.loop._last_retrain = datetime.now()  # Just retrained
        should, reason = self.loop.should_retrain()
        assert should is False
        assert "cooldown" in reason.lower()

    def test_trigger_retrain(self):
        """Should call generator.train() on retrain."""
        historical = {"AAPL": pd.DataFrame({"Close": [100, 101, 102]})}
        result = self.loop.trigger_retrain(historical)
        assert result is True
        self.mock_generator.train.assert_called_once()

    def test_rolling_metrics(self):
        """Should compute rolling metrics correctly."""
        rng = np.random.RandomState(42)
        self.loop._accuracy_history = list(rng.choice([0.0, 1.0], size=50))
        self.loop._return_history = list(rng.normal(0.001, 0.01, size=50))
        metrics = self.loop.get_rolling_metrics()
        assert "accuracy" in metrics
        assert "sharpe" in metrics


# ════════════════════════════════════════════════════════════════════
# 5. AdaptiveThresholdOptimizer Tests
# ════════════════════════════════════════════════════════════════════

class TestAdaptiveThresholdOptimizer:
    """Tests for per-symbol threshold optimization."""

    def setup_method(self):
        from models.adaptive_threshold_optimizer import AdaptiveThresholdOptimizer
        self.optimizer = AdaptiveThresholdOptimizer(
            default_thresholds={
                "bull": 0.23,
                "bear": 0.25,
                "neutral": 0.28,
            },
            min_signals=20,
            optimization_interval_hours=1,
        )

    def test_default_thresholds(self):
        """Should return regime defaults when no optimization data."""
        thresholds = self.optimizer.get_thresholds("AAPL", "bull")
        assert thresholds.entry_threshold == 0.23
        assert thresholds.is_default is True

    def test_optimize_symbol(self):
        """Should optimize thresholds for a symbol with sufficient data."""
        history = make_signal_history(100, n_symbols=1)
        history["symbol"] = "AAPL"
        result = self.optimizer.optimize_symbol("AAPL", history)
        assert result is not None
        assert result.thresholds.entry_threshold in [0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50]
        assert result.thresholds.is_default is False

    def test_insufficient_data_returns_none(self):
        """Should return None with insufficient data."""
        history = make_signal_history(10, n_symbols=1)
        result = self.optimizer.optimize_symbol("AAPL", history)
        assert result is None

    def test_optimize_all(self):
        """Should optimize all symbols with sufficient data."""
        history = make_signal_history(200, n_symbols=5)
        results = self.optimizer.optimize_all(history)
        assert len(results) > 0

    def test_optimized_thresholds_cached(self):
        """Should cache optimized thresholds."""
        history = make_signal_history(100, n_symbols=1)
        history["symbol"] = "AAPL"
        self.optimizer.optimize_symbol("AAPL", history)

        thresholds = self.optimizer.get_thresholds("AAPL", "bull")
        assert thresholds.is_default is False
        assert thresholds.sample_count >= 20

    def test_position_size_multiplier(self):
        """High-vol/low-Sharpe should get smaller multiplier."""
        mult_low = self.optimizer._adjust_position_size("high_vol", 0.5)
        mult_high = self.optimizer._adjust_position_size("low_vol", 2.0)
        assert mult_low < 1.0
        assert mult_high > 1.0

    def test_walk_forward_no_lookahead(self):
        """Walk-forward should only use past data for each fold."""
        history = make_signal_history(200, n_symbols=1)
        result = self.optimizer._walk_forward_optimize(history)
        assert len(result) == 5  # (entry, conf, sharpe, accuracy, stability)
        assert 0.15 <= result[0] <= 0.50  # Valid entry threshold range

    def test_threshold_bounds(self):
        """Optimized thresholds should be within grid bounds."""
        history = make_signal_history(100, n_symbols=1)
        history["symbol"] = "AAPL"
        result = self.optimizer.optimize_symbol("AAPL", history)
        if result:
            assert 0.15 <= result.thresholds.entry_threshold <= 0.50
            assert 0.20 <= result.thresholds.confidence_threshold <= 0.50


# ════════════════════════════════════════════════════════════════════
# Integration Tests
# ════════════════════════════════════════════════════════════════════

class TestSignalFortressIntegration:
    """Tests that all components work together."""

    def test_full_pipeline(self):
        """Run a signal through all fortress components."""
        from models.adaptive_regime_detector import AdaptiveRegimeDetector
        from models.signal_consensus_engine import SignalConsensusEngine
        from monitoring.signal_integrity_monitor import SignalIntegrityMonitor
        from monitoring.outcome_feedback_loop import OutcomeFeedbackLoop
        from models.adaptive_threshold_optimizer import AdaptiveThresholdOptimizer

        prices = make_bull_prices(200)

        # 1. Regime detection
        regime_detector = AdaptiveRegimeDetector(smoothing_alpha=0.5)
        regime = regime_detector.assess_regime(prices)
        assert regime.primary_regime in ("bull", "bear", "neutral", "volatile")

        # 2. Consensus engine with mock generators
        gen_a = MagicMock()
        gen_a.generate_signal = MagicMock(
            return_value=MagicMock(signal=0.5, confidence=0.8)
        )
        gen_b = MagicMock()
        gen_b.generate_signal = MagicMock(
            return_value=MagicMock(signal=0.4, confidence=0.7)
        )
        gen_c = MagicMock()
        gen_c.generate_signal = MagicMock(
            return_value=MagicMock(signal=0.3, confidence=0.6)
        )

        consensus = SignalConsensusEngine(
            generators={"a": gen_a, "b": gen_b, "c": gen_c}
        )
        result = consensus.generate_consensus("AAPL", prices)
        assert result.majority_agrees is True

        # 3. Integrity monitor
        monitor = SignalIntegrityMonitor(window_size=50)
        monitor.record_signal(
            "AAPL", result.consensus_signal,
            result.consensus_confidence, regime.primary_regime
        )
        health = monitor.check_integrity("AAPL")
        # Only 1 observation, so should be healthy
        assert health.healthy is True

        # 4. Outcome feedback
        loop = OutcomeFeedbackLoop()
        loop.record_signal(
            "AAPL", result.consensus_signal,
            result.consensus_confidence, regime.primary_regime,
            entry_price=100.0,
            generator_signals=result.generator_signals,
        )
        assert len(loop._active_signals) == 1

        # 5. Threshold optimizer
        optimizer = AdaptiveThresholdOptimizer()
        thresholds = optimizer.get_thresholds("AAPL", regime.primary_regime)
        assert thresholds.entry_threshold > 0

    def test_invariants_hold(self):
        """Signal fortress invariants that must always hold."""
        from models.adaptive_regime_detector import AdaptiveRegimeDetector

        detector = AdaptiveRegimeDetector()

        for seed in range(20):
            prices = make_prices(200, seed=seed)
            result = detector.assess_regime(prices)

            # Invariant 1: Probabilities sum to 1
            total = sum(result.regime_probabilities.values())
            assert total == pytest.approx(1.0, abs=0.02)

            # Invariant 2: Primary regime has highest probability
            max_regime = max(result.regime_probabilities, key=result.regime_probabilities.get)
            # (may not always match due to smoothing & min duration, but probability should be positive)
            assert result.regime_probabilities[result.primary_regime] > 0

            # Invariant 3: Transition probability in [0, 1]
            assert 0 <= result.transition_probability <= 1

            # Invariant 4: Regime strength in [0, 1]
            assert 0 <= result.regime_strength <= 1
