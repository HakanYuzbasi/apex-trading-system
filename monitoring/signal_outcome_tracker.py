"""
monitoring/signal_outcome_tracker.py

Forward-Looking Signal Performance Tracker

Tracks and evaluates signal quality by measuring actual price movements
after signal generation. Provides:
- Forward return labels (5-day, 10-day, 20-day)
- Maximum Favorable Excursion (MFE) - best price during holding period
- Maximum Adverse Excursion (MAE) - worst price during holding period
- Time-to-target metrics
- Signal quality scoring based on outcomes
"""

import logging
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional
from dataclasses import dataclass, field, asdict
from collections import defaultdict
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class SignalOutcome:
    """Represents a signal with its forward-looking outcome metrics."""

    # Signal identification
    signal_id: str
    symbol: str
    signal_time: datetime

    # Signal characteristics
    signal_value: float  # [-1, 1]
    signal_direction: str  # BUY, SELL, NEUTRAL
    confidence: float
    regime: str  # bull, bear, neutral, volatile

    # Price at signal generation
    entry_price: float

    # Forward returns (filled asynchronously as data becomes available)
    return_1d: Optional[float] = None
    return_5d: Optional[float] = None
    return_10d: Optional[float] = None
    return_20d: Optional[float] = None

    # MFE/MAE metrics (within different windows)
    mfe_5d: Optional[float] = None  # Max favorable excursion in 5 days (%)
    mae_5d: Optional[float] = None  # Max adverse excursion in 5 days (%)
    mfe_10d: Optional[float] = None
    mae_10d: Optional[float] = None
    mfe_20d: Optional[float] = None
    mae_20d: Optional[float] = None

    # Time-to-target metrics (days to reach X% return)
    time_to_2pct: Optional[int] = None
    time_to_5pct: Optional[int] = None
    time_to_10pct: Optional[int] = None

    # Outcome labels (filled when window completes)
    hit_5pct_5d: Optional[bool] = None  # Did price increase 5% within 5 days?
    hit_10pct_10d: Optional[bool] = None
    hit_target_within_window: Optional[bool] = None

    # Signal quality score (computed from outcomes)
    quality_score: Optional[float] = None

    # Metadata
    features_hash: Optional[str] = None
    model_version: Optional[str] = None
    is_complete: bool = False  # True when all windows have passed

    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization."""
        d = asdict(self)
        d['signal_time'] = self.signal_time.isoformat()
        return d

    @classmethod
    def from_dict(cls, data: Dict) -> 'SignalOutcome':
        """Create from dictionary."""
        data['signal_time'] = datetime.fromisoformat(data['signal_time'])
        return cls(**data)


@dataclass
class SignalQualityMetrics:
    """Aggregated quality metrics for a set of signals."""

    total_signals: int = 0

    # Directional accuracy
    accuracy_1d: float = 0.0
    accuracy_5d: float = 0.0
    accuracy_10d: float = 0.0
    accuracy_20d: float = 0.0

    # Average returns by signal direction
    avg_return_5d_buy: float = 0.0
    avg_return_5d_sell: float = 0.0
    avg_return_10d_buy: float = 0.0
    avg_return_10d_sell: float = 0.0

    # MFE/MAE statistics
    avg_mfe_10d: float = 0.0
    avg_mae_10d: float = 0.0
    mfe_mae_ratio: float = 0.0  # Higher is better

    # Target hit rates
    hit_rate_5pct_5d: float = 0.0
    hit_rate_10pct_10d: float = 0.0

    # Time-to-target statistics
    avg_time_to_2pct: float = 0.0
    avg_time_to_5pct: float = 0.0

    # Quality distribution
    high_quality_pct: float = 0.0  # % of signals with quality > 0.7
    low_quality_pct: float = 0.0   # % of signals with quality < 0.3

    # By regime
    accuracy_by_regime: Dict[str, float] = field(default_factory=dict)

    # By confidence bucket
    accuracy_by_confidence: Dict[str, float] = field(default_factory=dict)


class SignalOutcomeTracker:
    """
    Tracks signal outcomes for forward-looking performance analysis.

    Usage:
        tracker = SignalOutcomeTracker()

        # Record a signal when generated
        tracker.record_signal(
            symbol="AAPL",
            signal_value=0.75,
            confidence=0.85,
            entry_price=150.0,
            regime="bull"
        )

        # Update outcomes periodically (e.g., daily)
        tracker.update_outcomes(price_data)

        # Get quality metrics
        metrics = tracker.get_quality_metrics()
    """

    def __init__(
        self,
        data_dir: str = "data",
        lookback_windows: List[int] = None,
        target_returns: List[float] = None
    ):
        """
        Initialize the outcome tracker.

        Args:
            data_dir: Directory for persistence
            lookback_windows: Days to track (default: [1, 5, 10, 20])
            target_returns: Return targets to track (default: [0.02, 0.05, 0.10])
        """
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(exist_ok=True)

        self.lookback_windows = lookback_windows or [1, 5, 10, 20]
        self.target_returns = target_returns or [0.02, 0.05, 0.10]

        # Active signals being tracked (not yet complete)
        self.active_signals: Dict[str, SignalOutcome] = {}

        # Completed signals (all windows passed)
        self.completed_signals: List[SignalOutcome] = []

        # Signal history by symbol for quick lookup
        self.signals_by_symbol: Dict[str, List[str]] = defaultdict(list)

        # Load existing data
        self._load_state()

        logger.info("SignalOutcomeTracker initialized")
        logger.info(f"  Lookback windows: {self.lookback_windows} days")
        logger.info(f"  Target returns: {[f'{t*100}%' for t in self.target_returns]}")
        logger.info(f"  Active signals: {len(self.active_signals)}")
        logger.info(f"  Completed signals: {len(self.completed_signals)}")

    def record_signal(
        self,
        symbol: str,
        signal_value: float,
        confidence: float,
        entry_price: float,
        regime: str = "unknown",
        features_hash: Optional[str] = None,
        model_version: Optional[str] = None
    ) -> str:
        """
        Record a new signal for outcome tracking.

        Args:
            symbol: Stock symbol
            signal_value: Signal strength [-1, 1]
            confidence: Signal confidence [0, 1]
            entry_price: Price at signal generation
            regime: Market regime
            features_hash: Hash of features for reproducibility
            model_version: Model version that generated the signal

        Returns:
            signal_id: Unique identifier for this signal
        """
        signal_time = datetime.now()
        signal_id = f"{symbol}_{signal_time.strftime('%Y%m%d_%H%M%S')}_{abs(hash(signal_time))%10000}"

        # Determine direction
        if signal_value > 0.1:
            direction = "BUY"
        elif signal_value < -0.1:
            direction = "SELL"
        else:
            direction = "NEUTRAL"

        outcome = SignalOutcome(
            signal_id=signal_id,
            symbol=symbol,
            signal_time=signal_time,
            signal_value=signal_value,
            signal_direction=direction,
            confidence=confidence,
            regime=regime,
            entry_price=entry_price,
            features_hash=features_hash,
            model_version=model_version
        )

        self.active_signals[signal_id] = outcome
        self.signals_by_symbol[symbol].append(signal_id)

        logger.debug(f"Signal recorded: {signal_id} | {symbol} {direction} @ ${entry_price:.2f}")

        return signal_id

    def update_outcomes(self, price_data: Dict[str, pd.DataFrame]) -> int:
        """
        Update outcome metrics for all active signals using historical price data.

        Args:
            price_data: Dict of symbol -> DataFrame with 'close' column and datetime index

        Returns:
            Number of signals updated
        """
        updated_count = 0
        completed_ids = []

        for signal_id, outcome in self.active_signals.items():
            symbol = outcome.symbol

            if symbol not in price_data:
                continue

            df = price_data[symbol]
            if df.empty or 'close' not in df.columns:
                continue

            # Get prices after signal time
            signal_date = outcome.signal_time.date()
            future_prices = df[df.index.date > signal_date]['close']

            if len(future_prices) == 0:
                continue

            updated = self._calculate_outcomes(outcome, future_prices)
            if updated:
                updated_count += 1

            # Check if signal is complete (max window has passed)
            days_since_signal = (datetime.now() - outcome.signal_time).days
            if days_since_signal >= max(self.lookback_windows):
                outcome.is_complete = True
                outcome.quality_score = self._calculate_quality_score(outcome)
                completed_ids.append(signal_id)

        # Move completed signals to completed list
        for signal_id in completed_ids:
            self.completed_signals.append(self.active_signals.pop(signal_id))

        if updated_count > 0:
            self._save_state()
            logger.info(f"Updated {updated_count} signal outcomes, {len(completed_ids)} completed")

        return updated_count

    def _calculate_outcomes(self, outcome: SignalOutcome, future_prices: pd.Series) -> bool:
        """Calculate forward-looking metrics for a signal."""
        entry_price = outcome.entry_price
        is_buy = outcome.signal_direction == "BUY"

        prices = future_prices.values
        n_days = len(prices)

        if n_days == 0:
            return False

        # Calculate returns at each window
        for window in self.lookback_windows:
            if n_days >= window:
                window_return = (prices[window - 1] / entry_price - 1)
                # Adjust for direction (positive return = good for buy, bad for sell)
                if not is_buy:
                    window_return = -window_return

                if window == 1:
                    outcome.return_1d = window_return
                elif window == 5:
                    outcome.return_5d = window_return
                elif window == 10:
                    outcome.return_10d = window_return
                elif window == 20:
                    outcome.return_20d = window_return

        # Calculate MFE/MAE for each window
        for window in [5, 10, 20]:
            if n_days >= window:
                window_prices = prices[:window]

                if is_buy:
                    # For long: MFE is max price, MAE is min price
                    mfe = (max(window_prices) / entry_price - 1)
                    mae = (min(window_prices) / entry_price - 1)
                else:
                    # For short: MFE is min price (profit), MAE is max price (loss)
                    mfe = (entry_price / min(window_prices) - 1)
                    mae = -(max(window_prices) / entry_price - 1)

                if window == 5:
                    outcome.mfe_5d = mfe
                    outcome.mae_5d = mae
                elif window == 10:
                    outcome.mfe_10d = mfe
                    outcome.mae_10d = mae
                elif window == 20:
                    outcome.mfe_20d = mfe
                    outcome.mae_20d = mae

        # Calculate time-to-target
        for i, price in enumerate(prices):
            if is_buy:
                ret = price / entry_price - 1
            else:
                ret = entry_price / price - 1

            days = i + 1

            if ret >= 0.02 and outcome.time_to_2pct is None:
                outcome.time_to_2pct = days
            if ret >= 0.05 and outcome.time_to_5pct is None:
                outcome.time_to_5pct = days
            if ret >= 0.10 and outcome.time_to_10pct is None:
                outcome.time_to_10pct = days

        # Calculate hit rates
        if n_days >= 5:
            outcome.hit_5pct_5d = (outcome.mfe_5d or 0) >= 0.05
        if n_days >= 10:
            outcome.hit_10pct_10d = (outcome.mfe_10d or 0) >= 0.10

        return True

    def _calculate_quality_score(self, outcome: SignalOutcome) -> float:
        """
        Calculate an overall quality score for a signal based on its outcomes.

        Score components:
        - Directional accuracy (did price move in predicted direction?)
        - Return magnitude (how much did it move?)
        - MFE/MAE ratio (was the path favorable?)
        - Time efficiency (did it reach targets quickly?)
        """
        score = 0.0
        weights_sum = 0.0

        # Directional accuracy (40% weight)
        if outcome.return_5d is not None:
            accuracy_5d = 1.0 if outcome.return_5d > 0 else 0.0
            score += 0.2 * accuracy_5d
            weights_sum += 0.2

        if outcome.return_10d is not None:
            accuracy_10d = 1.0 if outcome.return_10d > 0 else 0.0
            score += 0.2 * accuracy_10d
            weights_sum += 0.2

        # Return magnitude (30% weight)
        if outcome.return_10d is not None:
            # Scale return: 0% = 0.0, 5%+ = 1.0
            return_score = min(1.0, max(0.0, outcome.return_10d / 0.05))
            score += 0.3 * return_score
            weights_sum += 0.3

        # MFE/MAE ratio (20% weight)
        if outcome.mfe_10d is not None and outcome.mae_10d is not None:
            if abs(outcome.mae_10d) > 0.001:
                ratio = outcome.mfe_10d / abs(outcome.mae_10d)
                # Scale: ratio of 1 = 0.5, ratio of 3+ = 1.0
                ratio_score = min(1.0, max(0.0, (ratio - 0.5) / 2.5))
                score += 0.2 * ratio_score
                weights_sum += 0.2

        # Time efficiency (10% weight)
        if outcome.time_to_2pct is not None:
            # Faster is better: 1 day = 1.0, 10 days = 0.0
            time_score = max(0.0, 1.0 - (outcome.time_to_2pct - 1) / 9)
            score += 0.1 * time_score
            weights_sum += 0.1

        if weights_sum > 0:
            return score / weights_sum
        return 0.5  # Default neutral score

    def get_quality_metrics(
        self,
        min_signals: int = 10,
        symbol: Optional[str] = None,
        regime: Optional[str] = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None
    ) -> SignalQualityMetrics:
        """
        Calculate aggregate quality metrics for completed signals.

        Args:
            min_signals: Minimum signals required for meaningful metrics
            symbol: Filter by symbol (optional)
            regime: Filter by regime (optional)
            start_date: Filter signals after this date
            end_date: Filter signals before this date

        Returns:
            SignalQualityMetrics with aggregate statistics
        """
        # Filter signals
        signals = self.completed_signals.copy()

        if symbol:
            signals = [s for s in signals if s.symbol == symbol]
        if regime:
            signals = [s for s in signals if s.regime == regime]
        if start_date:
            signals = [s for s in signals if s.signal_time >= start_date]
        if end_date:
            signals = [s for s in signals if s.signal_time <= end_date]

        metrics = SignalQualityMetrics(total_signals=len(signals))

        if len(signals) < min_signals:
            return metrics

        # Calculate directional accuracy
        for window_attr, metric_attr in [
            ('return_1d', 'accuracy_1d'),
            ('return_5d', 'accuracy_5d'),
            ('return_10d', 'accuracy_10d'),
            ('return_20d', 'accuracy_20d')
        ]:
            valid_signals = [s for s in signals if getattr(s, window_attr) is not None]
            if valid_signals:
                correct = sum(1 for s in valid_signals if getattr(s, window_attr) > 0)
                setattr(metrics, metric_attr, correct / len(valid_signals))

        # Average returns by direction
        buy_signals = [s for s in signals if s.signal_direction == "BUY"]
        sell_signals = [s for s in signals if s.signal_direction == "SELL"]

        if buy_signals:
            returns_5d = [s.return_5d for s in buy_signals if s.return_5d is not None]
            returns_10d = [s.return_10d for s in buy_signals if s.return_10d is not None]
            if returns_5d:
                metrics.avg_return_5d_buy = np.mean(returns_5d)
            if returns_10d:
                metrics.avg_return_10d_buy = np.mean(returns_10d)

        if sell_signals:
            returns_5d = [s.return_5d for s in sell_signals if s.return_5d is not None]
            returns_10d = [s.return_10d for s in sell_signals if s.return_10d is not None]
            if returns_5d:
                metrics.avg_return_5d_sell = np.mean(returns_5d)
            if returns_10d:
                metrics.avg_return_10d_sell = np.mean(returns_10d)

        # MFE/MAE statistics
        mfe_values = [s.mfe_10d for s in signals if s.mfe_10d is not None]
        mae_values = [s.mae_10d for s in signals if s.mae_10d is not None]

        if mfe_values:
            metrics.avg_mfe_10d = np.mean(mfe_values)
        if mae_values:
            metrics.avg_mae_10d = np.mean(mae_values)
        if metrics.avg_mae_10d != 0:
            metrics.mfe_mae_ratio = abs(metrics.avg_mfe_10d / metrics.avg_mae_10d)

        # Target hit rates
        hit_5_5 = [s for s in signals if s.hit_5pct_5d is not None]
        hit_10_10 = [s for s in signals if s.hit_10pct_10d is not None]

        if hit_5_5:
            metrics.hit_rate_5pct_5d = sum(1 for s in hit_5_5 if s.hit_5pct_5d) / len(hit_5_5)
        if hit_10_10:
            metrics.hit_rate_10pct_10d = sum(1 for s in hit_10_10 if s.hit_10pct_10d) / len(hit_10_10)

        # Time-to-target statistics
        time_2pct = [s.time_to_2pct for s in signals if s.time_to_2pct is not None]
        time_5pct = [s.time_to_5pct for s in signals if s.time_to_5pct is not None]

        if time_2pct:
            metrics.avg_time_to_2pct = np.mean(time_2pct)
        if time_5pct:
            metrics.avg_time_to_5pct = np.mean(time_5pct)

        # Quality distribution
        quality_scores = [s.quality_score for s in signals if s.quality_score is not None]
        if quality_scores:
            metrics.high_quality_pct = sum(1 for q in quality_scores if q > 0.7) / len(quality_scores)
            metrics.low_quality_pct = sum(1 for q in quality_scores if q < 0.3) / len(quality_scores)

        # Accuracy by regime
        for regime_name in ['bull', 'bear', 'neutral', 'volatile']:
            regime_signals = [s for s in signals if s.regime == regime_name and s.return_10d is not None]
            if regime_signals:
                correct = sum(1 for s in regime_signals if s.return_10d > 0)
                metrics.accuracy_by_regime[regime_name] = correct / len(regime_signals)

        # Accuracy by confidence bucket
        for bucket_name, (low, high) in [
            ('low', (0.0, 0.4)),
            ('medium', (0.4, 0.7)),
            ('high', (0.7, 1.0))
        ]:
            bucket_signals = [s for s in signals
                           if low <= s.confidence < high and s.return_10d is not None]
            if bucket_signals:
                correct = sum(1 for s in bucket_signals if s.return_10d > 0)
                metrics.accuracy_by_confidence[bucket_name] = correct / len(bucket_signals)

        return metrics

    def get_signals_for_ml(
        self,
        min_quality: Optional[float] = None,
        direction: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Export completed signals as a DataFrame for ML feature engineering.

        Args:
            min_quality: Minimum quality score filter
            direction: Filter by direction (BUY, SELL)

        Returns:
            DataFrame with signal features and outcome labels
        """
        signals = self.completed_signals.copy()

        if min_quality is not None:
            signals = [s for s in signals if s.quality_score and s.quality_score >= min_quality]
        if direction:
            signals = [s for s in signals if s.signal_direction == direction]

        if not signals:
            return pd.DataFrame()

        data = []
        for s in signals:
            row = {
                'signal_id': s.signal_id,
                'symbol': s.symbol,
                'signal_time': s.signal_time,
                'signal_value': s.signal_value,
                'signal_direction': s.signal_direction,
                'confidence': s.confidence,
                'regime': s.regime,
                'entry_price': s.entry_price,

                # Outcome labels (for supervised learning)
                'return_1d': s.return_1d,
                'return_5d': s.return_5d,
                'return_10d': s.return_10d,
                'return_20d': s.return_20d,

                # Binary labels
                'correct_5d': 1 if s.return_5d and s.return_5d > 0 else 0,
                'correct_10d': 1 if s.return_10d and s.return_10d > 0 else 0,
                'hit_5pct_5d': 1 if s.hit_5pct_5d else 0,
                'hit_10pct_10d': 1 if s.hit_10pct_10d else 0,

                # MFE/MAE features
                'mfe_5d': s.mfe_5d,
                'mae_5d': s.mae_5d,
                'mfe_10d': s.mfe_10d,
                'mae_10d': s.mae_10d,
                'mfe_mae_ratio_10d': s.mfe_10d / abs(s.mae_10d) if s.mae_10d and abs(s.mae_10d) > 0.001 else None,

                # Time features
                'time_to_2pct': s.time_to_2pct,
                'time_to_5pct': s.time_to_5pct,

                # Quality
                'quality_score': s.quality_score
            }
            data.append(row)

        return pd.DataFrame(data)

    def print_summary(self, detailed: bool = False):
        """Print a summary of signal tracking status and quality metrics."""
        logger.info("")
        logger.info("=" * 80)
        logger.info("SIGNAL OUTCOME TRACKER SUMMARY")
        logger.info("=" * 80)

        logger.info(f"Active signals: {len(self.active_signals)}")
        logger.info(f"Completed signals: {len(self.completed_signals)}")

        if self.completed_signals:
            metrics = self.get_quality_metrics()

            logger.info("")
            logger.info("QUALITY METRICS:")
            logger.info("  Directional Accuracy:")
            logger.info(f"    1-day:  {metrics.accuracy_1d*100:.1f}%")
            logger.info(f"    5-day:  {metrics.accuracy_5d*100:.1f}%")
            logger.info(f"    10-day: {metrics.accuracy_10d*100:.1f}%")
            logger.info(f"    20-day: {metrics.accuracy_20d*100:.1f}%")

            logger.info("  Average Returns:")
            logger.info(f"    BUY signals (10d):  {metrics.avg_return_10d_buy*100:+.2f}%")
            logger.info(f"    SELL signals (10d): {metrics.avg_return_10d_sell*100:+.2f}%")

            logger.info("  MFE/MAE Analysis:")
            logger.info(f"    Avg MFE (10d): {metrics.avg_mfe_10d*100:+.2f}%")
            logger.info(f"    Avg MAE (10d): {metrics.avg_mae_10d*100:+.2f}%")
            logger.info(f"    MFE/MAE Ratio: {metrics.mfe_mae_ratio:.2f}")

            logger.info("  Target Hit Rates:")
            logger.info(f"    5% in 5 days:  {metrics.hit_rate_5pct_5d*100:.1f}%")
            logger.info(f"    10% in 10 days: {metrics.hit_rate_10pct_10d*100:.1f}%")

            logger.info("  Quality Distribution:")
            logger.info(f"    High quality (>70%): {metrics.high_quality_pct*100:.1f}%")
            logger.info(f"    Low quality (<30%):  {metrics.low_quality_pct*100:.1f}%")

            if detailed and metrics.accuracy_by_regime:
                logger.info("  Accuracy by Regime:")
                for regime, acc in metrics.accuracy_by_regime.items():
                    logger.info(f"    {regime}: {acc*100:.1f}%")

            if detailed and metrics.accuracy_by_confidence:
                logger.info("  Accuracy by Confidence:")
                for bucket, acc in metrics.accuracy_by_confidence.items():
                    logger.info(f"    {bucket}: {acc*100:.1f}%")

        logger.info("=" * 80)

    def _save_state(self):
        """Persist tracker state to disk."""
        try:
            state = {
                'active_signals': {k: v.to_dict() for k, v in self.active_signals.items()},
                'completed_signals': [s.to_dict() for s in self.completed_signals[-1000:]],  # Keep last 1000
                'last_updated': datetime.now().isoformat()
            }

            state_file = self.data_dir / "signal_outcomes.json"
            with open(state_file, 'w') as f:
                json.dump(state, f, indent=2, default=str)

        except Exception as e:
            logger.error(f"Error saving signal outcome state: {e}")

    def _load_state(self):
        """Load tracker state from disk."""
        try:
            state_file = self.data_dir / "signal_outcomes.json"
            if not state_file.exists():
                return

            with open(state_file) as f:
                state = json.load(f)

            self.active_signals = {
                k: SignalOutcome.from_dict(v)
                for k, v in state.get('active_signals', {}).items()
            }

            self.completed_signals = [
                SignalOutcome.from_dict(s)
                for s in state.get('completed_signals', [])
            ]

            # Rebuild symbol index
            for signal_id, outcome in self.active_signals.items():
                self.signals_by_symbol[outcome.symbol].append(signal_id)

            logger.info(f"Loaded signal outcome state from {state_file}")

        except Exception as e:
            logger.error(f"Error loading signal outcome state: {e}")

    def export_to_csv(self, filepath: Optional[str] = None) -> str:
        """Export completed signals to CSV for analysis."""
        if filepath is None:
            filepath = str(self.data_dir / "signal_outcomes.csv")

        df = self.get_signals_for_ml()
        if not df.empty:
            df.to_csv(filepath, index=False)
            logger.info(f"Exported {len(df)} signals to {filepath}")

        return filepath
