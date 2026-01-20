"""
models/regime_detector.py - Market Regime Detection

Uses Hidden Markov Models (HMM) to classify market regimes:
- Bull (trending up, low volatility)
- Bear (trending down, high volatility)
- Sideways (range-bound, moderate volatility)
- High Volatility (crisis/fear)

Adapts trading strategy based on detected regime.

Usage:
    detector = RegimeDetector()
    detector.fit(spy_returns)
    regime = detector.detect_regime(recent_returns)
    strategy_params = detector.get_strategy_adjustment(regime)
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from enum import Enum
from datetime import datetime
import logging
import json
from pathlib import Path

logger = logging.getLogger(__name__)

# Check for HMM library
try:
    from hmmlearn.hmm import GaussianHMM
    HMM_AVAILABLE = True
except ImportError:
    HMM_AVAILABLE = False
    logger.warning("hmmlearn not installed. Install with: pip install hmmlearn")


class MarketRegime(Enum):
    """Market regime classifications."""
    BULL = "bull"              # Strong uptrend, low vol
    BEAR = "bear"              # Downtrend, high vol
    SIDEWAYS = "sideways"      # Range-bound
    HIGH_VOL = "high_vol"      # Crisis/extreme volatility
    UNKNOWN = "unknown"        # Unable to classify


@dataclass
class RegimeState:
    """Current regime state with metadata."""
    regime: MarketRegime
    confidence: float
    probability_distribution: Dict[str, float]
    volatility: float
    trend: float
    detected_at: datetime
    features_used: Dict[str, float]

    def to_dict(self) -> dict:
        return {
            'regime': self.regime.value,
            'confidence': self.confidence,
            'probabilities': self.probability_distribution,
            'volatility': self.volatility,
            'trend': self.trend,
            'detected_at': self.detected_at.isoformat(),
            'features': self.features_used
        }


@dataclass
class StrategyAdjustment:
    """Strategy parameters adjusted for current regime."""
    position_size_multiplier: float  # Scale position sizes
    signal_threshold_multiplier: float  # Adjust signal sensitivity
    stop_loss_multiplier: float  # Adjust stop loss distance
    take_profit_multiplier: float  # Adjust take profit distance
    max_positions_multiplier: float  # Adjust max concurrent positions
    prefer_long: bool  # Bias toward long positions
    prefer_short: bool  # Bias toward short positions
    use_mean_reversion: bool  # Use mean reversion signals
    use_momentum: bool  # Use momentum signals
    regime: MarketRegime

    def to_dict(self) -> dict:
        return {
            'position_size_mult': self.position_size_multiplier,
            'signal_threshold_mult': self.signal_threshold_multiplier,
            'stop_loss_mult': self.stop_loss_multiplier,
            'take_profit_mult': self.take_profit_multiplier,
            'max_positions_mult': self.max_positions_multiplier,
            'prefer_long': self.prefer_long,
            'prefer_short': self.prefer_short,
            'use_mean_reversion': self.use_mean_reversion,
            'use_momentum': self.use_momentum,
            'regime': self.regime.value
        }


class RegimeDetector:
    """
    Detects market regime using Hidden Markov Models and technical features.

    Features used:
    - Rolling returns (5, 20, 60 day)
    - Rolling volatility (20, 60 day)
    - Trend strength (slope of prices)
    - RSI
    - VIX level (if available)
    """

    def __init__(
        self,
        n_regimes: int = 4,
        lookback_days: int = 252,
        retrain_interval_days: int = 30,
        model_dir: Path = Path('models/saved')
    ):
        """
        Initialize regime detector.

        Args:
            n_regimes: Number of regimes to detect (default 4)
            lookback_days: Days of history for training
            retrain_interval_days: How often to retrain
            model_dir: Directory to save trained models
        """
        self.n_regimes = n_regimes
        self.lookback_days = lookback_days
        self.retrain_interval = retrain_interval_days
        self.model_dir = model_dir

        # Model
        self.hmm_model: Optional[GaussianHMM] = None
        self.is_fitted = False
        self.last_train_date: Optional[datetime] = None

        # Regime mapping (updated during training)
        self.regime_mapping: Dict[int, MarketRegime] = {}

        # State
        self.current_regime: Optional[RegimeState] = None
        self.regime_history: List[RegimeState] = []

        # Feature scaler params
        self._feature_means: Optional[np.ndarray] = None
        self._feature_stds: Optional[np.ndarray] = None

        # Ensure model directory exists
        self.model_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"📈 Regime Detector initialized")
        logger.info(f"   Regimes: {n_regimes}")
        logger.info(f"   Lookback: {lookback_days} days")
        logger.info(f"   HMM Available: {HMM_AVAILABLE}")

    def extract_features(self, prices: pd.Series) -> np.ndarray:
        """
        Extract regime detection features from price series.

        Args:
            prices: Price series (usually SPY or market index)

        Returns:
            Feature array of shape (n_samples, n_features)
        """
        if len(prices) < 60:
            return np.array([])

        features = []
        returns = prices.pct_change().dropna()

        # 1. Rolling returns
        ret_5 = returns.rolling(5).mean()
        ret_20 = returns.rolling(20).mean()
        ret_60 = returns.rolling(60).mean()

        # 2. Rolling volatility
        vol_20 = returns.rolling(20).std() * np.sqrt(252)
        vol_60 = returns.rolling(60).std() * np.sqrt(252)

        # 3. Volatility ratio (vol regime indicator)
        vol_ratio = vol_20 / vol_60

        # 4. Trend strength (linear regression slope)
        trend_20 = prices.rolling(20).apply(
            lambda x: np.polyfit(range(len(x)), x, 1)[0] / x.iloc[0] if len(x) > 1 else 0,
            raw=False
        )

        # 5. RSI
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / loss.replace(0, np.inf)
        rsi = 100 - (100 / (1 + rs))
        rsi_normalized = (rsi - 50) / 50  # Normalize to -1 to 1

        # 6. Price vs MA (trend indicator)
        ma_50 = prices.rolling(50).mean()
        price_vs_ma = (prices - ma_50) / ma_50

        # Combine features
        feature_df = pd.DataFrame({
            'ret_5': ret_5,
            'ret_20': ret_20,
            'ret_60': ret_60,
            'vol_20': vol_20,
            'vol_60': vol_60,
            'vol_ratio': vol_ratio,
            'trend_20': trend_20,
            'rsi': rsi_normalized,
            'price_vs_ma': price_vs_ma
        }).dropna()

        return feature_df.values

    def _scale_features(self, features: np.ndarray, fit: bool = False) -> np.ndarray:
        """Scale features to zero mean, unit variance."""
        if fit or self._feature_means is None:
            self._feature_means = features.mean(axis=0)
            self._feature_stds = features.std(axis=0)
            self._feature_stds[self._feature_stds == 0] = 1  # Avoid division by zero

        return (features - self._feature_means) / self._feature_stds

    def fit(self, prices: pd.Series, vix_prices: Optional[pd.Series] = None) -> bool:
        """
        Train the HMM model on historical data.

        Args:
            prices: Historical price series (SPY recommended)
            vix_prices: Optional VIX prices for enhanced detection

        Returns:
            True if training successful
        """
        if not HMM_AVAILABLE:
            logger.error("Cannot fit: hmmlearn not installed")
            return False

        logger.info("🎓 Training regime detection model...")

        try:
            # Extract features
            features = self.extract_features(prices)
            if len(features) < 100:
                logger.error(f"Not enough data for training ({len(features)} samples)")
                return False

            # Scale features
            features_scaled = self._scale_features(features, fit=True)

            # Train HMM
            self.hmm_model = GaussianHMM(
                n_components=self.n_regimes,
                covariance_type='full',
                n_iter=200,
                random_state=42
            )
            self.hmm_model.fit(features_scaled)

            # Identify regime characteristics
            self._identify_regimes(features, prices)

            self.is_fitted = True
            self.last_train_date = datetime.now()

            # Save model
            self._save_model()

            logger.info(f"✅ Regime model trained on {len(features)} samples")
            logger.info(f"   Regime mapping: {self.regime_mapping}")

            return True

        except Exception as e:
            logger.error(f"❌ Training failed: {e}")
            return False

    def _identify_regimes(self, features: np.ndarray, prices: pd.Series):
        """
        Identify which HMM state corresponds to which market regime.

        Uses mean return and volatility of each state to classify.
        """
        features_scaled = self._scale_features(features)
        states = self.hmm_model.predict(features_scaled)

        # Calculate characteristics of each state
        state_chars = {}
        returns = prices.pct_change().dropna().values[-len(states):]

        for state in range(self.n_regimes):
            mask = states == state
            if mask.sum() > 0:
                state_returns = returns[mask]
                state_vol = features[mask, 3]  # vol_20 column

                state_chars[state] = {
                    'mean_return': np.mean(state_returns),
                    'mean_vol': np.mean(state_vol),
                    'count': mask.sum()
                }

        # Map states to regimes based on characteristics
        # Sort by volatility to identify high/low vol regimes
        sorted_states = sorted(
            state_chars.items(),
            key=lambda x: x[1]['mean_vol']
        )

        # Assign regimes
        self.regime_mapping = {}
        for i, (state, chars) in enumerate(sorted_states):
            if chars['mean_vol'] > 0.25:  # High volatility threshold
                if chars['mean_return'] < 0:
                    self.regime_mapping[state] = MarketRegime.BEAR
                else:
                    self.regime_mapping[state] = MarketRegime.HIGH_VOL
            elif chars['mean_return'] > 0.0003:  # ~7.5% annualized
                self.regime_mapping[state] = MarketRegime.BULL
            elif chars['mean_return'] < -0.0001:
                self.regime_mapping[state] = MarketRegime.BEAR
            else:
                self.regime_mapping[state] = MarketRegime.SIDEWAYS

        logger.info(f"   State characteristics: {state_chars}")

    def detect_regime(self, prices: pd.Series) -> RegimeState:
        """
        Detect current market regime.

        Args:
            prices: Recent price series (at least 60 days)

        Returns:
            RegimeState with current regime and confidence
        """
        if not self.is_fitted:
            logger.warning("Model not fitted, returning UNKNOWN regime")
            return RegimeState(
                regime=MarketRegime.UNKNOWN,
                confidence=0.0,
                probability_distribution={},
                volatility=0.0,
                trend=0.0,
                detected_at=datetime.now(),
                features_used={}
            )

        try:
            # Extract features
            features = self.extract_features(prices)
            if len(features) == 0:
                return self._unknown_state()

            # Use last observation
            latest_features = features[-1:].reshape(1, -1)
            latest_scaled = self._scale_features(latest_features)

            # Predict state and probabilities
            state = self.hmm_model.predict(latest_scaled)[0]
            probs = self.hmm_model.predict_proba(latest_scaled)[0]

            # Get regime
            regime = self.regime_mapping.get(state, MarketRegime.UNKNOWN)
            confidence = float(probs[state])

            # Build probability distribution
            prob_dist = {
                self.regime_mapping.get(i, MarketRegime.UNKNOWN).value: float(p)
                for i, p in enumerate(probs)
            }

            # Calculate current metrics
            returns = prices.pct_change().dropna()
            current_vol = float(returns.iloc[-20:].std() * np.sqrt(252))
            current_trend = float((prices.iloc[-1] / prices.iloc[-20] - 1))

            # Build state
            regime_state = RegimeState(
                regime=regime,
                confidence=confidence,
                probability_distribution=prob_dist,
                volatility=current_vol,
                trend=current_trend,
                detected_at=datetime.now(),
                features_used={
                    'ret_5': float(returns.iloc[-5:].mean()),
                    'ret_20': float(returns.iloc[-20:].mean()),
                    'vol_20': current_vol
                }
            )

            # Update state
            self.current_regime = regime_state
            self.regime_history.append(regime_state)

            # Keep only last 1000 states
            if len(self.regime_history) > 1000:
                self.regime_history = self.regime_history[-1000:]

            logger.info(f"📈 Regime: {regime.value} (confidence: {confidence:.2%})")
            logger.debug(f"   Vol: {current_vol:.1%}, Trend: {current_trend:+.1%}")

            return regime_state

        except Exception as e:
            logger.error(f"Regime detection error: {e}")
            return self._unknown_state()

    def _unknown_state(self) -> RegimeState:
        """Return unknown regime state."""
        return RegimeState(
            regime=MarketRegime.UNKNOWN,
            confidence=0.0,
            probability_distribution={},
            volatility=0.0,
            trend=0.0,
            detected_at=datetime.now(),
            features_used={}
        )

    def get_strategy_adjustment(self, regime: Optional[MarketRegime] = None) -> StrategyAdjustment:
        """
        Get strategy parameters adjusted for current regime.

        Args:
            regime: Override regime (uses current if None)

        Returns:
            StrategyAdjustment with modified parameters
        """
        if regime is None:
            regime = self.current_regime.regime if self.current_regime else MarketRegime.UNKNOWN

        # Define adjustments for each regime
        adjustments = {
            MarketRegime.BULL: StrategyAdjustment(
                position_size_multiplier=1.2,
                signal_threshold_multiplier=0.9,
                stop_loss_multiplier=1.2,
                take_profit_multiplier=1.5,
                max_positions_multiplier=1.2,
                prefer_long=True,
                prefer_short=False,
                use_mean_reversion=False,
                use_momentum=True,
                regime=regime
            ),
            MarketRegime.BEAR: StrategyAdjustment(
                position_size_multiplier=0.5,
                signal_threshold_multiplier=1.3,
                stop_loss_multiplier=0.8,
                take_profit_multiplier=0.8,
                max_positions_multiplier=0.6,
                prefer_long=False,
                prefer_short=True,
                use_mean_reversion=True,
                use_momentum=False,
                regime=regime
            ),
            MarketRegime.SIDEWAYS: StrategyAdjustment(
                position_size_multiplier=0.8,
                signal_threshold_multiplier=1.1,
                stop_loss_multiplier=0.9,
                take_profit_multiplier=0.9,
                max_positions_multiplier=1.0,
                prefer_long=False,
                prefer_short=False,
                use_mean_reversion=True,
                use_momentum=False,
                regime=regime
            ),
            MarketRegime.HIGH_VOL: StrategyAdjustment(
                position_size_multiplier=0.3,
                signal_threshold_multiplier=1.5,
                stop_loss_multiplier=0.6,
                take_profit_multiplier=0.6,
                max_positions_multiplier=0.4,
                prefer_long=False,
                prefer_short=False,
                use_mean_reversion=False,
                use_momentum=False,
                regime=regime
            ),
            MarketRegime.UNKNOWN: StrategyAdjustment(
                position_size_multiplier=0.5,
                signal_threshold_multiplier=1.2,
                stop_loss_multiplier=0.8,
                take_profit_multiplier=1.0,
                max_positions_multiplier=0.7,
                prefer_long=False,
                prefer_short=False,
                use_mean_reversion=False,
                use_momentum=False,
                regime=regime
            )
        }

        return adjustments.get(regime, adjustments[MarketRegime.UNKNOWN])

    def _save_model(self):
        """Save trained model to disk."""
        if not self.is_fitted:
            return

        try:
            import pickle
            model_path = self.model_dir / 'regime_hmm.pkl'
            with open(model_path, 'wb') as f:
                pickle.dump({
                    'model': self.hmm_model,
                    'regime_mapping': self.regime_mapping,
                    'feature_means': self._feature_means,
                    'feature_stds': self._feature_stds,
                    'train_date': self.last_train_date
                }, f)
            logger.info(f"✅ Regime model saved to {model_path}")
        except Exception as e:
            logger.error(f"Failed to save model: {e}")

    def load_model(self) -> bool:
        """Load trained model from disk."""
        try:
            import pickle
            model_path = self.model_dir / 'regime_hmm.pkl'
            if not model_path.exists():
                return False

            with open(model_path, 'rb') as f:
                data = pickle.load(f)

            self.hmm_model = data['model']
            self.regime_mapping = data['regime_mapping']
            self._feature_means = data['feature_means']
            self._feature_stds = data['feature_stds']
            self.last_train_date = data['train_date']
            self.is_fitted = True

            logger.info(f"✅ Regime model loaded (trained: {self.last_train_date})")
            return True

        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            return False

    def needs_retrain(self) -> bool:
        """Check if model needs retraining."""
        if not self.is_fitted or self.last_train_date is None:
            return True

        days_since_train = (datetime.now() - self.last_train_date).days
        return days_since_train >= self.retrain_interval

    def get_status(self) -> Dict:
        """Get detector status for dashboard."""
        return {
            'is_fitted': self.is_fitted,
            'last_train_date': self.last_train_date.isoformat() if self.last_train_date else None,
            'needs_retrain': self.needs_retrain(),
            'current_regime': self.current_regime.regime.value if self.current_regime else None,
            'current_confidence': self.current_regime.confidence if self.current_regime else 0,
            'hmm_available': HMM_AVAILABLE,
            'regime_history_count': len(self.regime_history)
        }
