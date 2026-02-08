"""
models/base_signal_generator.py - Base Signal Generator Class

Provides a common foundation for all signal generators:
- Shared feature extraction methods
- Common signal processing utilities
- Regime detection interface
- Signal validation and normalization

All signal generator variants should inherit from this class
to reduce code duplication and ensure consistency.
"""

import numpy as np
import pandas as pd
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class MarketRegime(Enum):
    """Market regime classification."""
    STRONG_BULL = "strong_bull"
    BULL = "bull"
    NEUTRAL = "neutral"
    BEAR = "bear"
    STRONG_BEAR = "strong_bear"
    HIGH_VOLATILITY = "high_volatility"


@dataclass
class Signal:
    """Standardized signal output."""
    signal: float                    # Signal value [-1, 1]
    confidence: float               # Confidence [0, 1]
    direction: str                  # "LONG", "SHORT", "NEUTRAL"
    regime: MarketRegime           # Current market regime
    timestamp: datetime            # Signal generation time

    # Component signals (for transparency)
    momentum_signal: float = 0.0
    mean_reversion_signal: float = 0.0
    trend_signal: float = 0.0
    volatility_signal: float = 0.0
    ml_prediction: float = 0.0

    # Metadata
    model_agreement: float = 0.0   # Agreement between models
    feature_importance: Dict[str, float] = None


class BaseSignalGenerator(ABC):
    """
    Abstract base class for all signal generators.

    Provides common functionality:
    - Technical indicator calculations
    - Feature extraction
    - Signal normalization
    - Regime detection

    Subclasses must implement:
    - generate_signal(symbol, prices) -> Signal
    """

    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize base signal generator.

        Args:
            config: Optional configuration dictionary
        """
        self.config = config or {}

        # Default parameters
        self.lookback_short = self.config.get('lookback_short', 10)
        self.lookback_medium = self.config.get('lookback_medium', 20)
        self.lookback_long = self.config.get('lookback_long', 50)

        # Signal bounds
        self.signal_min = -1.0
        self.signal_max = 1.0

        # Feature cache
        self._feature_cache: Dict[str, pd.DataFrame] = {}

        logger.info(f"{self.__class__.__name__} initialized")

    @abstractmethod
    def generate_signal(
        self,
        symbol: str,
        prices: Union[pd.Series, pd.DataFrame]
    ) -> Signal:
        """
        Generate trading signal for a symbol.

        Args:
            symbol: Stock symbol
            prices: Price data (Series for close, DataFrame for OHLCV)

        Returns:
            Signal object with trading recommendation
        """
        pass

    # =========================================================================
    # Technical Indicators
    # =========================================================================

    def calculate_sma(self, prices: pd.Series, window: int) -> pd.Series:
        """Calculate Simple Moving Average."""
        return prices.rolling(window=window).mean()

    def calculate_ema(self, prices: pd.Series, span: int) -> pd.Series:
        """Calculate Exponential Moving Average."""
        return prices.ewm(span=span, adjust=False).mean()

    def calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """Calculate Relative Strength Index."""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()

        rs = gain / loss.replace(0, np.inf)
        rsi = 100 - (100 / (1 + rs))
        return rsi.fillna(50)

    def calculate_macd(
        self,
        prices: pd.Series,
        fast: int = 12,
        slow: int = 26,
        signal: int = 9
    ) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """
        Calculate MACD indicator.

        Returns:
            Tuple of (macd_line, signal_line, histogram)
        """
        ema_fast = self.calculate_ema(prices, fast)
        ema_slow = self.calculate_ema(prices, slow)

        macd_line = ema_fast - ema_slow
        signal_line = self.calculate_ema(macd_line, signal)
        histogram = macd_line - signal_line

        return macd_line, signal_line, histogram

    def calculate_bollinger_bands(
        self,
        prices: pd.Series,
        window: int = 20,
        num_std: float = 2.0
    ) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """
        Calculate Bollinger Bands.

        Returns:
            Tuple of (upper_band, middle_band, lower_band)
        """
        middle = self.calculate_sma(prices, window)
        std = prices.rolling(window=window).std()

        upper = middle + (std * num_std)
        lower = middle - (std * num_std)

        return upper, middle, lower

    def calculate_atr(
        self,
        high: pd.Series,
        low: pd.Series,
        close: pd.Series,
        period: int = 14
    ) -> pd.Series:
        """Calculate Average True Range."""
        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))

        true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = true_range.rolling(window=period).mean()

        return atr

    def calculate_stochastic(
        self,
        high: pd.Series,
        low: pd.Series,
        close: pd.Series,
        k_period: int = 14,
        d_period: int = 3
    ) -> Tuple[pd.Series, pd.Series]:
        """
        Calculate Stochastic Oscillator.

        Returns:
            Tuple of (%K, %D)
        """
        lowest_low = low.rolling(window=k_period).min()
        highest_high = high.rolling(window=k_period).max()

        k = 100 * (close - lowest_low) / (highest_high - lowest_low + 1e-10)
        d = k.rolling(window=d_period).mean()

        return k.fillna(50), d.fillna(50)

    def calculate_adx(
        self,
        high: pd.Series,
        low: pd.Series,
        close: pd.Series,
        period: int = 14
    ) -> pd.Series:
        """Calculate Average Directional Index."""
        plus_dm = high.diff()
        minus_dm = -low.diff()

        plus_dm[plus_dm < 0] = 0
        minus_dm[minus_dm < 0] = 0

        tr = self.calculate_atr(high, low, close, 1)
        tr14 = tr.rolling(window=period).sum()

        plus_di = 100 * (plus_dm.rolling(window=period).sum() / tr14)
        minus_di = 100 * (minus_dm.rolling(window=period).sum() / tr14)

        dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
        adx = dx.rolling(window=period).mean()

        return adx.fillna(20)

    def calculate_obv(self, close: pd.Series, volume: pd.Series) -> pd.Series:
        """Calculate On-Balance Volume."""
        direction = np.sign(close.diff())
        obv = (volume * direction).cumsum()
        return obv

    def calculate_vwap(
        self,
        high: pd.Series,
        low: pd.Series,
        close: pd.Series,
        volume: pd.Series
    ) -> pd.Series:
        """Calculate Volume Weighted Average Price."""
        typical_price = (high + low + close) / 3
        vwap = (typical_price * volume).cumsum() / volume.cumsum()
        return vwap

    # =========================================================================
    # Feature Extraction
    # =========================================================================

    def extract_features(
        self,
        prices: pd.DataFrame,
        include_volume: bool = True
    ) -> pd.DataFrame:
        """
        Extract comprehensive feature set from price data.

        Args:
            prices: DataFrame with OHLCV columns
            include_volume: Whether to include volume-based features

        Returns:
            DataFrame with extracted features
        """
        features = pd.DataFrame(index=prices.index)

        # Standardize column names
        close = prices.get('close', prices.get('Close', pd.Series()))
        high = prices.get('high', prices.get('High', close))
        low = prices.get('low', prices.get('Low', close))
        volume = prices.get('volume', prices.get('Volume', pd.Series()))

        if close.empty:
            logger.warning("No close prices found")
            return features

        # Price-based features
        features['return_1d'] = close.pct_change(1)
        features['return_5d'] = close.pct_change(5)
        features['return_10d'] = close.pct_change(10)
        features['return_20d'] = close.pct_change(20)

        # Moving averages
        features['sma_10'] = self.calculate_sma(close, 10)
        features['sma_20'] = self.calculate_sma(close, 20)
        features['sma_50'] = self.calculate_sma(close, 50)

        features['ema_10'] = self.calculate_ema(close, 10)
        features['ema_20'] = self.calculate_ema(close, 20)

        # MA crossovers
        features['sma_10_20_cross'] = (features['sma_10'] - features['sma_20']) / close
        features['sma_20_50_cross'] = (features['sma_20'] - features['sma_50']) / close
        features['price_vs_sma20'] = (close - features['sma_20']) / features['sma_20']
        features['price_vs_sma50'] = (close - features['sma_50']) / features['sma_50']

        # Momentum indicators
        features['rsi'] = self.calculate_rsi(close, 14)
        features['rsi_normalized'] = (features['rsi'] - 50) / 50  # [-1, 1]

        macd, signal, hist = self.calculate_macd(close)
        features['macd'] = macd
        features['macd_signal'] = signal
        features['macd_hist'] = hist
        features['macd_normalized'] = hist / close.rolling(20).std()

        k, d = self.calculate_stochastic(high, low, close)
        features['stoch_k'] = k
        features['stoch_d'] = d
        features['stoch_signal'] = (k - 50) / 50

        # Volatility indicators
        upper, middle, lower = self.calculate_bollinger_bands(close)
        features['bb_upper'] = upper
        features['bb_lower'] = lower
        features['bb_position'] = (close - lower) / (upper - lower + 1e-10)
        features['bb_width'] = (upper - lower) / middle

        features['atr'] = self.calculate_atr(high, low, close)
        features['atr_pct'] = features['atr'] / close

        features['volatility_10d'] = close.pct_change().rolling(10).std()
        features['volatility_20d'] = close.pct_change().rolling(20).std()
        features['vol_ratio'] = features['volatility_10d'] / features['volatility_20d']

        # Trend indicators
        features['adx'] = self.calculate_adx(high, low, close)

        # Price patterns
        features['higher_high'] = (high > high.shift(1)).astype(int)
        features['lower_low'] = (low < low.shift(1)).astype(int)
        features['higher_close'] = (close > close.shift(1)).astype(int)

        # Range features
        features['daily_range'] = (high - low) / close
        features['true_range'] = features['atr'] / close

        # Volume features
        if include_volume and not volume.empty:
            features['volume_sma'] = self.calculate_sma(volume, 20)
            features['volume_ratio'] = volume / features['volume_sma']
            features['obv'] = self.calculate_obv(close, volume)
            features['obv_normalized'] = features['obv'].pct_change(5)

            vwap = self.calculate_vwap(high, low, close, volume)
            features['price_vs_vwap'] = (close - vwap) / vwap

        # Clean up
        features = features.replace([np.inf, -np.inf], np.nan)
        features = features.fillna(0)

        return features

    def extract_microstructure_features(
        self,
        prices: pd.DataFrame,
        window: int = 20
    ) -> pd.DataFrame:
        """Extract market microstructure features."""
        features = pd.DataFrame(index=prices.index)

        close = prices.get('close', prices.get('Close', pd.Series()))
        high = prices.get('high', prices.get('High', close))
        low = prices.get('low', prices.get('Low', close))
        volume = prices.get('volume', prices.get('Volume', pd.Series()))

        if close.empty:
            return features

        # Price efficiency ratio
        net_move = abs(close - close.shift(window))
        total_path = abs(close.diff()).rolling(window).sum()
        features['efficiency_ratio'] = net_move / (total_path + 1e-10)

        # Amihud illiquidity
        if not volume.empty:
            daily_return = abs(close.pct_change())
            dollar_volume = close * volume
            features['amihud_illiquidity'] = (daily_return / dollar_volume).rolling(window).mean()

        # Kyle's lambda (simplified)
        if not volume.empty:
            price_change = close.diff()
            volume_signed = volume * np.sign(price_change)
            features['kyle_lambda'] = abs(price_change / volume_signed).rolling(window).mean()

        # Realized volatility vs implied (using range as proxy)
        features['realized_vol'] = close.pct_change().rolling(window).std() * np.sqrt(252)
        features['range_vol'] = ((high - low) / close).rolling(window).mean() * np.sqrt(252)
        features['vol_premium'] = features['range_vol'] - features['realized_vol']

        features = features.replace([np.inf, -np.inf], np.nan)
        features = features.fillna(0)

        return features

    # =========================================================================
    # Signal Processing
    # =========================================================================

    def normalize_signal(self, signal: float) -> float:
        """Normalize signal to [-1, 1] range."""
        return np.clip(signal, self.signal_min, self.signal_max)

    def calculate_confidence(
        self,
        signal: float,
        model_agreement: float = 1.0,
        regime_factor: float = 1.0
    ) -> float:
        """
        Calculate signal confidence.

        Args:
            signal: Raw signal value
            model_agreement: Agreement between multiple models [0, 1]
            regime_factor: Regime-based adjustment [0, 1]

        Returns:
            Confidence score [0, 1]
        """
        base_confidence = abs(signal)
        confidence = base_confidence * model_agreement * regime_factor
        return np.clip(confidence, 0.0, 1.0)

    def classify_direction(self, signal: float, threshold: float = 0.1) -> str:
        """Classify signal direction."""
        if signal > threshold:
            return "LONG"
        elif signal < -threshold:
            return "SHORT"
        return "NEUTRAL"

    def apply_regime_adjustment(
        self,
        signal: float,
        regime: MarketRegime,
        adjustments: Optional[Dict[MarketRegime, float]] = None
    ) -> float:
        """
        Adjust signal based on market regime.

        Args:
            signal: Raw signal
            regime: Current market regime
            adjustments: Dict of regime -> multiplier

        Returns:
            Adjusted signal
        """
        default_adjustments = {
            MarketRegime.STRONG_BULL: 1.2,
            MarketRegime.BULL: 1.1,
            MarketRegime.NEUTRAL: 1.0,
            MarketRegime.BEAR: 0.9,
            MarketRegime.STRONG_BEAR: 0.8,
            MarketRegime.HIGH_VOLATILITY: 0.7
        }

        adjustments = adjustments or default_adjustments
        multiplier = adjustments.get(regime, 1.0)

        return self.normalize_signal(signal * multiplier)

    # =========================================================================
    # Regime Detection
    # =========================================================================

    def detect_regime(self, prices: pd.Series, window: int = 50) -> MarketRegime:
        """
        Detect current market regime.

        Args:
            prices: Price series
            window: Lookback window

        Returns:
            MarketRegime classification
        """
        if len(prices) < window:
            return MarketRegime.NEUTRAL

        returns = prices.pct_change()
        recent_returns = returns.tail(window)

        # Calculate metrics
        cumulative_return = (prices.iloc[-1] / prices.iloc[-window] - 1)
        volatility = recent_returns.std() * np.sqrt(252)
        avg_return = recent_returns.mean() * 252

        # High volatility regime
        if volatility > 0.35:
            return MarketRegime.HIGH_VOLATILITY

        # Trend-based regimes
        if cumulative_return > 0.15:
            return MarketRegime.STRONG_BULL
        elif cumulative_return > 0.05:
            return MarketRegime.BULL
        elif cumulative_return < -0.15:
            return MarketRegime.STRONG_BEAR
        elif cumulative_return < -0.05:
            return MarketRegime.BEAR

        return MarketRegime.NEUTRAL

    # =========================================================================
    # Momentum Signals
    # =========================================================================

    def calculate_momentum_signal(self, features: pd.DataFrame) -> float:
        """Calculate momentum-based signal component."""
        signals = []

        # RSI signal
        if 'rsi' in features.columns:
            rsi = features['rsi'].iloc[-1]
            if rsi > 70:
                signals.append(-0.5 * (rsi - 70) / 30)  # Overbought
            elif rsi < 30:
                signals.append(0.5 * (30 - rsi) / 30)   # Oversold
            else:
                signals.append((rsi - 50) / 100)

        # MACD signal
        if 'macd_normalized' in features.columns:
            signals.append(np.tanh(features['macd_normalized'].iloc[-1]))

        # Stochastic signal
        if 'stoch_signal' in features.columns:
            signals.append(features['stoch_signal'].iloc[-1] * 0.5)

        if signals:
            return np.mean(signals)
        return 0.0

    def calculate_mean_reversion_signal(self, features: pd.DataFrame) -> float:
        """Calculate mean reversion signal component."""
        signals = []

        # Bollinger Band position (oversold/overbought)
        if 'bb_position' in features.columns:
            bb_pos = features['bb_position'].iloc[-1]
            if bb_pos < 0.2:
                signals.append(0.5)   # Near lower band
            elif bb_pos > 0.8:
                signals.append(-0.5)  # Near upper band
            else:
                signals.append((0.5 - bb_pos) * 0.5)

        # Price vs SMA deviation
        if 'price_vs_sma20' in features.columns:
            deviation = features['price_vs_sma20'].iloc[-1]
            signals.append(-np.tanh(deviation * 5))  # Mean revert

        if signals:
            return np.mean(signals)
        return 0.0

    def calculate_trend_signal(self, features: pd.DataFrame) -> float:
        """Calculate trend-following signal component."""
        signals = []

        # MA crossover signals
        if 'sma_10_20_cross' in features.columns:
            cross = features['sma_10_20_cross'].iloc[-1]
            signals.append(np.tanh(cross * 20))

        # ADX trend strength
        if 'adx' in features.columns:
            adx = features['adx'].iloc[-1]
            if adx > 25:  # Strong trend
                # Use price vs MA to determine direction
                if 'price_vs_sma20' in features.columns:
                    direction = 1 if features['price_vs_sma20'].iloc[-1] > 0 else -1
                    signals.append(direction * (adx / 100))

        # Multi-period returns (trend continuation)
        for col in ['return_5d', 'return_10d', 'return_20d']:
            if col in features.columns:
                signals.append(np.tanh(features[col].iloc[-1] * 10))

        if signals:
            return np.mean(signals)
        return 0.0

    def calculate_volatility_signal(self, features: pd.DataFrame) -> float:
        """Calculate volatility-based signal component."""
        # Volatility compression/expansion signal
        if 'vol_ratio' in features.columns:
            vol_ratio = features['vol_ratio'].iloc[-1]

            # Low vol ratio = compression, potential breakout
            if vol_ratio < 0.7:
                return 0.3  # Slight bullish bias (breakouts often bullish)
            # High vol ratio = expansion, potential reversal
            elif vol_ratio > 1.3:
                return -0.2

        return 0.0

    # =========================================================================
    # Utility Methods
    # =========================================================================

    def validate_prices(self, prices: Union[pd.Series, pd.DataFrame]) -> bool:
        """Validate price data quality."""
        if prices is None or (hasattr(prices, 'empty') and prices.empty):
            return False

        if isinstance(prices, pd.DataFrame):
            close = prices.get('close', prices.get('Close', pd.Series()))
        else:
            close = prices

        if len(close) < self.lookback_long:
            return False

        # Check for NaN/Inf
        if close.isna().any() or np.isinf(close).any():
            return False

        return True

    def create_signal_output(
        self,
        signal: float,
        confidence: float,
        regime: MarketRegime,
        momentum_signal: float = 0.0,
        mean_reversion_signal: float = 0.0,
        trend_signal: float = 0.0,
        volatility_signal: float = 0.0,
        ml_prediction: float = 0.0,
        model_agreement: float = 1.0
    ) -> Signal:
        """Create standardized Signal output."""
        return Signal(
            signal=self.normalize_signal(signal),
            confidence=confidence,
            direction=self.classify_direction(signal),
            regime=regime,
            timestamp=datetime.now(),
            momentum_signal=momentum_signal,
            mean_reversion_signal=mean_reversion_signal,
            trend_signal=trend_signal,
            volatility_signal=volatility_signal,
            ml_prediction=ml_prediction,
            model_agreement=model_agreement
        )
