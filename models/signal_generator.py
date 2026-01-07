"""
models/signal_generator.py - Extensible Signal Generation Framework

This module provides a plugin-based signal generator architecture that makes
it easy to add new ML-based signals without modifying core logic.

Architecture:
- SignalPlugin: Base class for all signal generators
- SignalRegistry: Manages registered signal plugins
- SignalGenerator: Main class that orchestrates signal generation
"""

import numpy as np
import pandas as pd
from abc import ABC, abstractmethod
from typing import Dict, Optional, List, Type, Callable
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


# =============================================================================
# SIGNAL PLUGIN ARCHITECTURE
# =============================================================================

class SignalPlugin(ABC):
    """
    Base class for signal generation plugins.

    To create a new signal plugin:
    1. Inherit from SignalPlugin
    2. Implement the `generate` method
    3. Set the `name` and `weight` attributes
    4. Register with SignalRegistry.register()

    Example:
        class MyCustomSignal(SignalPlugin):
            name = "my_custom"
            weight = 0.25

            def generate(self, prices: pd.Series, **kwargs) -> float:
                # Your signal logic here
                return 0.5
    """

    name: str = "base"
    weight: float = 0.0
    requires_training: bool = False

    @abstractmethod
    def generate(self, prices: pd.Series, **kwargs) -> float:
        """
        Generate a signal from price data.

        Args:
            prices: Series of closing prices
            **kwargs: Additional data (e.g., volume, high, low)

        Returns:
            Signal between -1 (bearish) and 1 (bullish)
        """
        pass

    def train(self, historical_data: Dict[str, pd.DataFrame]) -> None:
        """
        Optional training method for ML-based signals.

        Args:
            historical_data: Dict of {symbol: DataFrame} with OHLCV data
        """
        pass

    def is_ready(self) -> bool:
        """Check if the signal plugin is ready to generate signals."""
        return True


class SignalRegistry:
    """
    Registry for signal plugins.

    Allows dynamic registration and discovery of signal generators.
    """

    _plugins: Dict[str, Type[SignalPlugin]] = {}
    _instances: Dict[str, SignalPlugin] = {}

    @classmethod
    def register(cls, plugin_class: Type[SignalPlugin]) -> Type[SignalPlugin]:
        """
        Register a signal plugin class.

        Can be used as a decorator:
            @SignalRegistry.register
            class MySignal(SignalPlugin):
                ...
        """
        name = plugin_class.name
        cls._plugins[name] = plugin_class
        logger.debug(f"Registered signal plugin: {name}")
        return plugin_class

    @classmethod
    def get(cls, name: str) -> Optional[SignalPlugin]:
        """Get an instance of a registered plugin."""
        if name not in cls._instances:
            if name in cls._plugins:
                cls._instances[name] = cls._plugins[name]()
            else:
                return None
        return cls._instances[name]

    @classmethod
    def get_all(cls) -> List[SignalPlugin]:
        """Get instances of all registered plugins."""
        for name, plugin_class in cls._plugins.items():
            if name not in cls._instances:
                cls._instances[name] = plugin_class()
        return list(cls._instances.values())

    @classmethod
    def list_plugins(cls) -> List[str]:
        """List all registered plugin names."""
        return list(cls._plugins.keys())


# =============================================================================
# BUILT-IN SIGNAL PLUGINS
# =============================================================================

@SignalRegistry.register
class MomentumSignal(SignalPlugin):
    """Momentum-based signal using price rate of change."""

    name = "momentum"
    weight = 0.30

    def __init__(self, lookback: int = 20):
        self.lookback = lookback

    def generate(self, prices: pd.Series, **kwargs) -> float:
        if len(prices) < self.lookback:
            return 0.0

        try:
            returns = prices.pct_change(self.lookback).iloc[-1]
            if pd.isna(returns):
                return 0.0
            return float(np.tanh(returns * 10))
        except Exception as e:
            logger.debug(f"Error in momentum signal: {e}")
            return 0.0


@SignalRegistry.register
class MeanReversionSignal(SignalPlugin):
    """Mean reversion signal using z-score."""

    name = "mean_reversion"
    weight = 0.20

    def __init__(self, lookback: int = 20):
        self.lookback = lookback

    def generate(self, prices: pd.Series, **kwargs) -> float:
        if len(prices) < self.lookback:
            return 0.0

        try:
            mean = prices.rolling(self.lookback).mean().iloc[-1]
            std = prices.rolling(self.lookback).std().iloc[-1]

            if std == 0 or pd.isna(std) or pd.isna(mean):
                return 0.0

            z_score = (prices.iloc[-1] - mean) / std
            return float(-np.tanh(z_score))  # Buy when oversold
        except Exception as e:
            logger.debug(f"Error in mean reversion signal: {e}")
            return 0.0


@SignalRegistry.register
class TrendSignal(SignalPlugin):
    """Trend-following signal based on moving average crossover."""

    name = "trend"
    weight = 0.30

    def __init__(self, short_period: int = 20, long_period: int = 50):
        self.short_period = short_period
        self.long_period = long_period

    def generate(self, prices: pd.Series, **kwargs) -> float:
        if len(prices) < self.long_period:
            return 0.0

        try:
            ma_short = prices.rolling(self.short_period).mean().iloc[-1]
            ma_long = prices.rolling(self.long_period).mean().iloc[-1]

            if ma_long == 0 or pd.isna(ma_short) or pd.isna(ma_long):
                return 0.0

            trend = (ma_short - ma_long) / ma_long
            return float(np.tanh(trend * 20))
        except Exception as e:
            logger.debug(f"Error in trend signal: {e}")
            return 0.0


@SignalRegistry.register
class RSISignal(SignalPlugin):
    """Signal based on Relative Strength Index."""

    name = "rsi"
    weight = 0.20

    def __init__(self, period: int = 14):
        self.period = period

    def generate(self, prices: pd.Series, **kwargs) -> float:
        if len(prices) < self.period + 1:
            return 0.0

        try:
            delta = prices.diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=self.period).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=self.period).mean()

            gain_val = gain.iloc[-1]
            loss_val = loss.iloc[-1]

            if loss_val == 0 or pd.isna(gain_val) or pd.isna(loss_val):
                return 0.0

            rs = gain_val / loss_val
            rsi = 100 - (100 / (1 + rs))

            # RSI < 30 = buy, RSI > 70 = sell
            normalized_rsi = (rsi - 50) / 50
            return float(-normalized_rsi)
        except Exception as e:
            logger.debug(f"Error in RSI signal: {e}")
            return 0.0


# =============================================================================
# MAIN SIGNAL GENERATOR
# =============================================================================

class SignalGenerator:
    """
    Main signal generator that orchestrates multiple signal plugins.

    Features:
    - Plugin-based architecture for easy extension
    - Weighted ensemble of signals
    - Confidence calculation based on signal agreement
    - Support for ML model training
    """

    def __init__(self, lookback: int = 20, use_plugins: Optional[List[str]] = None):
        """
        Initialize the signal generator.

        Args:
            lookback: Default lookback period for indicators
            use_plugins: List of plugin names to use (None = use all)
        """
        self.lookback = lookback
        self.use_plugins = use_plugins

        # Initialize active plugins
        self._active_plugins: List[SignalPlugin] = []
        self._load_plugins()

        logger.info(f"SignalGenerator initialized with {len(self._active_plugins)} plugins")
        for plugin in self._active_plugins:
            logger.debug(f"  - {plugin.name} (weight={plugin.weight})")

    def _load_plugins(self):
        """Load and initialize signal plugins."""
        all_plugins = SignalRegistry.get_all()

        if self.use_plugins:
            self._active_plugins = [p for p in all_plugins if p.name in self.use_plugins]
        else:
            self._active_plugins = all_plugins

        # Normalize weights
        total_weight = sum(p.weight for p in self._active_plugins)
        if total_weight > 0:
            for p in self._active_plugins:
                p.weight = p.weight / total_weight

    def add_plugin(self, plugin: SignalPlugin):
        """
        Add a custom plugin at runtime.

        Args:
            plugin: Instance of SignalPlugin
        """
        self._active_plugins.append(plugin)

        # Renormalize weights
        total_weight = sum(p.weight for p in self._active_plugins)
        if total_weight > 0:
            for p in self._active_plugins:
                p.weight = p.weight / total_weight

        logger.info(f"Added plugin: {plugin.name}")

    def train_plugins(self, historical_data: Dict[str, pd.DataFrame]):
        """
        Train all plugins that require training.

        Args:
            historical_data: Dict of {symbol: DataFrame} with OHLCV data
        """
        for plugin in self._active_plugins:
            if plugin.requires_training:
                logger.info(f"Training plugin: {plugin.name}")
                try:
                    plugin.train(historical_data)
                except Exception as e:
                    logger.error(f"Error training {plugin.name}: {e}")

    def generate_ml_signal(self, symbol: str, prices: Optional[pd.Series] = None) -> Dict:
        """
        Generate combined signal using all active plugins.

        Args:
            symbol: Stock ticker symbol (for logging)
            prices: Series of closing prices

        Returns:
            Dictionary with signal, confidence, and component signals
        """
        result = {
            'signal': 0.0,
            'confidence': 0.0,
            'components': {},
            'timestamp': datetime.now()
        }

        if prices is None or len(prices) == 0:
            logger.warning(f"{symbol}: No price data provided")
            return result

        # Ensure prices is a Series
        if isinstance(prices, pd.DataFrame):
            if 'Close' in prices.columns:
                prices = prices['Close']
            else:
                prices = prices.iloc[:, 0]

        # Generate signals from each plugin
        signals = []
        for plugin in self._active_plugins:
            if not plugin.is_ready():
                continue

            try:
                signal = plugin.generate(prices)
                signals.append((plugin.name, signal, plugin.weight))
                result['components'][plugin.name] = float(signal)
            except Exception as e:
                logger.debug(f"Error in {plugin.name}: {e}")
                result['components'][plugin.name] = 0.0

        # Calculate weighted ensemble signal
        if signals:
            weighted_sum = sum(s * w for _, s, w in signals)
            result['signal'] = float(np.clip(weighted_sum, -1, 1))

            # Calculate confidence based on agreement
            non_zero = [(s, w) for _, s, w in signals if s != 0]
            if non_zero and result['signal'] != 0:
                same_sign = sum(w for s, w in non_zero if np.sign(s) == np.sign(result['signal']))
                total_weight = sum(w for _, w in non_zero)
                agreement = same_sign / total_weight if total_weight > 0 else 0
                result['confidence'] = float(min(abs(result['signal']) * agreement, 1.0))

        # Add legacy fields for backward compatibility
        result['momentum'] = result['components'].get('momentum', 0.0)
        result['mean_reversion'] = result['components'].get('mean_reversion', 0.0)
        result['trend'] = result['components'].get('trend', 0.0)
        result['rsi'] = result['components'].get('rsi', 0.0)

        return result

    def generate_signal_for_backtest(self, prices: pd.Series) -> float:
        """
        Simplified signal for backtesting.

        Args:
            prices: Series of closing prices

        Returns:
            Signal between -1 and 1
        """
        result = self.generate_ml_signal("BACKTEST", prices)
        return result['signal']

    def get_plugin_info(self) -> List[Dict]:
        """Get information about active plugins."""
        return [
            {
                'name': p.name,
                'weight': p.weight,
                'requires_training': p.requires_training,
                'is_ready': p.is_ready()
            }
            for p in self._active_plugins
        ]


# =============================================================================
# EXAMPLE: CUSTOM ML PLUGIN (Template for users)
# =============================================================================

class MLSignalPlugin(SignalPlugin):
    """
    Template for ML-based signal plugins.

    To use:
    1. Copy this class and rename it
    2. Implement the train() and generate() methods
    3. Register with @SignalRegistry.register decorator
    """

    name = "ml_template"
    weight = 0.0  # Set to 0 to disable by default
    requires_training = True

    def __init__(self):
        self.model = None
        self._is_trained = False

    def train(self, historical_data: Dict[str, pd.DataFrame]) -> None:
        """
        Train your ML model here.

        Example using sklearn:
            from sklearn.ensemble import RandomForestClassifier

            X, y = self._prepare_features(historical_data)
            self.model = RandomForestClassifier()
            self.model.fit(X, y)
            self._is_trained = True
        """
        # Implement your training logic
        logger.info(f"Training {self.name} model...")
        self._is_trained = True

    def generate(self, prices: pd.Series, **kwargs) -> float:
        """
        Generate signal using trained model.

        Example:
            features = self._extract_features(prices)
            prediction = self.model.predict_proba(features)[0]
            return prediction[1] - prediction[0]  # Bullish - Bearish probability
        """
        if not self._is_trained:
            return 0.0

        # Implement your prediction logic
        return 0.0

    def is_ready(self) -> bool:
        """Check if model is trained and ready."""
        return self._is_trained and self.model is not None
