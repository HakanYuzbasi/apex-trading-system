"""
models/signal_generator.py - Basic Signal Generation

This module provides a simpler signal generator that can be used
for backtesting or when ML models are not available.
"""

import numpy as np
import pandas as pd
from typing import Dict, Optional
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


class SignalGenerator:
    """
    Generate trading signals using technical analysis strategies.

    This is a simpler alternative to AdvancedSignalGenerator that doesn't
    require ML training but still provides useful signals.
    """

    def __init__(self, lookback: int = 20):
        """
        Initialize the signal generator.

        Args:
            lookback: Default lookback period for indicators
        """
        self.lookback = lookback
        logger.info(f"SignalGenerator initialized (lookback={lookback})")

    def generate_momentum_signal(self, prices: pd.Series) -> float:
        """
        Generate momentum-based signal.

        Uses price rate of change over the lookback period.

        Args:
            prices: Series of closing prices

        Returns:
            Signal between -1 (bearish) and 1 (bullish)
        """
        if len(prices) < self.lookback:
            return 0.0

        try:
            returns = prices.pct_change(self.lookback).iloc[-1]
            if pd.isna(returns):
                return 0.0
            return float(np.tanh(returns * 10))
        except Exception as e:
            logger.debug(f"Error calculating momentum signal: {e}")
            return 0.0

    def generate_mean_reversion_signal(self, prices: pd.Series) -> float:
        """
        Generate mean reversion signal.

        Uses z-score to identify overbought/oversold conditions.

        Args:
            prices: Series of closing prices

        Returns:
            Signal between -1 (sell/short) and 1 (buy/cover)
        """
        if len(prices) < self.lookback:
            return 0.0

        try:
            mean = prices.rolling(self.lookback).mean().iloc[-1]
            std = prices.rolling(self.lookback).std().iloc[-1]

            if std == 0 or pd.isna(std) or pd.isna(mean):
                return 0.0

            z_score = (prices.iloc[-1] - mean) / std
            # Negative because we want to buy when oversold (low z-score)
            return float(-np.tanh(z_score))
        except Exception as e:
            logger.debug(f"Error calculating mean reversion signal: {e}")
            return 0.0

    def generate_trend_signal(self, prices: pd.Series) -> float:
        """
        Generate trend-following signal based on moving average crossover.

        Args:
            prices: Series of closing prices

        Returns:
            Signal between -1 (bearish trend) and 1 (bullish trend)
        """
        if len(prices) < 50:
            return 0.0

        try:
            ma_short = prices.rolling(20).mean().iloc[-1]
            ma_long = prices.rolling(50).mean().iloc[-1]

            if ma_long == 0 or pd.isna(ma_short) or pd.isna(ma_long):
                return 0.0

            trend = (ma_short - ma_long) / ma_long
            return float(np.tanh(trend * 20))
        except Exception as e:
            logger.debug(f"Error calculating trend signal: {e}")
            return 0.0

    def generate_rsi_signal(self, prices: pd.Series, period: int = 14) -> float:
        """
        Generate signal based on RSI (Relative Strength Index).

        Args:
            prices: Series of closing prices
            period: RSI calculation period

        Returns:
            Signal between -1 (overbought) and 1 (oversold)
        """
        if len(prices) < period + 1:
            return 0.0

        try:
            delta = prices.diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()

            gain_val = gain.iloc[-1]
            loss_val = loss.iloc[-1]

            if loss_val == 0 or pd.isna(gain_val) or pd.isna(loss_val):
                return 0.0

            rs = gain_val / loss_val
            rsi = 100 - (100 / (1 + rs))

            # Convert RSI to signal: RSI < 30 = buy (1), RSI > 70 = sell (-1)
            normalized_rsi = (rsi - 50) / 50  # -1 to 1 range
            return float(-normalized_rsi)  # Negative because high RSI = sell
        except Exception as e:
            logger.debug(f"Error calculating RSI signal: {e}")
            return 0.0

    def generate_ml_signal(self, symbol: str, prices: Optional[pd.Series] = None) -> Dict:
        """
        Generate combined ML-style signal using ensemble of strategies.

        This method provides a consistent interface with AdvancedSignalGenerator.

        Args:
            symbol: Stock ticker symbol (for logging)
            prices: Series of closing prices (required for actual signals)

        Returns:
            Dictionary with signal, confidence, and component signals
        """
        # If no prices provided, return neutral signal
        if prices is None or len(prices) == 0:
            logger.warning(f"{symbol}: No price data provided to generate_ml_signal")
            return {
                'signal': 0.0,
                'confidence': 0.0,
                'momentum': 0.0,
                'mean_reversion': 0.0,
                'trend': 0.0,
                'rsi': 0.0,
                'timestamp': datetime.now()
            }

        # Ensure prices is a Series
        if isinstance(prices, pd.DataFrame):
            if 'Close' in prices.columns:
                prices = prices['Close']
            else:
                prices = prices.iloc[:, 0]

        # Generate component signals
        momentum = self.generate_momentum_signal(prices)
        mean_rev = self.generate_mean_reversion_signal(prices)
        trend = self.generate_trend_signal(prices)
        rsi = self.generate_rsi_signal(prices)

        # Ensemble weights (can be tuned)
        weights = {
            'momentum': 0.30,
            'mean_reversion': 0.20,
            'trend': 0.30,
            'rsi': 0.20
        }

        # Calculate weighted signal
        signal = (
            weights['momentum'] * momentum +
            weights['mean_reversion'] * mean_rev +
            weights['trend'] * trend +
            weights['rsi'] * rsi
        )

        # Clip signal to [-1, 1]
        signal = float(np.clip(signal, -1, 1))

        # Calculate confidence based on component agreement
        components = [momentum, mean_rev, trend, rsi]
        non_zero = [c for c in components if c != 0]

        if non_zero:
            # Agreement: how many components have the same sign as the signal
            same_sign = sum(1 for c in non_zero if np.sign(c) == np.sign(signal))
            agreement = same_sign / len(non_zero)
            confidence = float(min(abs(signal) * agreement, 1.0))
        else:
            confidence = 0.0

        return {
            'signal': signal,
            'confidence': confidence,
            'momentum': float(momentum),
            'mean_reversion': float(mean_rev),
            'trend': float(trend),
            'rsi': float(rsi),
            'timestamp': datetime.now()
        }

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
