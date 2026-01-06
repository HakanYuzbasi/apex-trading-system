"""
models/feature_engineering.py
STATE-OF-THE-ART FEATURE ENGINEERING
- 50+ technical features
- Statistical features
- Market microstructure
- Feature selection
"""

import numpy as np
import pandas as pd
import logging
from typing import List, Dict, Tuple
from sklearn.feature_selection import mutual_info_classif
from scipy.stats import skew, kurtosis

logger = logging.getLogger(__name__)


class FeatureEngineering:
    """
    Advanced feature engineering for ML trading models.
    
    Creates 50+ features across categories:
    - Price action
    - Volatility
    - Momentum
    - Mean reversion
    - Volume
    - Market microstructure
    """
    
    def __init__(self):
        self.feature_names = []
        logger.info("✅ Feature Engineering initialized")
    
    def create_all_features(self, ohlcv: pd.DataFrame) -> pd.DataFrame:
        """
        Create comprehensive feature set.
        
        Args:
            ohlcv: DataFrame with OHLCV data
        
        Returns:
            DataFrame with 50+ engineered features
        """
        df = ohlcv.copy()
        
        # Ensure we have required columns
        required = ['Open', 'High', 'Low', 'Close', 'Volume']
        if not all(col in df.columns for col in required):
            raise ValueError(f"Missing required columns. Need: {required}")
        
        logger.info(f"Creating features for {len(df)} bars...")
        
        # 1. Price Features
        df = self._add_price_features(df)
        
        # 2. Volatility Features
        df = self._add_volatility_features(df)
        
        # 3. Momentum Features
        df = self._add_momentum_features(df)
        
        # 4. Mean Reversion Features
        df = self._add_mean_reversion_features(df)
        
        # 5. Volume Features
        df = self._add_volume_features(df)
        
        # 6. Statistical Features
        df = self._add_statistical_features(df)
        
        # 7. Autocorrelation Features
        df = self._add_autocorrelation_features(df)
        
        # 8. Market Microstructure
        df = self._add_microstructure_features(df)
        
        # Remove NaN rows (from rolling windows)
        initial_len = len(df)
        df = df.dropna()
        logger.info(f"✅ Created {len(df.columns)} features ({initial_len - len(df)} rows dropped)")
        
        self.feature_names = [col for col in df.columns if col not in required]
        
        return df
    
    def _add_price_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Basic price-based features."""
        df['returns'] = df['Close'].pct_change()
        df['log_returns'] = np.log(df['Close'] / df['Close'].shift(1))
        df['high_low_ratio'] = df['High'] / df['Low']
        df['close_open_ratio'] = df['Close'] / df['Open']
        df['price_range'] = (df['High'] - df['Low']) / df['Close']
        df['body_size'] = abs(df['Close'] - df['Open']) / df['Close']
        df['upper_shadow'] = (df['High'] - df[['Open', 'Close']].max(axis=1)) / df['Close']
        df['lower_shadow'] = (df[['Open', 'Close']].min(axis=1) - df['Low']) / df['Close']
        
        return df
    
    def _add_volatility_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Volatility-based features (MOST IMPORTANT for ML)."""
        # Historical volatility
        df['volatility_5'] = df['returns'].rolling(5).std()
        df['volatility_10'] = df['returns'].rolling(10).std()
        df['volatility_20'] = df['returns'].rolling(20).std()
        df['volatility_60'] = df['returns'].rolling(60).std()
        
        # Volatility ratios
        df['vol_ratio_20_60'] = df['volatility_20'] / df['volatility_60']
        df['vol_ratio_5_20'] = df['volatility_5'] / df['volatility_20']
        
        # Parkinson's volatility (uses high-low range)
        df['parkinson_vol'] = np.sqrt(
            np.log(df['High'] / df['Low'])**2 / (4 * np.log(2))
        ).rolling(20).mean()
        
        # Garman-Klass volatility (more efficient estimator)
        df['gk_vol'] = np.sqrt(
            0.5 * np.log(df['High'] / df['Low'])**2 - 
            (2 * np.log(2) - 1) * np.log(df['Close'] / df['Open'])**2
        ).rolling(20).mean()
        
        # Exponentially weighted volatility
        df['ewm_vol'] = df['returns'].ewm(span=20).std()
        
        return df
    
    def _add_momentum_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Momentum and trend features."""
        # Price momentum
        df['momentum_5'] = df['Close'] - df['Close'].shift(5)
        df['momentum_10'] = df['Close'] - df['Close'].shift(10)
        df['momentum_20'] = df['Close'] - df['Close'].shift(20)
        df['momentum_60'] = df['Close'] - df['Close'].shift(60)
        
        # Rate of change
        df['roc_5'] = (df['Close'] / df['Close'].shift(5) - 1) * 100
        df['roc_10'] = (df['Close'] / df['Close'].shift(10) - 1) * 100
        df['roc_20'] = (df['Close'] / df['Close'].shift(20) - 1) * 100
        
        # RSI
        df['rsi_14'] = self._calculate_rsi(df['Close'], 14)
        df['rsi_divergence'] = df['rsi_14'].diff()
        
        # MACD
        exp1 = df['Close'].ewm(span=12).mean()
        exp2 = df['Close'].ewm(span=26).mean()
        df['macd'] = exp1 - exp2
        df['macd_signal'] = df['macd'].ewm(span=9).mean()
        df['macd_histogram'] = df['macd'] - df['macd_signal']
        
        # ADX (trend strength)
        df['adx'] = self._calculate_adx(df, 14)
        
        return df
    
    def _add_mean_reversion_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Mean reversion signals."""
        # Moving averages
        df['sma_5'] = df['Close'].rolling(5).mean()
        df['sma_10'] = df['Close'].rolling(10).mean()
        df['sma_20'] = df['Close'].rolling(20).mean()
        df['sma_60'] = df['Close'].rolling(60).mean()
        df['sma_200'] = df['Close'].rolling(200).mean()
        
        # Distance from MA
        df['dist_sma_20'] = (df['Close'] - df['sma_20']) / df['sma_20']
        df['dist_sma_60'] = (df['Close'] - df['sma_60']) / df['sma_60']
        
        # Bollinger Bands
        df['bb_upper'] = df['sma_20'] + 2 * df['volatility_20'] * df['Close']
        df['bb_lower'] = df['sma_20'] - 2 * df['volatility_20'] * df['Close']
        df['bb_position'] = (df['Close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
        df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['sma_20']
        
        # Z-score
        df['zscore_20'] = (df['Close'] - df['sma_20']) / df['volatility_20']
        
        return df
    
    def _add_volume_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Volume-based features."""
        # Volume metrics
        df['volume_sma'] = df['Volume'].rolling(20).mean()
        df['volume_ratio'] = df['Volume'] / df['volume_sma']
        df['volume_change'] = df['Volume'].pct_change()
        
        # On-Balance Volume (OBV)
        df['obv'] = (np.sign(df['Close'].diff()) * df['Volume']).fillna(0).cumsum()
        df['obv_slope'] = df['obv'].diff(5)
        
        # Volume-Price Trend
        df['vpt'] = ((df['Close'].diff() / df['Close'].shift(1)) * df['Volume']).fillna(0).cumsum()
        
        # Money Flow Index (MFI)
        df['mfi'] = self._calculate_mfi(df, 14)
        
        return df
    
    def _add_statistical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Statistical distribution features."""
        # Skewness (asymmetry of returns)
        df['skew_20'] = df['returns'].rolling(20).apply(lambda x: skew(x), raw=True)
        df['skew_60'] = df['returns'].rolling(60).apply(lambda x: skew(x), raw=True)
        
        # Kurtosis (fat tails indicator)
        df['kurt_20'] = df['returns'].rolling(20).apply(lambda x: kurtosis(x), raw=True)
        df['kurt_60'] = df['returns'].rolling(60).apply(lambda x: kurtosis(x), raw=True)
        
        # Percentile ranks
        df['percentile_rank_20'] = df['Close'].rolling(20).apply(
            lambda x: pd.Series(x).rank(pct=True).iloc[-1]
        )
        
        return df
    
    def _add_autocorrelation_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Autocorrelation features (trend persistence)."""
        df['autocorr_1'] = df['returns'].rolling(20).apply(
            lambda x: x.autocorr(lag=1), raw=False
        )
        df['autocorr_5'] = df['returns'].rolling(20).apply(
            lambda x: x.autocorr(lag=5), raw=False
        )
        df['autocorr_10'] = df['returns'].rolling(20).apply(
            lambda x: x.autocorr(lag=10), raw=False
        )
        
        return df
    
    def _add_microstructure_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Market microstructure features."""
        # Amihud illiquidity ratio
        df['amihud_illiquidity'] = abs(df['returns']) / (df['Volume'] * df['Close'] + 1e-10)
        
        # Roll's spread estimate
        df['roll_spread'] = 2 * np.sqrt(-df['returns'].rolling(20).apply(
            lambda x: x.autocorr(lag=1) if x.autocorr(lag=1) < 0 else 0
        ))
        
        # Tick direction (buying vs selling pressure)
        df['tick_direction'] = np.sign(df['Close'] - df['Close'].shift(1))
        df['tick_momentum'] = df['tick_direction'].rolling(10).sum()
        
        return df
    
    @staticmethod
    def _calculate_rsi(prices: pd.Series, period: int = 14) -> pd.Series:
        """Calculate RSI indicator."""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        
        rs = gain / (loss + 1e-10)
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    @staticmethod
    def _calcula
