"""
models/advanced_signal_generator.py - GOD LEVEL ADVANCED
================================================================================
APEX TRADING SYSTEM - 30+ FEATURES WITH REGIME AWARENESS

Features:
✅ Vectorized Feature Extraction (1000x faster)
✅ Walk-Forward Validation
✅ Regime-Aware Training (Separate models per regime)
✅ Drift Monitoring
✅ Model Persistence
"""

import pandas as pd
import numpy as np
from typing import Deque, Dict, List, Optional, Tuple, Union
from datetime import datetime, timedelta
import logging
import os
import pickle
from collections import deque
from joblib import dump, load

from core.symbols import parse_symbol, AssetClass
from config import ApexConfig
from models.regime_common import get_regime
from models.binary_signal_classifier import BinarySignalClassifier

logger = logging.getLogger(__name__)

# ML imports
ML_AVAILABLE = False
XGBOOST_AVAILABLE = False
LIGHTGBM_AVAILABLE = False

try:
    from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
    from sklearn.preprocessing import RobustScaler
    from sklearn.impute import SimpleImputer
    import warnings
    warnings.filterwarnings('ignore')
    ML_AVAILABLE = True
except ImportError:
    logger.warning("scikit-learn not available")

try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    pass

try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    pass


# RegimeDetector class removed - using models.regime_common.get_regime


class AdvancedSignalGenerator:
    """
    GOD LEVEL: 30+ Feature Advanced ML Signal Generator.
    """
    
    def __init__(self, model_dir: str = "models/saved_advanced"):
        self.model_dir = model_dir
        os.makedirs(model_dir, exist_ok=True)
        
        self.lookback = 300  # Need 300 bars for 200-day MA
        
        # Regime-specific models
        self.regime_models: Dict[str, Dict] = {
            'bull': {},
            'bear': {},
            'neutral': {},
            'volatile': {}
        }

        # Asset-class-specific regime models (equity/forex/crypto)
        self.asset_class_models: Dict[str, Dict[str, Dict]] = {
            AssetClass.EQUITY.value: {'bull': {}, 'bear': {}, 'neutral': {}, 'volatile': {}},
            AssetClass.FOREX.value: {'bull': {}, 'bear': {}, 'neutral': {}, 'volatile': {}},
            AssetClass.CRYPTO.value: {'bull': {}, 'bear': {}, 'neutral': {}, 'volatile': {}},
        }
        
        # Preprocessors per regime
        self.regime_scalers: Dict[str, RobustScaler] = {}
        self.regime_imputers: Dict[str, SimpleImputer] = {}

        # Preprocessors per asset class + regime
        self.asset_class_scalers: Dict[str, Dict[str, RobustScaler]] = {
            AssetClass.EQUITY.value: {},
            AssetClass.FOREX.value: {},
            AssetClass.CRYPTO.value: {},
        }
        self.asset_class_imputers: Dict[str, Dict[str, SimpleImputer]] = {
            AssetClass.EQUITY.value: {},
            AssetClass.FOREX.value: {},
            AssetClass.CRYPTO.value: {},
        }
        
        self.feature_names = []
        self.is_trained = False
        self.training_date = None
        self.last_retrain_date = None
        
        # Drift monitoring
        self.prediction_history = deque(maxlen=100)
        self.outcome_history = deque(maxlen=100)
        self.performance_baseline = 0.52
        self.retrain_interval_days = 30

        # Prediction-distribution drift: rolling buffer of *raw* model outputs
        # (pre-clip, pre-EMA) sampled at inference. Compared against the
        # training-time baseline (mean, std) persisted in model metadata to
        # catch silent distribution shifts. Training code populates
        # ``self.pred_baseline_mean/std``; absence disables the check.
        self._drift_window: int = max(
            0, int(getattr(ApexConfig, "ML_DRIFT_WINDOW_BARS", 0))
        )
        self._drift_buffer: Deque[float] = deque(maxlen=self._drift_window or 1)
        self.pred_baseline_mean: float = float(
            getattr(ApexConfig, "ML_BASELINE_PRED_MEAN", 0.0)
        )
        self.pred_baseline_std: float = float(
            getattr(ApexConfig, "ML_BASELINE_PRED_STD", 0.0)
        )
        self.signal_ema: Dict[str, float] = {}
        self.signal_ema_alpha = 0.35  # Smoothing to reduce noise for better Sharpe

        # Online learning: last feature vector per symbol (populated at inference,
        # consumed when the trade closes to do an incremental model update).
        self._last_features: Dict[str, np.ndarray] = {}
        # Rolling online training buffer per (asset_class, regime): [(X_row, y_val), ...]
        self._online_buffer: Dict[str, List[tuple]] = {}
        self._online_buffer_maxsize: int = 60
        self._online_update_trigger: int = int(getattr(ApexConfig, "ONLINE_UPDATE_TRIGGER_N", 20))

        # Statistical feature flags
        self.hurst_feature_enabled: bool = bool(getattr(ApexConfig, "HURST_FEATURE_ENABLED", True))
        self._hurst_lags: int = int(getattr(ApexConfig, "HURST_LAGS", 20))

        # Binary direction classifier (runs alongside regression ensemble)
        self._binary_enabled: bool = bool(getattr(ApexConfig, "BINARY_SIGNAL_ENABLED", True))
        self._binary_weight: float = float(getattr(ApexConfig, "BINARY_SIGNAL_WEIGHT", 0.40))
        self._binary_label_horizon: int = int(getattr(ApexConfig, "BINARY_LABEL_HORIZON_DAYS", 1))
        self._binary_clf: Optional[BinarySignalClassifier] = None
        if self._binary_enabled and ML_AVAILABLE:
            try:
                self._binary_clf = BinarySignalClassifier(
                    model_dir=os.path.join(model_dir, "binary")
                )
            except Exception as _e:
                logger.warning("BinarySignalClassifier init failed (non-fatal): %s", _e)

        # Try to load existing models
        self._load_models()

        logger.info("✅ GOD LEVEL Advanced Signal Generator initialized")
    
    def _intraday_momentum_features(
        self,
        _daily_df: Union[pd.Series, pd.DataFrame],
        intraday_df: Optional[pd.DataFrame] = None,
    ) -> dict:
        """
        Compute intraday momentum features from 5-minute bar data.

        Args:
            daily_df: Daily OHLCV DataFrame (used for fallback close price)
            intraday_df: Optional DataFrame of intraday bars with
                         Open/High/Low/Close/Volume columns

        Returns:
            Dict with keys:
              intra_vwap_deviation  – (close - VWAP) / VWAP using today's bars
              intra_trend_5bar      – normalised linear-regression slope of last 5 bars
              intra_vol_ratio       – current session vol vs prior session vol
            All values default to 0.0 when intraday data is unavailable.
        """
        result = {
            "intra_vwap_deviation": 0.0,
            "intra_trend_5bar": 0.0,
            "intra_vol_ratio": 0.0,
        }

        if not getattr(ApexConfig, "INTRADAY_FEATURES_ENABLED", True):
            return result

        if intraday_df is None or intraday_df.empty:
            return result

        if "Close" not in intraday_df.columns:
            return result

        try:
            closes = intraday_df["Close"].dropna()
            if closes.empty:
                return result

            current_close = float(closes.iloc[-1])

            # --- VWAP deviation ---
            if "Volume" in intraday_df.columns:
                vol = intraday_df["Volume"].reindex(closes.index).fillna(0)
                total_vol = float(vol.sum())
                if total_vol > 0:
                    vwap = float((closes * vol).sum() / total_vol)
                    if vwap != 0:
                        result["intra_vwap_deviation"] = float(
                            np.clip((current_close - vwap) / vwap, -0.10, 0.10)
                        )

            # --- Trend of last 5 bars (linear regression slope, normalised) ---
            if len(closes) >= 5:
                last5 = closes.iloc[-5:].values.astype(float)
                x = np.arange(len(last5), dtype=float)
                # Normalise slope by mean price so it is scale-free
                mean_px = float(np.mean(last5)) if np.mean(last5) != 0 else 1.0
                slope = float(np.polyfit(x, last5, 1)[0]) / mean_px
                result["intra_trend_5bar"] = float(np.clip(slope, -0.05, 0.05))

            # --- Vol ratio: current session vs prior session ---
            if len(closes) >= 20:
                mid = len(closes) // 2
                prior_vol = float(closes.iloc[:mid].pct_change().dropna().std())
                curr_vol = float(closes.iloc[mid:].pct_change().dropna().std())
                if prior_vol > 0:
                    result["intra_vol_ratio"] = float(
                        np.clip(curr_vol / prior_vol, 0.0, 5.0)
                    )

        except Exception:
            pass  # Return zeros on any computation error

        return result

    def compute_features_vectorized(
        self,
        data: Union[pd.Series, pd.DataFrame],
        sentiment_score: Optional[float] = None,
        momentum_rank: Optional[float] = None,
        intraday_df: Optional[pd.DataFrame] = None,
    ) -> pd.DataFrame:
        """Vectorized 40+ feature extraction with OHLCV and Context."""
        if isinstance(data, pd.Series):
            p = data
            df = pd.DataFrame(index=p.index)
            v, h, low_px, _o = None, None, None, None
        else:
            p = data['Close']
            df = pd.DataFrame(index=p.index)
            v = data.get('Volume')
            h = data.get('High')
            low_px = data.get('Low')
            data.get('Open')

        returns = p.pct_change()
        
        # 1. Multi-period returns
        for d in [3, 5, 10, 20, 60]:
            df[f'ret_{d}d'] = p.pct_change(d)
        
        # 2. Volatility
        for d in [10, 20, 60]:
            df[f'vol_{d}d'] = returns.rolling(d).std() * np.sqrt(252)
        
        # 3. RSI
        delta = p.diff()
        gain = delta.where(delta > 0, 0).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / loss.replace(0, np.nan)
        rsi = 100 - (100 / (1 + rs.fillna(0)))
        df['rsi_14'] = rsi
        df['rsi_norm'] = (rsi - 50) / 50
        
        # 4. MACD
        ema12 = p.ewm(span=12).mean()
        ema26 = p.ewm(span=26).mean()
        df['macd'] = (ema12 - ema26) / p.replace(0, np.nan)
        df['macd_signal'] = df['macd'].ewm(span=9).mean()
        df['macd_hist'] = df['macd'] - df['macd_signal']
        
        # 5. Bollinger Bands
        sma20 = p.rolling(20).mean()
        std20 = p.rolling(20).std()
        df['bb_width'] = (4 * std20) / sma20.replace(0, np.nan)
        df['bb_pos'] = (p - (sma20 - 2*std20)) / (4*std20).replace(0, np.nan)
        
        # 6. Moving Averages
        for d in [10, 20, 50, 200]:
            ma = p.rolling(d).mean()
            df[f'dist_ma_{d}'] = (p - ma) / ma.replace(0, np.nan)
        
        # 7. Momentum
        df['roc_5'] = p.pct_change(5)
        df['roc_10'] = p.pct_change(10)
        df['mom_accel'] = df['roc_5'] - (df['roc_10'] / 2)
        
        # 8. Statistical
        df['skew_20'] = returns.rolling(20).skew()
        df['kurt_20'] = returns.rolling(20).kurt()

        # 8b. Advanced statistical regime features
        if self.hurst_feature_enabled:
            _lags = self._hurst_lags

            def _hurst(x: np.ndarray) -> float:
                """R/S Hurst exponent. Returns H-0.5: >0=trending, <0=mean-reverting."""
                if len(x) < 10 or np.std(x) == 0:
                    return 0.0
                try:
                    lx = np.log(x / x[0] + 1e-12)
                    rs_vals = []
                    for lag in range(2, min(len(x) // 2, _lags) + 1):
                        sub = np.array([lx[i:i + lag] for i in range(0, len(lx) - lag, lag)])
                        if len(sub) == 0:
                            continue
                        devs = sub - sub.mean(axis=1, keepdims=True)
                        rs = (devs.cumsum(axis=1).max(axis=1) - devs.cumsum(axis=1).min(axis=1)) / (
                            sub.std(axis=1) + 1e-12
                        )
                        rs_vals.append((np.log(lag), np.log(np.mean(rs) + 1e-12)))
                    if len(rs_vals) < 3:
                        return 0.0
                    lags_arr, rs_arr = zip(*rs_vals)
                    H = np.polyfit(lags_arr, rs_arr, 1)[0]
                    return float(np.clip(H - 0.5, -0.5, 0.5))
                except Exception:
                    return 0.0

            df['hurst_bias'] = returns.rolling(_lags * 2).apply(
                lambda x: _hurst(x), raw=True
            )

            # Lag-1 autocorrelation of returns (20-bar): >0=trend persists, <0=reversal
            def _autocorr1(x: np.ndarray) -> float:
                if len(x) < 4:
                    return 0.0
                try:
                    c = np.corrcoef(x[:-1], x[1:])[0, 1]
                    return float(c) if not np.isnan(c) else 0.0
                except Exception:
                    return 0.0

            df['autocorr_lag1'] = returns.rolling(20).apply(_autocorr1, raw=True)

            # OU half-life: Δp = -κ·p + ε → half_life = ln(2)/κ; feature = 1/half_life_days
            def _ou_speed(x: np.ndarray) -> float:
                if len(x) < 5:
                    return 0.0
                try:
                    y = np.diff(x)
                    x_lag = x[:-1]
                    if np.std(x_lag) < 1e-10:
                        return 0.0
                    kappa = -np.polyfit(x_lag, y, 1)[0]
                    kappa = max(kappa, 1e-6)
                    half_life = np.log(2) / kappa
                    return float(np.clip(1.0 / half_life, 0.0, 1.0))
                except Exception:
                    return 0.0

            df['ou_reversion_speed'] = p.rolling(20).apply(_ou_speed, raw=True)
        else:
            df['hurst_bias'] = 0.0
            df['autocorr_lag1'] = 0.0
            df['ou_reversion_speed'] = 0.0

        # 9. Z-score
        df['zscore_60'] = (p - p.rolling(60).mean()) / p.rolling(60).std().replace(0, np.nan)

        # 10. Trend strength
        df['trend_strength'] = returns.rolling(20).apply(
            lambda x: np.polyfit(np.arange(len(x)), x, 1)[0] * 100 if len(x) > 1 else 0,
            raw=False
        )

        # 11. Quality Factors
        df['range_position'] = (p - p.rolling(20).min()) / (p.rolling(20).max() - p.rolling(20).min()).replace(0, np.nan)
        # mean_rev_20 REMOVED: was = -zscore_60/3, which hardwires a SHORT signal whenever
        # price is above its 60-day mean — always negative in trending/bull markets.

        def calc_r2(x):
            if len(x) < 5: return 0
            y = np.arange(len(x))
            corr = np.corrcoef(x, y)[0, 1]
            return corr ** 2 if not np.isnan(corr) else 0

        df['trend_consistency'] = p.rolling(20).apply(calc_r2, raw=True)

        # 12. Volume Features (if available)
        if v is not None:
            df['volume_surge'] = v / v.rolling(20).mean().replace(0, np.nan)
            # OBV momentum
            obv = (np.sign(returns) * v).fillna(0).cumsum()
            df['obv_momentum'] = (obv - obv.rolling(20).mean()) / obv.rolling(20).std().replace(0, np.nan)
        else:
            df['volume_surge'] = 1.0
            df['obv_momentum'] = 0.0

        # 13. Intraday Features (if available)
        if h is not None and low_px is not None:
            df['intraday_volat'] = (h - low_px) / p.replace(0, np.nan)
        else:
            df['intraday_volat'] = 0.0

        # 14. Contextual Features (direct injection)
        df['sentiment_feature'] = sentiment_score if sentiment_score is not None else 0.0
        df['momentum_rank_feature'] = momentum_rank if momentum_rank is not None else 0.5

        # 15. Intraday 5-minute bar features (scalar broadcast onto every row)
        intra_feats = self._intraday_momentum_features(data, intraday_df)
        for feat_name, feat_val in intra_feats.items():
            df[feat_name] = feat_val

        # 16. Order Flow Imbalance proxies (volume-based, works with daily OHLCV)
        if v is not None and h is not None and low_px is not None:
            # Close Location Value (CLV): (C - L) / (H - L) → +1 = closes at high (buying), -1 = low (selling)
            _hl_range = (h - low_px).replace(0, np.nan)
            _clv = (p - low_px) / _hl_range  # [0, 1]
            # Money Flow Volume = CLV * Volume; positive = buyer-driven
            _mfv = (_clv * 2 - 1) * v  # scaled to [-1, 1] × volume
            df['order_flow_imbalance'] = _mfv.rolling(5).sum() / v.rolling(5).sum().replace(0, np.nan)
            # Volume-weighted directional pressure (up_bars vs down_bars over 5 bars)
            _up_vol = v.where(returns > 0, 0).rolling(5).sum()
            _dn_vol = v.where(returns < 0, 0).rolling(5).sum()
            _tot_vol = (_up_vol + _dn_vol).replace(0, np.nan)
            df['vol_imbalance_5d'] = (_up_vol - _dn_vol) / _tot_vol
        else:
            df['order_flow_imbalance'] = 0.0
            df['vol_imbalance_5d'] = 0.0

        return df.fillna(0).replace([np.inf, -np.inf], 0)

    def _align_features(self, df_feats: pd.DataFrame) -> pd.DataFrame:
        """Align current feature set to training feature names for compatibility."""
        if not self.feature_names:
            return df_feats
        # Ensure stable column order and fill missing with zeros
        return df_feats.reindex(columns=self.feature_names, fill_value=0.0)

    # Features that carry absolute directional meaning — cross-sectional z-scoring
    # removes the common market direction signal (e.g. all symbols overbought in bull).
    # These must NOT be normalized across symbols; only relative/rank features should be.
    _ABS_DIRECTION_FEATURES = frozenset({
        'zscore_60', 'bb_pos', 'range_position',
        'rsi_14', 'rsi_norm',
        'macd', 'macd_signal', 'macd_hist',
        'dist_ma_10', 'dist_ma_20', 'dist_ma_50', 'dist_ma_200',
        'roc_5', 'roc_10', 'mom_accel',
        'trend_strength',
        'hurst_bias', 'autocorr_lag1',
    })

    def _cross_sectional_normalize(
        self,
        panel: pd.DataFrame,
        feature_cols: List[str]
    ) -> pd.DataFrame:
        """Z-score features cross-sectionally per timestamp.

        Absolute-direction features (RSI, MACD, z-score, momentum, MA distance)
        are intentionally skipped — normalizing them across symbols destroys the
        common bull/bear market signal and causes 0% win rates in trending regimes.
        Only relative/dispersion features are normalized.
        """
        if not feature_cols:
            return panel
        cols_to_norm = [c for c in feature_cols if c not in self._ABS_DIRECTION_FEATURES]
        if not cols_to_norm:
            return panel
        grouped = panel.groupby(level=0)
        for col in cols_to_norm:
            mean = grouped[col].transform("mean")
            std = grouped[col].transform("std")
            # When std is 0 or NaN (≤1 symbol at timestamp), keep demeaned value
            std = std.replace(0, np.nan).fillna(1.0)
            panel[col] = (panel[col] - mean) / std
        return panel

    def _build_training_panel(
        self,
        historical_data: Dict[str, pd.DataFrame]
    ) -> Tuple[pd.DataFrame, List[str]]:
        """Build time-sorted MultiIndex panel for training."""
        frames = []
        feature_names: List[str] = []

        for symbol, data in historical_data.items():
            if len(data) < self.lookback + 20:
                continue

            try:
                parsed = parse_symbol(symbol)
            except ValueError:
                continue

            prices = data['Close']
            df_feats = self.compute_features_vectorized(data)

            # Targets: volatility-scaled future return
            future_ret = prices.pct_change(5).shift(-5)
            vol = prices.pct_change().rolling(ApexConfig.ADV_LABEL_VOL_LOOKBACK).std()
            vol = vol.replace(0, np.nan)
            scaled = (future_ret / vol).clip(
                -ApexConfig.ADV_LABEL_VOL_CLIP,
                ApexConfig.ADV_LABEL_VOL_CLIP
            )

            # Regimes — use same AdaptiveRegimeDetector as live inference for label consistency
            try:
                from models.adaptive_regime_detector import AdaptiveRegimeDetector as _ARDetector
                regimes_series = _ARDetector().classify_history(prices)
            except Exception:
                # Fallback: fast MA-based approximation
                _ma20 = prices.rolling(20).mean()
                _ma60 = prices.rolling(60).mean()
                _vol20 = prices.pct_change().rolling(20).std() * np.sqrt(252)
                _trend = (_ma20 - _ma60) / _ma60
                regimes_series = pd.Series('neutral', index=prices.index)
                regimes_series.loc[_vol20 > 0.35] = 'volatile'
                regimes_series.loc[(_vol20 <= 0.35) & (_trend > 0.05)] = 'bull'
                regimes_series.loc[(_vol20 <= 0.35) & (_trend < -0.05)] = 'bear'

            valid = ~df_feats.isnull().any(axis=1) & ~scaled.isnull()
            if valid.sum() == 0:
                continue

            df = df_feats.loc[valid].copy()
            df["target"] = scaled.loc[valid]
            df["regime"] = regimes_series.loc[valid]
            df["asset_class"] = parsed.asset_class.value
            df["symbol"] = symbol

            df = df.set_index(pd.MultiIndex.from_arrays(
                [df.index, df["symbol"]],
                names=["timestamp", "symbol"]
            ))
            df = df.drop(columns=["symbol"])

            if not feature_names:
                feature_names = list(df_feats.columns)

            frames.append(df)

        if not frames:
            return pd.DataFrame(), []

        panel = pd.concat(frames).sort_index(level=0)

        if getattr(ApexConfig, "ADV_CROSS_SECTIONAL_NORM", True):
            panel = self._cross_sectional_normalize(panel, feature_names)

        panel = panel.dropna(subset=feature_names + ["target", "regime", "asset_class"])
        return panel, feature_names

    def _quality_adjust(self, signal: float, features: pd.Series, regime: str) -> Tuple[float, float]:
        """Apply risk-aware adjustments to improve Sharpe (reduce noisy trades)."""
        vol_20 = float(features.get('vol_20d', 0.0))
        zscore = float(features.get('zscore_60', 0.0))
        dist_ma_50 = float(features.get('dist_ma_50', 0.0))
        bb_width = float(features.get('bb_width', 0.0))
        trend_strength = float(features.get('trend_strength', 0.0))

        # Volatility penalty: reduce size in high vol regimes
        vol_penalty = np.clip((vol_20 - 0.25) / 0.35, 0.0, 0.6)

        # Mean-reversion penalty if signal fights extreme zscore
        z_penalty = 0.0
        if signal > 0 and zscore > 2.0:
            z_penalty = 0.20
        elif signal < 0 and zscore < -2.0:
            z_penalty = 0.20

        # Trend alignment bonus: align with MA slope/trend strength
        trend_bonus = np.clip(abs(trend_strength) * 8.0, 0.0, 0.15)
        if signal > 0 and dist_ma_50 < -0.02:
            trend_bonus *= 0.5
        if signal < 0 and dist_ma_50 > 0.02:
            trend_bonus *= 0.5

        # Bandwidth penalty for noisy ranges
        band_penalty = np.clip((bb_width - 0.10) / 0.30, 0.0, 0.25)

        # Regime bias: reduce counter-trend signals
        regime_bias = 0.0
        if regime == 'bull' and signal < 0:
            regime_bias = 0.15
        elif regime == 'bear' and signal > 0:
            regime_bias = 0.15

        quality = 1.0 - vol_penalty - z_penalty - band_penalty - regime_bias + trend_bonus
        quality = float(np.clip(quality, 0.2, 1.1))

        adjusted_signal = float(np.clip(signal * quality, -1, 1))
        return adjusted_signal, quality
    
    def train_walk_forward(
        self,
        historical_data: Dict[str, pd.DataFrame],
        n_splits: int = 4
    ):
        """Walk-forward training with regime awareness."""
        if not ML_AVAILABLE:
            return
        
        logger.info("\n" + "="*80)
        logger.info("🧠 GOD LEVEL WALK-FORWARD TRAINING")
        logger.info("="*80)
        
        panel, feature_names = self._build_training_panel(historical_data)
        if panel.empty:
            logger.error("No valid training data")
            return

        self.feature_names = feature_names

        # Time-sorted panel to prevent leakage
        times = panel.index.get_level_values(0).unique().sort_values()
        if len(times) < 50:
            logger.error("Insufficient time points for walk-forward training")
            return

        min_train = int(len(times) * 0.5)
        window_size = max(5, int((len(times) - min_train) / max(1, n_splits)))

        regime_results = {r: [] for r in ['bull', 'bear', 'neutral', 'volatile']}
        regime_dir_acc = {r: [] for r in ['bull', 'bear', 'neutral', 'volatile']}

        purge_days = getattr(ApexConfig, "ADV_PURGE_DAYS", 5)
        embargo_days = getattr(ApexConfig, "ADV_EMBARGO_DAYS", 2)

        for split in range(n_splits):
            test_start_idx = min_train + (split * window_size)
            test_end_idx = min(test_start_idx + window_size, len(times))

            if test_end_idx - test_start_idx < 5:
                break

            test_start_time = times[test_start_idx]
            test_end_time = times[test_end_idx - 1]

            purge_start = test_start_time - timedelta(days=purge_days)
            embargo_end = test_end_time + timedelta(days=embargo_days)

            train_mask = (times < purge_start) | (times > embargo_end)
            train_times = times[train_mask]
            test_times = times[test_start_idx:test_end_idx]

            logger.info(
                f"\n[Split {split+1}/{n_splits}] Train: {train_times.min()}-{train_times.max()} "
                f"Test: {test_start_time}-{test_end_time} (purge={purge_days}d embargo={embargo_days}d)"
            )

            train_panel = panel.loc[train_times]
            test_panel = panel.loc[test_times]

            for asset_class in [AssetClass.EQUITY.value, AssetClass.FOREX.value, AssetClass.CRYPTO.value]:
                train_ac = train_panel[train_panel["asset_class"] == asset_class]
                test_ac = test_panel[test_panel["asset_class"] == asset_class]

                if train_ac.empty or test_ac.empty:
                    continue

                X_train_all = train_ac[feature_names].values
                y_train_all = train_ac["target"].values
                regimes_train = train_ac["regime"].values

                X_test_all = test_ac[feature_names].values
                y_test_all = test_ac["target"].values
                regimes_test = test_ac["regime"].values

                for regime in ['bull', 'bear', 'neutral', 'volatile']:
                    mask_train = (regimes_train == regime)
                    mask_test = (regimes_test == regime)

                    if mask_train.sum() < 100 or mask_test.sum() < 20:
                        continue

                    X_r_train = X_train_all[mask_train]
                    y_r_train = y_train_all[mask_train]
                    X_r_test = X_test_all[mask_test]
                    y_r_test = y_test_all[mask_test]

                    if regime not in self.asset_class_imputers[asset_class]:
                        self.asset_class_imputers[asset_class][regime] = SimpleImputer(strategy='median')
                        self.asset_class_scalers[asset_class][regime] = RobustScaler()

                    imputer = self.asset_class_imputers[asset_class][regime]
                    scaler = self.asset_class_scalers[asset_class][regime]

                    X_r_train_imp = imputer.fit_transform(X_r_train)
                    X_r_train_sc = scaler.fit_transform(X_r_train_imp)

                    X_r_test_imp = imputer.transform(X_r_test)
                    X_r_test_sc = scaler.transform(X_r_test_imp)

                    models = self._train_regime_models(
                        X_r_train_sc, y_r_train, X_r_test_sc, y_r_test, regime
                    )
                    self.asset_class_models[asset_class][regime] = models

                    if 'r2_score' in models:
                        regime_results[regime].append(models['r2_score'])
                    if 'dir_acc' in models:
                        regime_dir_acc[regime].append(models['dir_acc'])

                    # --- Binary classifier training (same preprocessed data) ---
                    if self._binary_clf is not None:
                        try:
                            # Binary labels derived from regression target sign
                            # (positive vol-scaled return → price went up)
                            y_bin_train = (y_r_train > 0).astype(int)
                            y_bin_test = (y_r_test > 0).astype(int)
                            self._binary_clf.train(
                                X_r_train_sc, y_bin_train,
                                X_r_test_sc, y_bin_test,
                                regime, asset_class,
                            )
                        except Exception as _be:
                            logger.debug("Binary training error (%s/%s): %s", asset_class, regime, _be)

        # Summary
        logger.info("\n" + "="*80)
        logger.info("📊 WALK-FORWARD REGIME RESULTS")
        logger.info("="*80)
        all_dir_accs = []
        for regime, scores in regime_results.items():
            if scores:
                dir_scores = regime_dir_acc.get(regime, [])
                dir_str = f" DirAcc={np.mean(dir_scores):.1%}" if dir_scores else ""
                logger.info(f"{regime.upper():>10}: R²={np.mean(scores):.4f} ± {np.std(scores):.4f}{dir_str}")
                all_dir_accs.extend(dir_scores)

        # ── Model Quality Gate ───────────────────────────────────────────────
        # Block deployment of models that are worse than random on direction.
        # A model with dir_acc < threshold has no edge and will lose money.
        _min_dir_acc = float(
            getattr(ApexConfig, "MODEL_MIN_DIR_ACC", 0.48)
        )
        _gate_enabled = getattr(ApexConfig, "MODEL_QUALITY_GATE_ENABLED", True)
        if _gate_enabled and all_dir_accs:
            _overall_dir_acc = float(np.mean(all_dir_accs))
            if _overall_dir_acc < _min_dir_acc:
                logger.warning(
                    "⚠️  MODEL QUALITY GATE: overall DirAcc=%.1f%% < %.1f%% threshold — "
                    "models NOT deployed (is_trained stays False). "
                    "Signal generator will use previous saved models or return empty signals.",
                    _overall_dir_acc * 100, _min_dir_acc * 100,
                )
                # Do NOT set is_trained = True; save models for debugging artifacts
                self._save_models()
                return
            else:
                logger.info(
                    "✅ Model Quality Gate PASSED: DirAcc=%.1f%% >= %.1f%% threshold",
                    _overall_dir_acc * 100, _min_dir_acc * 100,
                )

        # ── Feature Importance Summary ───────────────────────────────────────
        # Aggregate importances across all regimes and log top-10 features.
        # Low-importance features are flagged for potential future pruning.
        try:
            _all_fi: dict[str, list] = {}
            for _ac_models in self.asset_class_models.values():
                for _regime_models in _ac_models.values():
                    for _feat, _imp in _regime_models.get('feature_importances', {}).items():
                        _all_fi.setdefault(_feat, []).append(_imp)
            if _all_fi:
                _mean_fi = {f: float(np.mean(v)) for f, v in _all_fi.items()}
                _top10 = sorted(_mean_fi.items(), key=lambda x: x[1], reverse=True)[:10]
                _bot5 = sorted(_mean_fi.items(), key=lambda x: x[1])[:5]
                self.feature_importances_ = _mean_fi  # expose for external inspection
                logger.info("📊 Top-10 features: %s",
                            ', '.join(f"{n}={v:.3f}" for n, v in _top10))
                logger.info("📊 Bottom-5 features (possible noise): %s",
                            ', '.join(f"{n}={v:.4f}" for n, v in _bot5))
        except Exception as _fi_err:
            logger.debug("Feature importance summary skipped: %s", _fi_err)

        # Aggregate prediction-distribution baseline across all deployed
        # regime models. Mean-of-means and pooled std give a single reference
        # against which the live output drift can be measured.
        try:
            _baseline_means: List[float] = []
            _baseline_stds: List[float] = []
            for _ac_models in self.asset_class_models.values():
                for _regime_models in _ac_models.values():
                    if 'baseline_pred_mean' in _regime_models:
                        _baseline_means.append(float(_regime_models['baseline_pred_mean']))
                    if 'baseline_pred_std' in _regime_models:
                        _baseline_stds.append(float(_regime_models['baseline_pred_std']))
            if _baseline_means:
                self.pred_baseline_mean = float(np.mean(_baseline_means))
            if _baseline_stds:
                self.pred_baseline_std = float(
                    np.sqrt(np.mean(np.square(_baseline_stds)))
                )
            logger.info(
                "ML prediction baseline: mean=%.4f std=%.4f",
                self.pred_baseline_mean, self.pred_baseline_std,
            )
        except Exception as _base_err:
            logger.debug("Baseline aggregation skipped: %s", _base_err)

        # Finalize
        self.is_trained = True
        self.training_date = datetime.now()
        self.last_retrain_date = datetime.now()

        # Save binary classifier alongside regression models
        if self._binary_clf is not None and self._binary_clf.is_trained:
            try:
                self._binary_clf.save_models()
            except Exception as _be:
                logger.debug("Binary classifier save error: %s", _be)

        self._save_models()
    
    def _train_regime_models(self, X_train, y_train, X_test, y_test, _regime) -> Dict:
        """Train regression models for a regime."""
        models = {}
        predictions = []
        
        # Random Forest
        try:
            rf = RandomForestRegressor(n_estimators=100, max_depth=8, n_jobs=-1, random_state=42)
            rf.fit(X_train, y_train)
            models['rf'] = rf
            predictions.append(rf.predict(X_test))
        except:
            pass
        
        # Gradient Boosting
        try:
            gb = GradientBoostingRegressor(n_estimators=100, learning_rate=0.05, max_depth=5, random_state=42)
            gb.fit(X_train, y_train)
            models['gb'] = gb
            predictions.append(gb.predict(X_test))
        except:
            pass
        
        # XGBoost
        if XGBOOST_AVAILABLE:
            try:
                xgb_model = xgb.XGBRegressor(n_estimators=150, max_depth=6, learning_rate=0.03, random_state=42, verbosity=0)
                xgb_model.fit(X_train, y_train)
                models['xgb'] = xgb_model
                predictions.append(xgb_model.predict(X_test))
            except:
                pass
        
        # LightGBM
        if LIGHTGBM_AVAILABLE:
            try:
                lgb_model = lgb.LGBMRegressor(n_estimators=150, max_depth=6, learning_rate=0.03, random_state=42, verbose=-1)
                lgb_model.fit(X_train, y_train)
                models['lgb'] = lgb_model
                predictions.append(lgb_model.predict(X_test))
            except:
                pass
        
        # Per-model directional accuracy on test set (used for dynamic weighting at inference)
        _model_preds: Dict[str, np.ndarray] = {}
        # Re-predict each model individually to get per-model accuracy
        for _mn in ('rf', 'gb', 'xgb', 'lgb'):
            _m = models.get(_mn)
            if _m is not None:
                try:
                    _mp = _m.predict(X_test)
                    _dacc = float(np.mean(np.sign(_mp) == np.sign(y_test)))
                    models[f'{_mn}_dir_acc'] = _dacc
                    _model_preds[_mn] = _mp
                except Exception:
                    models[f'{_mn}_dir_acc'] = 0.5

        # Calculate ensemble R² and directional accuracy
        if predictions:
            ensemble_pred = np.mean(predictions, axis=0)
            ss_res = np.sum((y_test - ensemble_pred) ** 2)
            ss_tot = np.sum((y_test - np.mean(y_test)) ** 2)
            r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
            models['r2_score'] = r2
            models['dir_acc'] = float(np.mean(np.sign(ensemble_pred) == np.sign(y_test)))

            # Capture the baseline distribution of the *inference-space*
            # output — matches the squashed signal used at runtime so the
            # drift comparison uses like-for-like units.
            squashed = np.tanh(ensemble_pred * 25.0)
            models['baseline_pred_mean'] = float(np.mean(squashed))
            models['baseline_pred_std'] = float(np.std(squashed))

        # Extract feature importances (from tree-based models) for post-training audit.
        # Aggregated across all available models; stored for logging and dropout decisions.
        try:
            _fi_arrays = []
            _feature_names = list(X_train.columns)
            for _m_key in ('rf', 'gb', 'xgb', 'lgb'):
                _m = models.get(_m_key)
                if _m is not None and hasattr(_m, 'feature_importances_'):
                    _fi = _m.feature_importances_
                    if len(_fi) == len(_feature_names):
                        _fi_arrays.append(_fi)
            if _fi_arrays:
                _mean_fi = np.mean(_fi_arrays, axis=0)
                models['feature_importances'] = {
                    _feature_names[i]: float(_mean_fi[i])
                    for i in range(len(_feature_names))
                }
        except Exception:
            pass

        return models
    
    def generate_ml_signal(
        self, 
        symbol: str, 
        data: Union[pd.Series, pd.DataFrame], 
        sentiment_score: float = 0.0,
        momentum_rank: float = 0.5,
        track: bool = True
    ) -> Dict:
        if not self.is_trained:
            if not self._load_models():
                return self._empty_signal()

        # Check retrain
        if self._should_retrain():
            logger.warning("⚠️ Retrain recommended")

        try:
            try:
                parsed = parse_symbol(symbol)
                asset_class = parsed.asset_class.value
            except ValueError:
                asset_class = AssetClass.EQUITY.value

            # Feature-staleness guard: if the latest feature bar is older than
            # ``ML_FEATURE_MAX_AGE_SECONDS``, the model is predicting from
            # outdated data. Returning a zero-confidence empty signal prevents
            # acting on ghost information after a data-feed hiccup.
            max_age = float(getattr(ApexConfig, "ML_FEATURE_MAX_AGE_SECONDS", 0.0))
            if max_age > 0.0 and len(data) > 0:
                last_ts = self._resolve_feature_timestamp(data)
                if last_ts is not None:
                    now_ts = datetime.now(last_ts.tzinfo) if last_ts.tzinfo else datetime.now()
                    age_seconds = (now_ts - last_ts).total_seconds()
                    if age_seconds > max_age:
                        logger.warning(
                            "ML feature staleness blocked %s: feature_bar_ts=%s "
                            "now_ts=%s age=%.1fs max=%.1fs",
                            symbol, last_ts.isoformat(),
                            now_ts.isoformat(), age_seconds, max_age,
                        )
                        return self._empty_signal()

            # Extract prices for regime detection
            if isinstance(data, pd.DataFrame):
                prices = data['Close']
            else:
                prices = data
                
            # Robustly extract a clean 1D Series from any input
            if isinstance(prices, pd.DataFrame):
                # Handle both regular and MultiIndex columns
                if 'Close' in prices.columns:
                    prices = prices['Close']
                elif len(prices.columns) > 0:
                    prices = prices.iloc[:, 0]

            # Handle any remaining DataFrame or Series issues
            if isinstance(prices, pd.DataFrame):
                prices = prices.iloc[:, 0]

            # Force to clean 1D numpy-backed Series with simple integer index
            values = np.asarray(prices).flatten()
            prices_clean = pd.Series(values, index=range(len(values)))

            # Detect regime
            regime = get_regime(prices_clean, 60)
            
            # Get models
            models_by_class = self.asset_class_models.get(asset_class) or self.regime_models
            if regime not in models_by_class or not models_by_class[regime]:
                regime = 'neutral'

            models = models_by_class.get(regime, {})
            if not models:
                return self._empty_signal()
            
            # Features (passing full window + context)
            window = data.iloc[-self.lookback:] if len(data) > self.lookback else data
            df_feats = self.compute_features_vectorized(
                window, 
                sentiment_score=sentiment_score,
                momentum_rank=momentum_rank
            )
            df_feats = self._align_features(df_feats)
            current = df_feats.iloc[-1:].values
            
            # Preprocess
            imputer = self.asset_class_imputers.get(asset_class, {}).get(regime) or self.regime_imputers.get(regime)
            scaler = self.asset_class_scalers.get(asset_class, {}).get(regime) or self.regime_scalers.get(regime)
            if not imputer or not scaler:
                return self._empty_signal()
            
            current = imputer.transform(current)
            current = scaler.transform(current)

            # Cache scaled feature vector for potential online update at trade close
            try:
                _ol_key = f"{asset_class}:{regime}"
                self._last_features[symbol] = (current.copy(), _ol_key)
            except Exception:
                pass

            # Predict — accuracy-weighted ensemble
            # Each model's contribution is proportional to its training-set
            # directional accuracy raised to MODEL_WEIGHT_TEMPERATURE power.
            # Falls back to equal weight when accuracy data is unavailable.
            predictions = []
            pred_accs   = []
            for name in ['rf', 'gb', 'xgb', 'lgb']:
                model = models.get(name)
                if model:
                    try:
                        pred_return = model.predict(current)[0]
                        # scale=25: maps actual model output range [0.002, 0.011] → signal [0.05, 0.27]
                        # This is the correct scale given the model's regularized prediction magnitude.
                        predictions.append(float(np.tanh(pred_return * 25)))
                        # Per-model accuracy stored during training (default 0.5 = equal weight)
                        _acc = float(models.get(f'{name}_dir_acc', 0.5))
                        pred_accs.append(max(0.40, _acc))   # floor prevents zero-weighting
                    except Exception:
                        pass

            if not predictions:
                return self._empty_signal()

            # Temperature-scaled softmax weighting (config-driven)
            _weight_enabled = bool(getattr(ApexConfig, 'MODEL_WEIGHT_ACCURACY_BONUS', True))
            if _weight_enabled and len(predictions) > 1:
                _temp = float(getattr(ApexConfig, 'MODEL_WEIGHT_TEMPERATURE', 3.0))
                _w = np.array(pred_accs) ** _temp
                _w = _w / _w.sum()
                avg_signal = float(np.average(predictions, weights=_w))
            else:
                avg_signal = float(np.mean(predictions))

            # Record raw (pre-EMA, pre-clip) ensemble output for drift tracking.
            if self._drift_window > 0:
                self._drift_buffer.append(avg_signal)

            std_signal = float(np.std(predictions))
            confidence = max(0, 1.0 - (std_signal * 2))

            # Risk-aware adjustment for Sharpe improvement
            features_last = df_feats.iloc[-1]
            adjusted_signal, quality = self._quality_adjust(avg_signal, features_last, regime)
            confidence = float(np.clip(confidence * quality, 0.0, 1.0))

            # Smooth signal to reduce noise and improve stability
            prev = self.signal_ema.get(symbol, adjusted_signal)
            smoothed = self.signal_ema_alpha * adjusted_signal + (1 - self.signal_ema_alpha) * prev
            self.signal_ema[symbol] = smoothed
            
            regression_signal = float(np.clip(smoothed, -1, 1))

            # --- Binary classifier signal ---
            binary_signal = 0.0
            binary_confidence = 0.0
            if self._binary_clf is not None and self._binary_clf.is_trained:
                try:
                    binary_signal, binary_confidence = self._binary_clf.predict(
                        current, regime, asset_class
                    )
                except Exception:
                    pass

            # --- Blend regression + binary ---
            if self._binary_enabled and binary_signal != 0.0:
                w_bin = self._binary_weight
                blended = (1.0 - w_bin) * regression_signal + w_bin * binary_signal
                blended_conf = (1.0 - w_bin) * confidence + w_bin * binary_confidence
            else:
                blended = regression_signal
                blended_conf = confidence

            result = {
                'signal': float(np.clip(blended, -1, 1)),
                'confidence': float(np.clip(blended_conf, 0, 1)),
                'quality': float(np.clip(quality, 0, 1)),
                'regime': regime,
                'regression_signal': regression_signal,
                'binary_signal': binary_signal,
                'binary_confidence': binary_confidence,
                'timestamp': datetime.now().isoformat()
            }

            if track:
                self.prediction_history.append(result)

            return result
        
        except Exception as e:
            logger.error(f"Inference error: {e}")
            return self._empty_signal()
    
    def record_outcome(self, actual_return: float):
        """Record outcome for drift monitoring."""
        self.outcome_history.append(actual_return)
    
    def check_drift(self) -> Dict:
        """
        Detect ML drift across two axes:

        1. *Performance drift* — directional accuracy and trade Sharpe measured
           from ``prediction_history`` vs ``outcome_history``.
        2. *Output-distribution drift* — rolling mean/std of the raw ensemble
           prediction compared to the baseline captured at training time. A
           blown-out mean indicates prediction bias; a blown-out std indicates
           the model is reacting to features it never saw during training.

        Returns:
            Dict with keys ``drift_detected`` (bool), ``accuracy``, ``sharpe``,
            ``rolling_mean`` / ``rolling_std`` (when output drift is
            computable), and ``recommendation`` ∈ ``{"OK", "RETRAIN"}``.
        """
        result: Dict[str, object] = {
            'drift_detected': False,
            'accuracy': 0.0,
            'sharpe': 0.0,
            'rolling_mean': 0.0,
            'rolling_std': 0.0,
            'recommendation': 'OK',
            'reason': None,
        }

        perf_drift = False
        if len(self.prediction_history) >= 30 and len(self.outcome_history) >= 30:
            min_len = min(len(self.prediction_history), len(self.outcome_history))
            correct = sum(
                (self.prediction_history[i]['signal'] > 0) == (self.outcome_history[i] > 0)
                for i in range(-min_len, 0)
            )
            accuracy = correct / min_len
            returns = [
                self.prediction_history[i]['signal'] * self.outcome_history[i]
                for i in range(-min_len, 0)
            ]
            std_ret = float(np.std(returns))
            sharpe = float(np.mean(returns) / std_ret * np.sqrt(252)) if std_ret > 0 else 0.0
            perf_drift = (accuracy < self.performance_baseline) or (sharpe < 1.0)
            result.update({'accuracy': float(accuracy), 'sharpe': sharpe})
        else:
            result['reason'] = 'insufficient_performance_data'

        # Output-distribution drift
        dist_drift = False
        min_drift_samples = max(30, self._drift_window // 2) if self._drift_window else 30
        if self._drift_window > 0 and len(self._drift_buffer) >= min_drift_samples:
            samples = np.asarray(self._drift_buffer, dtype=float)
            roll_mean = float(np.mean(samples))
            roll_std = float(np.std(samples))
            result['rolling_mean'] = roll_mean
            result['rolling_std'] = roll_std

            mean_threshold = float(
                getattr(ApexConfig, "ML_DRIFT_MEAN_THRESHOLD", 0.35)
            )
            std_mult = float(getattr(ApexConfig, "ML_DRIFT_STD_MULT", 1.75))
            baseline_mean = self.pred_baseline_mean
            baseline_std = self.pred_baseline_std

            mean_breach = abs(roll_mean - baseline_mean) > mean_threshold
            std_breach = (
                baseline_std > 0.0
                and roll_std > baseline_std * std_mult
            )
            dist_drift = mean_breach or std_breach
            if dist_drift:
                logger.warning(
                    "ML output drift: roll_mean=%.4f (baseline=%.4f, |Δ|>%.2f=%s) "
                    "roll_std=%.4f (baseline=%.4f × %.2f=%s)",
                    roll_mean, baseline_mean, mean_threshold, mean_breach,
                    roll_std, baseline_std, std_mult, std_breach,
                )

        drift = perf_drift or dist_drift
        result['drift_detected'] = drift
        result['recommendation'] = 'RETRAIN' if drift else 'OK'
        if drift and result['reason'] is None:
            result['reason'] = (
                'output_distribution_shift' if dist_drift and not perf_drift
                else 'performance_degraded'
            )
        return result
    
    def _should_retrain(self) -> bool:
        """Check if retrain needed."""
        if self.last_retrain_date:
            days = (datetime.now() - self.last_retrain_date).days
            if days >= self.retrain_interval_days:
                return True
        
        if self.check_drift()['drift_detected']:
            return True
        
        return False
    
    def _empty_signal(self):
        return {'signal': 0.0, 'confidence': 0.0, 'regime': 'unknown', 'timestamp': datetime.now().isoformat()}

    @staticmethod
    def _resolve_feature_timestamp(
        data: Union[pd.Series, pd.DataFrame]
    ) -> Optional[datetime]:
        """
        Extract the timestamp of the most-recent feature bar.

        Supports DatetimeIndex and integer-index DataFrames/Series that carry a
        ``timestamp`` or ``Datetime`` column.

        Args:
            data: Feature bar set (rows = time).

        Returns:
            The last bar timestamp as a naive or tz-aware ``datetime``, or
            ``None`` when no interpretable timestamp is present.
        """
        try:
            idx = data.index
            if isinstance(idx, pd.DatetimeIndex) and len(idx) > 0:
                ts = idx[-1]
                if isinstance(ts, pd.Timestamp):
                    return ts.to_pydatetime()
                if isinstance(ts, datetime):
                    return ts
        except Exception:
            pass
        if isinstance(data, pd.DataFrame):
            for col in ("timestamp", "Datetime", "datetime", "Date", "date"):
                if col in data.columns and len(data[col]) > 0:
                    try:
                        val = pd.to_datetime(data[col].iloc[-1])
                        if isinstance(val, pd.Timestamp):
                            return val.to_pydatetime()
                    except Exception:
                        continue
        return None

    # ─────────────────────────────────────────────────────────────────────────
    # Incremental Online Learning
    # Called after a trade closes: pairs the cached feature vector with the
    # realized return and triggers a warm-start micro-update on the GBM model
    # for that (asset_class, regime) bucket.
    # ─────────────────────────────────────────────────────────────────────────
    def online_update(self, symbol: str, actual_return: float) -> bool:
        """
        Feed a realized return back into the model after a trade closes.

        Args:
            symbol:         Trading symbol (e.g. "BTC/USD")
            actual_return:  Realized P&L% as a fraction (e.g. 0.012 = +1.2%)

        Returns:
            True if a warm-start model update was performed, False otherwise.
        """
        if not getattr(ApexConfig, "ONLINE_LEARNING_ENABLED", True):
            return False

        cached = self._last_features.pop(symbol, None)
        if cached is None:
            return False

        try:
            X_row, ol_key = cached          # (1, n_features) array, "asset_class:regime"
            parts = ol_key.split(":", 1)
            if len(parts) != 2:
                return False
            asset_class, regime = parts[0], parts[1]

            # Append to rolling buffer
            buf = self._online_buffer.setdefault(ol_key, [])
            buf.append((X_row, float(actual_return)))

            # Trim to max size (drop oldest)
            if len(buf) > self._online_buffer_maxsize:
                del buf[: len(buf) - self._online_buffer_maxsize]

            # Only retrain once we have enough new samples
            if len(buf) < self._online_update_trigger:
                return False

            # Retrieve GB model for this bucket (most amenable to warm_start)
            ac_models = self.asset_class_models.get(asset_class, {})
            regime_models = ac_models.get(regime, {})
            gb_model = regime_models.get("gb")
            if gb_model is None or not hasattr(gb_model, "n_estimators_"):
                return False

            # Build mini training set from buffer
            X_mini = np.vstack([row for row, _ in buf])   # (N, features)
            y_mini = np.array([lbl for _, lbl in buf])     # (N,)

            # Warm-start: add a small number of new trees on top of existing ones
            _n_extra = max(5, int(gb_model.n_estimators_ * 0.05))
            gb_model.n_estimators = gb_model.n_estimators_ + _n_extra
            gb_model.warm_start = True
            gb_model.fit(X_mini, y_mini)

            # Reset buffer after update (keep last quarter for continuity)
            keep = max(self._online_update_trigger // 4, 2)
            self._online_buffer[ol_key] = buf[-keep:]

            logger.info(
                "🔄 OnlineLearning: %s [%s] warm_start +%d trees (%d samples) "
                "actual_return=%.4f",
                symbol, ol_key, _n_extra, len(X_mini), actual_return,
            )
            return True

        except Exception as exc:
            logger.debug("online_update failed for %s: %s", symbol, exc)
            return False

    def _save_models(self):
        """Save all models."""
        try:
            has_asset_models = any(
                any(self.asset_class_models[ac][r] for r in self.asset_class_models[ac])
                for ac in self.asset_class_models
            )

            if has_asset_models:
                for asset_class, regimes in self.asset_class_models.items():
                    for regime, models in regimes.items():
                        regime_dir = f"{self.model_dir}/{asset_class}/{regime}"
                        os.makedirs(regime_dir, exist_ok=True)

                        for name, model in models.items():
                            if name == 'r2_score':
                                continue
                            if name == 'xgb':
                                model.save_model(f"{regime_dir}/xgb.json")
                            else:
                                dump(model, f"{regime_dir}/{name}.pkl")

                        imputer = self.asset_class_imputers.get(asset_class, {}).get(regime)
                        scaler = self.asset_class_scalers.get(asset_class, {}).get(regime)
                        if imputer:
                            dump(imputer, f"{regime_dir}/imputer.pkl")
                        if scaler:
                            dump(scaler, f"{regime_dir}/scaler.pkl")
            else:
                for regime, models in self.regime_models.items():
                    regime_dir = f"{self.model_dir}/{regime}"
                    os.makedirs(regime_dir, exist_ok=True)

                    for name, model in models.items():
                        if name == 'r2_score':
                            continue
                        if name == 'xgb':
                            model.save_model(f"{regime_dir}/xgb.json")
                        else:
                            dump(model, f"{regime_dir}/{name}.pkl")

                    if regime in self.regime_imputers:
                        dump(self.regime_imputers[regime], f"{regime_dir}/imputer.pkl")
                        dump(self.regime_scalers[regime], f"{regime_dir}/scaler.pkl")
            
            metadata = {
                'feature_names': self.feature_names,
                'training_date': self.training_date.isoformat() if self.training_date else None,
                'last_retrain_date': self.last_retrain_date.isoformat() if self.last_retrain_date else None,
                'prediction_history': list(self.prediction_history),
                'outcome_history': list(self.outcome_history),
                'model_layout': 'asset_class_regime' if has_asset_models else 'legacy',
                'pred_baseline_mean': float(self.pred_baseline_mean),
                'pred_baseline_std': float(self.pred_baseline_std),
            }
            
            with open(f"{self.model_dir}/metadata.pkl", 'wb') as f:
                pickle.dump(metadata, f)
            
            logger.info("✅ Advanced models saved")
        except Exception as e:
            logger.error(f"Save error: {e}")
    
    def _load_models(self) -> bool:
        """Load models."""
        try:
            if not os.path.exists(f"{self.model_dir}/metadata.pkl"):
                return False
            
            with open(f"{self.model_dir}/metadata.pkl", 'rb') as f:
                metadata = pickle.load(f)
            
            self.feature_names = metadata['feature_names']
            if metadata.get('last_retrain_date'):
                self.last_retrain_date = datetime.fromisoformat(metadata['last_retrain_date'])

            self.prediction_history = deque(metadata.get('prediction_history', []), maxlen=100)
            self.outcome_history = deque(metadata.get('outcome_history', []), maxlen=100)

            # Restore prediction-distribution baseline if present in metadata;
            # otherwise keep the config defaults seeded in ``__init__``.
            if 'pred_baseline_mean' in metadata:
                self.pred_baseline_mean = float(metadata['pred_baseline_mean'])
            if 'pred_baseline_std' in metadata:
                self.pred_baseline_std = float(metadata['pred_baseline_std'])
            
            layout = metadata.get('model_layout', 'legacy')
            if layout == 'asset_class_regime':
                for asset_class in [AssetClass.EQUITY.value, AssetClass.FOREX.value, AssetClass.CRYPTO.value]:
                    for regime in ['bull', 'bear', 'neutral', 'volatile']:
                        regime_dir = f"{self.model_dir}/{asset_class}/{regime}"
                        if not os.path.exists(regime_dir):
                            continue

                        self.asset_class_models[asset_class][regime] = {}

                        if os.path.exists(f"{regime_dir}/imputer.pkl"):
                            self.asset_class_imputers[asset_class][regime] = load(f"{regime_dir}/imputer.pkl")
                        if os.path.exists(f"{regime_dir}/scaler.pkl"):
                            self.asset_class_scalers[asset_class][regime] = load(f"{regime_dir}/scaler.pkl")

                        for name in ['rf', 'gb', 'lgb']:
                            path = f"{regime_dir}/{name}.pkl"
                            if os.path.exists(path):
                                self.asset_class_models[asset_class][regime][name] = load(path)

                        if os.path.exists(f"{regime_dir}/xgb.json"):
                            model = xgb.XGBRegressor()
                            model.load_model(f"{regime_dir}/xgb.json")
                            self.asset_class_models[asset_class][regime]['xgb'] = model
            else:
                for regime in ['bull', 'bear', 'neutral', 'volatile']:
                    regime_dir = f"{self.model_dir}/{regime}"
                    if not os.path.exists(regime_dir):
                        continue
                    
                    self.regime_models[regime] = {}
                    
                    if os.path.exists(f"{regime_dir}/imputer.pkl"):
                        self.regime_imputers[regime] = load(f"{regime_dir}/imputer.pkl")
                    if os.path.exists(f"{regime_dir}/scaler.pkl"):
                        self.regime_scalers[regime] = load(f"{regime_dir}/scaler.pkl")
                    
                    for name in ['rf', 'gb', 'lgb']:
                        path = f"{regime_dir}/{name}.pkl"
                        if os.path.exists(path):
                            self.regime_models[regime][name] = load(path)
                    
                    if os.path.exists(f"{regime_dir}/xgb.json"):
                        model = xgb.XGBRegressor()
                        model.load_model(f"{regime_dir}/xgb.json")
                        self.regime_models[regime]['xgb'] = model
            
            self.is_trained = True
            logger.info("✅ Advanced models loaded")
            return True
        except Exception as e:
            logger.error(f"Load error: {e}")
            return False

    def train_models(self, historical_data: Dict[str, pd.DataFrame]):
        """Alias for train_walk_forward for API compatibility."""
        self.train_walk_forward(historical_data)
