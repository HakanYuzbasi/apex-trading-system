"""
quant_system/features/feature_engineer.py
================================================================================
Feature Engineering & Temporal Alignment (MANDATORY REQUIREMENT 2)
================================================================================
Generates 100% aligned, lookahead-free feature matrices including L2 imbalances,
volatility boundaries, and cross-asset correlations.
"""

import numpy as np
import pandas as pd
from typing import Dict, Tuple

class FeatureEngineer:
    def __init__(self, target_lag_bars: int = 15):
        self.target_lag = target_lag_bars

    def engineer_primary_asset(self, raw_df: pd.DataFrame) -> pd.DataFrame:
        """Adds foundational tabular and temporal metrics for the specific asset."""
        df = raw_df.copy()
        
        # 1. Standard OHLCV transformations
        df['Returns'] = df['Close'].pct_change()
        df['Log_Returns'] = np.log(df['Close'] / df['Close'].shift(1))
        
        # 2. Volatility
        df['Vol_20'] = df['Returns'].rolling(20).std()
        df['Vol_60'] = df['Returns'].rolling(60).std()
        
        # 3. Momentum
        df['Mom_10'] = df['Close'] / df['Close'].shift(10) - 1.0
        df['NATR'] = (df['High'] - df['Low']) / df['Close']
        
        # 4. Mocked L2 Order Book Imbalance (if volume exists)
        # B/A Pressure = (Close - Low) / (High - Low)
        buying_pressure = (df['Close'] - df['Low']) / (df['High'] - df['Low'] + 1e-8)
        df['l2_imbalance'] = (buying_pressure - 0.5) * df['Volume']
        df['l2_imbalance_surge'] = df['l2_imbalance'] / (df['l2_imbalance'].rolling(20).mean() + 1e-8)
        
        # 5. Target Generation (Labeling)
        # Target: Return exactly sequence_len bars into the future (Classification)
        df['Target_Return'] = df['Close'].shift(-self.target_lag) / df['Close'] - 1.0
        df['Target_Label'] = (df['Target_Return'] > 0).astype(int)
        
        return df

    def integrate_cross_assets(self, primary_df: pd.DataFrame, benchmark_df: pd.DataFrame, name: str) -> pd.DataFrame:
        """Injects running correlations between a primary asset and a benchmark (e.g., BTC or SPY)."""
        df = primary_df.copy()
        # Strictly align timestamps
        bench_aligned = benchmark_df['Close'].reindex(df.index).ffill()
        bench_ret = bench_aligned.pct_change()
        
        # Rolling correlation (Strictly backward-looking, no leakage)
        df[f'cross_asset_corr_{name}'] = df['Returns'].rolling(30).corr(bench_ret)
        return df

    def produce_aligned_matrices(self, df: pd.DataFrame, seq_length: int = 30) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Produce mathematically safe X (spatial), X_seq (temporal), and Y.
        Ensures STATE_t strictly matches features_{t-1}.
        """
        # Drop NaNs caused by rolling windows and targets
        clean_df = df.dropna()
        
        features = [
            'Returns', 'Log_Returns', 'Vol_20', 'Vol_60', 'Mom_10', 'NATR',
            'l2_imbalance', 'l2_imbalance_surge', 
            'cross_asset_corr_BTC', 'cross_asset_corr_SPY'
        ]
        
        # Keep only available features
        valid_features = [f for f in features if f in clean_df.columns]
        X_raw = clean_df[valid_features].values
        Y_raw = clean_df['Target_Label'].values
        prices = clean_df['Close'].values
        
        X_spatial = []
        X_seq = []
        Y_aligned = []
        price_aligned = []
        
        for i in range(seq_length, len(clean_df)):
            X_seq.append(X_raw[i - seq_length : i])     # Sequence of [t-seq..t-1]
            X_spatial.append(X_raw[i-1])                # Last known step (t-1)
            Y_aligned.append(Y_raw[i-1])                # Target formed at t-1 (Shifted -15 bars ahead)
            price_aligned.append(prices[i-1])
            
        return np.array(X_seq), np.array(X_spatial), np.array(Y_aligned), np.array(price_aligned)
