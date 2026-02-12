import pytest
import pandas as pd
import numpy as np
from models.advanced_features import FeatureEngine

@pytest.fixture
def sample_data():
    """Create sample OHLCV data."""
    dates = pd.date_range(start="2023-01-01", periods=100)
    df = pd.DataFrame({
        "Open": np.random.uniform(100, 200, 100),
        "High": np.random.uniform(100, 200, 100),
        "Low": np.random.uniform(100, 200, 100),
        "Close": np.random.uniform(100, 200, 100),
        "Volume": np.random.uniform(1000, 2000, 100)
    }, index=dates)
    
    # Ensure High is highest and Low is lowest
    df["High"] = df[["Open", "Close", "High"]].max(axis=1)
    df["Low"] = df[["Open", "Close", "Low"]].min(axis=1)
    return df

def test_initialization():
    fe = FeatureEngine(lookback=60)
    assert len(fe.feature_names) > 30
    assert "rsi_14" in fe.feature_names

def test_extract_features_vectorized(sample_data):
    fe = FeatureEngine()
    features = fe.extract_features_vectorized(sample_data)
    
    assert isinstance(features, pd.DataFrame)
    assert len(features) == len(sample_data)
    
    # Check if key features are present
    assert "rsi_14" in features.columns
    assert "bb_location" in features.columns or "bb_position" in features.columns
    assert "vol_20d" in features.columns
    
    # Check for NaNs (should be filled)
    assert not features["rsi_14"].isna().all()

def test_extract_single_sample(sample_data):
    fe = FeatureEngine()
    features, quality = fe.extract_single_sample(sample_data)
    
    assert isinstance(features, np.ndarray)
    assert len(features) == len(fe.feature_names)
    assert isinstance(quality, float)
    assert 0.0 <= quality <= 1.0

def test_volume_features(sample_data):
    fe = FeatureEngine()
    features = fe.extract_features_vectorized(sample_data)
    
    # Check volume features
    if "obv_zscore" in fe.feature_names:
         assert "obv_zscore" in features.columns

def test_intraday_features(sample_data):
    fe = FeatureEngine()
    features = fe.extract_features_vectorized(sample_data)
    
    # Check intraday features
    if "intraday_range" in fe.feature_names:
         assert "intraday_range" in features.columns
