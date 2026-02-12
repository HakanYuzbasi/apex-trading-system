import pytest
import pandas as pd
import numpy as np
from models.advanced_signal_generator import AdvancedSignalGenerator
from models.ensemble_signal_generator import EnsembleSignalGenerator
from models.regime_common import get_regime

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

def test_get_regime(sample_data):
    """Test shared regime utility."""
    regime = get_regime(sample_data["Close"], lookback=60)
    assert regime in ["bull", "bear", "neutral", "volatile"]

def test_advanced_signal_generator_integration(sample_data):
    """Test AdvancedSignalGenerator uses new regime logic without error."""
    # We mock the model loading to avoid needing trained models
    gen = AdvancedSignalGenerator()
    
    # We just want to check if generate_ml_signal runs without import/syntax errors
    # It might return empty signal if no models, but it should not crash on regime detection
    signal = gen.generate_ml_signal("TEST", sample_data, track=False)
    
    assert isinstance(signal, dict)
    assert "regime" in signal
    # The regime might be 'unknown' or 'neutral' depending on internal logic if models missing
    # but the key should be there.

def test_ensemble_signal_generator_integration(sample_data):
    """Test EnsembleSignalGenerator uses new regime logic without error."""
    gen = EnsembleSignalGenerator()
    
    # Check generate_signal
    # For ensemble, generate_signal prepares features then calls predict
    # We need to construct a feature row first?
    # No, generate_signal signature: generate_signal(features, prices, track_for_drift)
    # It expects features as DataFrame (1 row) and prices as Series
    
    features = pd.DataFrame([np.zeros(10)], columns=[f"f{i}" for i in range(10)]) # Dummy
    gen.feature_names = features.columns.tolist() # Hack to bypass feature check
    
    signal = gen.generate_signal(features, sample_data["Close"], track_for_drift=False)
    
    assert isinstance(signal, dict)
    assert "regime" in signal
