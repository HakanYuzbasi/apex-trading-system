import pandas as pd
import numpy as np
from models.adaptive_regime_detector import AdaptiveRegimeDetector

def test_regime_detection():
    # Create synthetic test data with a clear trend
    pd.date_range('2024-01-01', periods=200, freq='D')
    
    # 1. Neutral portion
    prices_neutral = 100 + np.random.randn(60)
    
    # 2. Bull portion
    prices_bull = 100 + np.cumsum(np.random.randn(70) + 0.5)
    
    # 3. Bear portion
    prices_bear = prices_bull[-1] + np.cumsum(np.random.randn(70) - 0.5)
    
    prices = np.concatenate([prices_neutral, prices_bull, prices_bear])
    prices_series = pd.Series(prices, index=pd.date_range('2024-01-01', periods=len(prices), freq='D'))
    
    detector = AdaptiveRegimeDetector()
    regimes = detector.classify_history(prices_series)
    
    print(f"Total samples: {len(prices_series)}")
    print(f"Regime counts:\n{regimes.value_counts()}")
    
    # Verify that we have some non-neutral regimes
    has_bull = (regimes == 'bull').any()
    has_bear = (regimes == 'bear').any()
    
    print(f"Has Bull: {has_bull}")
    print(f"Has Bear: {has_bear}")

if __name__ == "__main__":
    test_regime_detection()
