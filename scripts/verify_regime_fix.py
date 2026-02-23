
import sys
import os
import pandas as pd
import numpy as np

# Add project root to path
sys.path.append(os.getcwd())

from models.regime_common import get_regime

def test_singleton_behavior():
    print("Testing get_regime singleton behavior...")
    
    # Create dummy data
    prices = pd.Series(np.random.normal(100, 1, 100))
    
    # First call
    print("Calling get_regime(1)...")
    get_regime(prices)
    
    # Check if singleton is set
    from models.regime_common import _GLOBAL_DETECTOR as DETECTOR_1
    if DETECTOR_1 is None:
        print("❌ _GLOBAL_DETECTOR is None after first call!")
        return
    print("✅ _GLOBAL_DETECTOR initialized.")
    
    # Second call
    print("Calling get_regime(2)...")
    get_regime(prices)
    
    from models.regime_common import _GLOBAL_DETECTOR as DETECTOR_2
    
    # Verify exact object identity
    if DETECTOR_1 is DETECTOR_2:
        print("✅ Singleton persisted (Same Object ID)")
    else:
        print(f"❌ Singleton failed: {id(DETECTOR_1)} != {id(DETECTOR_2)}")

if __name__ == "__main__":
    test_singleton_behavior()
