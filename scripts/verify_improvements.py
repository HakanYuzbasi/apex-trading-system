
import logging
import sys
import os

# Add project root to path
sys.path.append(os.getcwd())

from core.logging_config import setup_logging
from monitoring.model_tracker import ModelPerformanceTracker

def test_logging_suppression():
    print("Testing logging suppression...")
    setup_logging()
    
    uvicorn_logger = logging.getLogger("uvicorn.access")
    if uvicorn_logger.level == logging.WARNING:
        print("✅ uvicorn.access level is WARNING")
    else:
        print(f"❌ uvicorn.access level is {logging.getLevelName(uvicorn_logger.level)}")

def test_model_tracker():
    print("\nTesting ModelPerformanceTracker...")
    tracker = ModelPerformanceTracker()
    
    symbol = "TEST"
    price_1 = 100.0
    pred_return = 0.01 # +1%
    
    # Log prediction
    tracker.log_prediction(symbol, pred_return, price_1)
    print("Logged prediction +1% at $100")
    
    # Update price (move up)
    price_2 = 102.0
    tracker.on_price_update(symbol, price_2)
    
    acc = tracker.get_accuracy(symbol)
    print(f"Accuracy after correct move: {acc:.2%}")
    
    if acc == 1.0:
        print("✅ Accuracy calculation correct (1.0)")
    else:
        print(f"❌ Accuracy calculation failed ({acc})")

def test_signal_generator_init():
    print("\nTesting Component Initialization...")
    try:
        from models.institutional_signal_generator import UltimateSignalGenerator
        # We won't fully init it as it needs models/data, but just checking import and class existence
        # and that imports didn't break
        print("✅ UltimateSignalGenerator imported successfully")
        return True
    except ImportError as e:
        print(f"❌ Failed to import UltimateSignalGenerator: {e}")
        return False
    except Exception as e:
        print(f"❌ Error during import: {e}")
        return False

if __name__ == "__main__":
    test_logging_suppression()
    test_model_tracker()
    test_signal_generator_init()
