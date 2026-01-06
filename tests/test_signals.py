"""
tests/test_signals.py - Test signal generation
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from models.signal_generator import SignalGenerator


def test_signals():
    print("=" * 70)
    print("APEX - Signal Generation Test")
    print("=" * 70)
    
    gen = SignalGenerator()
    
    for symbol in ['AAPL', 'GOOGL', 'MSFT']:
        signal = gen.generate_ml_signal(symbol)
        print(f"\n{symbol}:")
        print(f"  Signal: {signal['signal']:.3f}")
        print(f"  Confidence: {signal['confidence']:.3f}")
        print(f"  Momentum: {signal['momentum']:.3f}")
        print(f"  Mean Rev: {signal['mean_reversion']:.3f}")
    
    print("\n✅ Test complete\n")


if __name__ == "__main__":
    test_signals()
