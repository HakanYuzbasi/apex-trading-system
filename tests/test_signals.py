"""
tests/test_signals.py - Test signal generation

Tests both basic SignalGenerator and AdvancedSignalGenerator with real data.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np
from models.signal_generator import SignalGenerator
from data.market_data import MarketDataFetcher


def test_signals_with_real_data():
    """Test signal generation with real market data."""
    print("=" * 70)
    print("APEX - Signal Generation Test (Real Data)")
    print("=" * 70)

    # Initialize components
    gen = SignalGenerator()
    market_data = MarketDataFetcher()

    test_symbols = ['AAPL', 'GOOGL', 'MSFT', 'SPY']

    for symbol in test_symbols:
        print(f"\n{symbol}:")

        # Fetch real historical data
        data = market_data.fetch_historical_data(symbol, days=100)

        if data.empty:
            print(f"  ⚠️  No data available for {symbol}")
            continue

        prices = data['Close']
        print(f"  Data points: {len(prices)}")
        print(f"  Price range: ${prices.min():.2f} - ${prices.max():.2f}")

        # Generate signal with real prices
        signal = gen.generate_ml_signal(symbol, prices)

        print(f"  Signal: {signal['signal']:+.3f}")
        print(f"  Confidence: {signal['confidence']:.3f}")
        print(f"  Momentum: {signal['momentum']:+.3f}")
        print(f"  Mean Rev: {signal['mean_reversion']:+.3f}")
        print(f"  Trend: {signal.get('trend', 0):+.3f}")
        print(f"  RSI: {signal.get('rsi', 0):+.3f}")

        # Validate signal bounds
        assert -1 <= signal['signal'] <= 1, f"Signal out of bounds: {signal['signal']}"
        assert 0 <= signal['confidence'] <= 1, f"Confidence out of bounds: {signal['confidence']}"

    print("\n✅ Signal generation test complete\n")


def test_signals_with_synthetic_data():
    """Test signal generation with synthetic price data."""
    print("=" * 70)
    print("APEX - Signal Generation Test (Synthetic Data)")
    print("=" * 70)

    gen = SignalGenerator()

    # Create synthetic uptrend data
    np.random.seed(42)
    uptrend = pd.Series(np.cumsum(np.random.randn(100) * 0.5 + 0.1) + 100)

    print("\n📈 Uptrend data:")
    signal = gen.generate_ml_signal("UPTREND", uptrend)
    print(f"  Signal: {signal['signal']:+.3f}")
    print(f"  Expected: positive (bullish)")
    assert signal['signal'] > 0, f"Expected positive signal for uptrend, got {signal['signal']}"

    # Create synthetic downtrend data
    downtrend = pd.Series(np.cumsum(np.random.randn(100) * 0.5 - 0.1) + 100)

    print("\n📉 Downtrend data:")
    signal = gen.generate_ml_signal("DOWNTREND", downtrend)
    print(f"  Signal: {signal['signal']:+.3f}")
    print(f"  Expected: negative (bearish)")
    assert signal['signal'] < 0, f"Expected negative signal for downtrend, got {signal['signal']}"

    # Test with insufficient data
    print("\n⚠️  Insufficient data:")
    short_prices = pd.Series([100, 101, 102])
    signal = gen.generate_ml_signal("SHORT", short_prices)
    print(f"  Signal: {signal['signal']:.3f}")
    print(f"  Expected: near zero (insufficient data)")
    assert abs(signal['signal']) < 0.5, "Expected weak signal with insufficient data"

    print("\n✅ Synthetic data test complete\n")


def test_signal_generator_no_prices():
    """Test that signal generator handles missing prices gracefully."""
    print("=" * 70)
    print("APEX - Signal Generator Edge Cases")
    print("=" * 70)

    gen = SignalGenerator()

    # Test with None prices
    print("\n📭 No prices provided:")
    signal = gen.generate_ml_signal("TEST", None)
    print(f"  Signal: {signal['signal']:.3f}")
    assert signal['signal'] == 0.0, "Expected zero signal with no prices"
    assert signal['confidence'] == 0.0, "Expected zero confidence with no prices"

    # Test with empty series
    print("\n📭 Empty price series:")
    signal = gen.generate_ml_signal("TEST", pd.Series([]))
    print(f"  Signal: {signal['signal']:.3f}")
    assert signal['signal'] == 0.0, "Expected zero signal with empty prices"

    print("\n✅ Edge case tests complete\n")


if __name__ == "__main__":
    try:
        test_signals_with_synthetic_data()
        test_signal_generator_no_prices()

        print("\n" + "=" * 70)
        print("Attempting real data test (requires internet)...")
        print("=" * 70)
        test_signals_with_real_data()

    except KeyboardInterrupt:
        print("\n\n👋 Test cancelled")
    except Exception as e:
        print(f"\n❌ Test error: {e}")
        import traceback
        traceback.print_exc()
