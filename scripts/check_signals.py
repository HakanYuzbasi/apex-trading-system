#!/usr/bin/env python3
"""
Quick script to check current signal values and see which symbols
would trigger trades.
"""
import sys
sys.path.insert(0, '/home/user/apex-trading-system')

from models.advanced_signal_generator import AdvancedSignalGenerator
from data.market_data import MarketDataFetcher
from config import ApexConfig

def check_signals():
    print("=" * 70)
    print("APEX SIGNAL CHECK - Which symbols would trigger trades?")
    print("=" * 70)
    print(f"\nEntry thresholds: Signal >= {ApexConfig.MIN_SIGNAL_THRESHOLD}, Confidence >= {ApexConfig.MIN_CONFIDENCE}")
    print("-" * 70)

    # Initialize
    signal_gen = AdvancedSignalGenerator()
    market_data = MarketDataFetcher()

    # Fetch data for a subset of symbols
    symbols = ApexConfig.SYMBOLS[:20]  # Check first 20 symbols
    print(f"\nFetching data for {len(symbols)} symbols...")

    historical_data = {}
    for symbol in symbols:
        try:
            df = market_data.fetch_historical_data(symbol, days=252)
            if df is not None and not df.empty:
                historical_data[symbol] = df
                print(f"  ✓ {symbol}: {len(df)} bars")
        except Exception as e:
            print(f"  ✗ {symbol}: {e}")

    if not historical_data:
        print("No data fetched. Check your internet connection.")
        return

    # Train models
    print(f"\nTraining models on {len(historical_data)} symbols...")
    signal_gen.train_models(historical_data)

    if not signal_gen.models_trained:
        print("Model training failed!")
        return

    # Check signals
    print("\n" + "=" * 70)
    print("SIGNAL ANALYSIS")
    print("=" * 70)

    would_trade = []
    all_signals = []

    for symbol, df in historical_data.items():
        # This version expects a Series (Close prices)
        prices = df['Close']
        result = signal_gen.generate_ml_signal(symbol, prices)
        signal = result['signal']
        confidence = result['confidence']

        all_signals.append({
            'symbol': symbol,
            'signal': signal,
            'confidence': confidence
        })

        # Check if would trigger trade
        meets_signal = abs(signal) >= ApexConfig.MIN_SIGNAL_THRESHOLD
        meets_confidence = confidence >= ApexConfig.MIN_CONFIDENCE

        if meets_signal and meets_confidence:
            direction = "BUY" if signal > 0 else "SELL"
            would_trade.append((symbol, signal, confidence, direction))

    # Sort by signal strength
    all_signals.sort(key=lambda x: abs(x['signal']), reverse=True)

    print("\nAll signals (sorted by strength):")
    print(f"{'Symbol':<8} {'Signal':>8} {'Conf':>8} {'Would Trade?':>14}")
    print("-" * 40)

    for s in all_signals:
        meets = "✓ YES" if abs(s['signal']) >= ApexConfig.MIN_SIGNAL_THRESHOLD and s['confidence'] >= ApexConfig.MIN_CONFIDENCE else "  no"
        print(f"{s['symbol']:<8} {s['signal']:>8.3f} {s['confidence']:>8.3f} {meets:>14}")

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Total symbols checked: {len(all_signals)}")
    print(f"Would trigger trades:  {len(would_trade)}")

    if would_trade:
        print("\nSymbols ready to trade:")
        for symbol, signal, conf, direction in would_trade:
            print(f"  {symbol}: {direction} (signal={signal:.3f}, confidence={conf:.3f})")
    else:
        print("\n⚠️  No symbols currently meet entry criteria.")
        print("   This could mean:")
        print("   1. Market is in neutral/uncertain state (normal)")
        print("   2. Thresholds are too strict (try lowering MIN_SIGNAL_THRESHOLD)")
        print("   3. Models need more diverse training data")

if __name__ == "__main__":
    check_signals()
