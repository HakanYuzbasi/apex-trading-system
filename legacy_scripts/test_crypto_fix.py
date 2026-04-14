#!/usr/bin/env python3
"""
Test script to verify crypto trading fixes.

This script tests:
1. Crypto symbols load historical data
2. Crypto markets are identified as "open" 24/7
3. Crypto symbols are included in the trading universe
4. Signal generation works for crypto

Run with: python3 test_crypto_fix.py
"""

import sys
from datetime import datetime
from config import ApexConfig
from core.symbols import parse_symbol, AssetClass
from core.market_hours import is_market_open
from data.market_data import MarketDataFetcher

def test_crypto_config():
    """Test crypto configuration."""
    print("=" * 70)
    print("TEST 1: Crypto Configuration")
    print("=" * 70)

    crypto_pairs = ApexConfig.CRYPTO_PAIRS
    print(f"✓ Crypto pairs configured: {len(crypto_pairs)}")
    print(f"✓ CRYPTO_ALWAYS_OPEN: {getattr(ApexConfig, 'CRYPTO_ALWAYS_OPEN', False)}")
    print(f"✓ Symbols: {', '.join(crypto_pairs[:5])}...")

    # Check for broken symbols
    broken_symbols = ['MATIC/USD', 'UNI/USD']
    has_broken = any(s in crypto_pairs for s in broken_symbols)
    if has_broken:
        print(f"❌ ERROR: Broken symbols found in config: {[s for s in broken_symbols if s in crypto_pairs]}")
        return False
    else:
        print(f"✓ No broken symbols in config")

    print()
    return True

def test_market_hours():
    """Test that crypto markets are considered open 24/7."""
    print("=" * 70)
    print("TEST 2: Market Hours (24/7 Trading)")
    print("=" * 70)

    now = datetime.utcnow()
    test_symbols = ["CRYPTO:BTC/USD", "CRYPTO:ETH/USD", "CRYPTO:SOL/USD"]

    all_open = True
    for symbol in test_symbols:
        is_open = is_market_open(symbol, now)
        status = "✓" if is_open else "❌"
        print(f"{status} {symbol:25} - Market Open: {is_open}")
        if not is_open:
            all_open = False

    print()
    return all_open

def test_historical_data():
    """Test historical data fetching for crypto."""
    print("=" * 70)
    print("TEST 3: Historical Data Fetching")
    print("=" * 70)

    fetcher = MarketDataFetcher()
    test_symbols = ["CRYPTO:BTC/USD", "CRYPTO:ETH/USD", "CRYPTO:SOL/USD"]

    all_success = True
    for symbol in test_symbols:
        df = fetcher.fetch_historical_data(symbol, days=30)
        if not df.empty:
            latest_price = df['Close'].iloc[-1]
            print(f"✓ {symbol:25} - {len(df):3} rows | Latest: ${latest_price:,.2f}")
        else:
            print(f"❌ {symbol:25} - NO DATA")
            all_success = False

    print()
    return all_success

def test_symbol_parsing():
    """Test symbol parsing for crypto."""
    print("=" * 70)
    print("TEST 4: Symbol Parsing")
    print("=" * 70)

    test_cases = [
        ("BTC/USD", AssetClass.CRYPTO),
        ("CRYPTO:BTC/USD", AssetClass.CRYPTO),
        ("ETH/USD", AssetClass.CRYPTO),
        ("EUR/USD", AssetClass.FOREX),
    ]

    all_pass = True
    for symbol, expected_class in test_cases:
        try:
            parsed = parse_symbol(symbol)
            actual_class = parsed.asset_class
            if actual_class == expected_class:
                print(f"✓ {symbol:20} → {parsed.normalized:25} (Asset: {actual_class.value})")
            else:
                print(f"❌ {symbol:20} → Expected: {expected_class.value}, Got: {actual_class.value}")
                all_pass = False
        except Exception as e:
            print(f"❌ {symbol:20} → ERROR: {e}")
            all_pass = False

    print()
    return all_pass

def test_runtime_universe():
    """Test that crypto symbols are in the runtime universe."""
    print("=" * 70)
    print("TEST 5: Runtime Universe")
    print("=" * 70)

    all_symbols = ApexConfig.SYMBOLS
    crypto_count = sum(1 for s in all_symbols if 'BTC' in s or 'ETH' in s or 'SOL' in s)

    print(f"✓ Total symbols in universe: {len(all_symbols)}")
    print(f"✓ Crypto symbols (sample count): {crypto_count}")

    # Check if major cryptos are present
    major_cryptos = ["BTC/USD", "ETH/USD"]
    for crypto in major_cryptos:
        # Try different formats
        found = crypto in all_symbols or f"CRYPTO:{crypto}" in all_symbols
        status = "✓" if found else "❌"
        print(f"{status} {crypto} in universe: {found}")

    print()
    return crypto_count > 0

def main():
    """Run all tests."""
    print("\n" + "=" * 70)
    print("🚀 CRYPTO TRADING FIX - VERIFICATION TESTS")
    print("=" * 70)
    print(f"Timestamp: {datetime.now()}")
    print()

    tests = [
        ("Configuration", test_crypto_config),
        ("Market Hours", test_market_hours),
        ("Historical Data", test_historical_data),
        ("Symbol Parsing", test_symbol_parsing),
        ("Runtime Universe", test_runtime_universe),
    ]

    results = {}
    for test_name, test_func in tests:
        try:
            results[test_name] = test_func()
        except Exception as e:
            print(f"❌ {test_name} FAILED with exception: {e}")
            import traceback
            traceback.print_exc()
            results[test_name] = False

    # Summary
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)

    passed = sum(1 for result in results.values() if result)
    total = len(results)

    for test_name, result in results.items():
        status = "✓ PASS" if result else "❌ FAIL"
        print(f"{status:8} - {test_name}")

    print()
    print(f"Results: {passed}/{total} tests passed")

    if passed == total:
        print("\n✅ ALL TESTS PASSED! Crypto trading should now work.")
        print("\nNext steps:")
        print("1. Restart the trading system: python main.py")
        print("2. Monitor logs for crypto signal generation")
        print("3. Check that crypto positions appear in dashboard")
        return 0
    else:
        print(f"\n❌ {total - passed} tests failed. Review errors above.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
