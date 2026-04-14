#!/usr/bin/env python3
"""
Quick script to check what's in the open_universe during runtime.
"""

import sys
from datetime import datetime
from config import ApexConfig
from core.symbols import is_market_open, parse_symbol, AssetClass

def main():
    print("=" * 70)
    print("🔍 CHECKING OPEN UNIVERSE")
    print("=" * 70)
    print()

    # Simulate _runtime_symbols()
    all_symbols = ApexConfig.SYMBOLS
    print(f"Total symbols in ApexConfig.SYMBOLS: {len(all_symbols)}")

    # Count by asset class
    crypto_count = sum(1 for s in all_symbols if 'CRYPTO:' in s or '/' in s)
    equity_count = len(all_symbols) - crypto_count
    print(f"  Equities: {equity_count}")
    print(f"  Crypto: {crypto_count}")
    print()

    # Filter to open markets (simulating line 6783)
    now = datetime.utcnow()
    open_universe = [s for s in all_symbols if is_market_open(s, now)]

    print(f"Open markets now ({now.strftime('%Y-%m-%d %H:%M:%S UTC')}):")
    print(f"  Total open: {len(open_universe)}")

    # Breakdown by asset class
    open_crypto = []
    open_equity = []
    open_fx = []

    for symbol in open_universe:
        try:
            parsed = parse_symbol(symbol)
            if parsed.asset_class == AssetClass.CRYPTO:
                open_crypto.append(symbol)
            elif parsed.asset_class == AssetClass.FOREX:
                open_fx.append(symbol)
            else:
                open_equity.append(symbol)
        except:
            open_equity.append(symbol)

    print(f"  Open equities: {len(open_equity)}")
    print(f"  Open FX: {len(open_fx)}")
    print(f"  Open crypto: {len(open_crypto)}")
    print()

    if open_crypto:
        print(f"✅ Crypto symbols in open_universe (first 10):")
        for sym in open_crypto[:10]:
            print(f"   - {sym}")
    else:
        print(f"❌ NO CRYPTO in open_universe!")
        print()
        print("Checking why...")

        # Check if crypto exists in ApexConfig.SYMBOLS
        all_crypto = [s for s in all_symbols if 'CRYPTO:' in s or ('/' in s and 'EUR/' not in s)]
        print(f"  Crypto in ApexConfig.SYMBOLS: {len(all_crypto)}")
        if all_crypto:
            print(f"  Sample: {all_crypto[:3]}")

            # Check if they're considered open
            for sym in all_crypto[:5]:
                is_open = is_market_open(sym, now)
                print(f"    {sym}: is_market_open = {is_open}")

    print()
    print("=" * 70)

    return 0 if open_crypto else 1

if __name__ == "__main__":
    sys.exit(main())
