#!/usr/bin/env python3
"""
Debug script to check why crypto isn't trading.

This will check the actual runtime state of the trading system.
"""

import sys
import asyncio
from datetime import datetime
from config import ApexConfig
from core.symbols import parse_symbol, AssetClass, is_market_open
from data.market_data import MarketDataFetcher

async def main():
    print("=" * 70)
    print("🔍 CRYPTO TRADING DEBUG")
    print("=" * 70)
    print()

    # 1. Check config
    print("1. Configuration Check:")
    print(f"   CRYPTO_ALWAYS_OPEN: {ApexConfig.CRYPTO_ALWAYS_OPEN}")
    print(f"   BROKER_MODE: {ApexConfig.BROKER_MODE}")
    print(f"   Total SYMBOLS: {len(ApexConfig.SYMBOLS)}")

    crypto_in_config = [s for s in ApexConfig.SYMBOLS
                        if parse_symbol(s).asset_class == AssetClass.CRYPTO]
    print(f"   Crypto symbols in config: {len(crypto_in_config)}")
    print(f"   Sample: {crypto_in_config[:3]}")
    print()

    # 2. Check market hours
    print("2. Market Hours Check:")
    now = datetime.utcnow()
    test_symbols = ["CRYPTO:BTC/USD", "CRYPTO:ETH/USD", "SPY"]
    for sym in test_symbols:
        is_open = is_market_open(sym, now)
        print(f"   {sym:25} -> {'OPEN' if is_open else 'CLOSED'}")
    print()

    # 3. Check data availability
    print("3. Historical Data Check (simulating startup):")
    fetcher = MarketDataFetcher()

    test_cryptos = crypto_in_config[:3]
    for symbol in test_cryptos:
        df = fetcher.fetch_historical_data(symbol, days=400)
        if not df.empty:
            print(f"   ✅ {symbol:25} - {len(df)} days")
        else:
            print(f"   ❌ {symbol:25} - NO DATA")
    print()

    # 4. Simulate open_universe filtering
    print("4. Runtime Universe Simulation:")
    print(f"   All symbols: {len(ApexConfig.SYMBOLS)}")

    # Simulate the filter used in execution loop
    open_universe = [s for s in ApexConfig.SYMBOLS
                     if is_market_open(s, now)]
    crypto_open = [s for s in open_universe
                   if parse_symbol(s).asset_class == AssetClass.CRYPTO]

    print(f"   Open markets now: {len(open_universe)}")
    print(f"   Crypto in open_universe: {len(crypto_open)}")
    if crypto_open:
        print(f"   Sample crypto in open_universe: {crypto_open[:3]}")
    else:
        print(f"   ⚠️  NO CRYPTO in open_universe!")
    print()

    # 5. Check Alpaca connection
    print("5. Alpaca Connection Check:")
    try:
        from execution.alpaca_connector import AlpacaConnector
        alpaca = AlpacaConnector(
            api_key=getattr(ApexConfig, "ALPACA_API_KEY", ""),
            secret_key=getattr(ApexConfig, "ALPACA_SECRET_KEY", ""),
            base_url=getattr(ApexConfig, "ALPACA_BASE_URL", ""),
        )
        await alpaca.connect()
        equity = await alpaca.get_portfolio_value()
        print(f"   ✅ Connected: ${equity:,.2f} available")

        # Get discovered symbols
        if getattr(ApexConfig, "ALPACA_DISCOVER_CRYPTO_SYMBOLS", True):
            discovered = alpaca.get_discovered_crypto_symbols(
                limit=24,
                preferred_quotes=["USD", "USDT", "USDC"]
            )
            print(f"   ✅ Discovered {len(discovered)} crypto symbols")
            print(f"   Sample: {discovered[:3]}")
    except Exception as e:
        print(f"   ❌ Alpaca error: {e}")
    print()

    # 6. Check why signals might not generate
    print("6. Potential Issues:")
    issues = []

    if len(crypto_open) == 0:
        issues.append("⚠️  No crypto symbols in open_universe")

    if not ApexConfig.CRYPTO_ALWAYS_OPEN:
        issues.append("⚠️  CRYPTO_ALWAYS_OPEN is False")

    if ApexConfig.BROKER_MODE not in ("both", "alpaca"):
        issues.append(f"⚠️  BROKER_MODE is '{ApexConfig.BROKER_MODE}' (needs 'both' or 'alpaca')")

    if not issues:
        print("   ✅ No obvious issues found!")
        print("   → Crypto SHOULD be trading")
        print()
        print("   Possible causes:")
        print("   1. Historical data didn't load at startup (check logs)")
        print("   2. System needs time to generate first signals")
        print("   3. No strong signals yet (normal)")
    else:
        for issue in issues:
            print(f"   {issue}")
    print()

    print("=" * 70)
    print("Next Steps:")
    print("1. Check logs: tail -f logs/apex.log | grep -i crypto")
    print("2. Wait for equity market open (9:30 AM EST)")
    print("3. Monitor for crypto signals in next 1-2 hours")
    print("=" * 70)

if __name__ == "__main__":
    asyncio.run(main())
