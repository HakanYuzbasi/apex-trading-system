#!/usr/bin/env python3
"""
Diagnose P&L calculation discrepancy
"""
import asyncio
from ib_insync import IB, util
from execution.alpaca_connector import AlpacaConnector
from config import ApexConfig
import os

async def main():
    print("=" * 80)
    print("P&L DIAGNOSTIC TOOL")
    print("=" * 80)
    print()

    # Connect to IBKR
    print("1. Connecting to IBKR...")
    ib = IB()
    await ib.connectAsync('127.0.0.1', 7497, clientId=999)

    # Get account summary
    account_values = ib.accountValues()

    print("\n📊 IBKR Account Values:")
    for av in account_values:
        if av.tag in ['NetLiquidation', 'TotalCashValue', 'GrossPositionValue',
                      'RealizedPnL', 'UnrealizedPnL', 'EquityWithLoanValue']:
            print(f"   {av.tag:25} = ${float(av.value):,.2f} ({av.currency})")

    # Get portfolio items
    portfolio = ib.portfolio()

    total_realized = 0.0
    total_unrealized = 0.0

    print(f"\n📈 IBKR Portfolio Items ({len(portfolio)} items):")
    for item in portfolio:
        if item.realizedPNL != 0 or item.unrealizedPNL != 0 or item.position != 0:
            print(f"   {item.contract.symbol:10} pos={item.position:6.0f} "
                  f"realizedPnL=${item.realizedPNL:8,.2f} "
                  f"unrealizedPnL=${item.unrealizedPNL:8,.2f}")
            total_realized += item.realizedPNL
            total_unrealized += item.unrealizedPNL

    print(f"\n   TOTALS: realizedPnL=${total_realized:,.2f} unrealizedPnL=${total_unrealized:,.2f}")

    # Connect to Alpaca
    print("\n2. Connecting to Alpaca...")
    alpaca = AlpacaConnector(
        api_key=os.getenv("ALPACA_API_KEY", ""),
        secret_key=os.getenv("ALPACA_SECRET_KEY", ""),
        base_url=os.getenv("ALPACA_BASE_URL", "https://paper-api.alpaca.markets"),
    )
    await alpaca.connect()

    alpaca_equity = await alpaca.get_portfolio_value()
    print(f"\n📊 Alpaca Account:")
    print(f"   Portfolio Value = ${alpaca_equity:,.2f}")

    # Calculate combined
    ibkr_net_liq = 0.0
    for av in account_values:
        if av.tag == 'NetLiquidation' and av.currency == 'USD':
            ibkr_net_liq = float(av.value)
            break

    combined_equity = ibkr_net_liq + alpaca_equity

    print(f"\n💰 COMBINED EQUITY:")
    print(f"   IBKR NetLiquidation:  ${ibkr_net_liq:,.2f}")
    print(f"   Alpaca Portfolio:     ${alpaca_equity:,.2f}")
    print(f"   ─────────────────────────────────")
    print(f"   TOTAL:                ${combined_equity:,.2f}")

    # Check against trading_state.json
    import json
    from pathlib import Path

    state_file = Path("data/trading_state.json")
    if state_file.exists():
        with open(state_file) as f:
            state = json.load(f)

        print(f"\n📄 trading_state.json:")
        print(f"   capital:              ${state['capital']:,.2f}")
        print(f"   starting_capital:     ${state['starting_capital']:,.2f}")
        print(f"   daily_pnl:            ${state['daily_pnl']:,.2f}")
        print(f"   total_pnl:            ${state['total_pnl']:,.2f}")
        print(f"   daily_pnl_source:     {state['daily_pnl_source']}")

        print(f"\n🔍 DISCREPANCY CHECK:")
        diff = state['capital'] - combined_equity
        print(f"   Stated Capital:       ${state['capital']:,.2f}")
        print(f"   Actual Combined:      ${combined_equity:,.2f}")
        print(f"   DIFFERENCE:           ${diff:,.2f} ({'BUG!' if abs(diff) > 1000 else 'OK'})")

    print("\n" + "=" * 80)

    ib.disconnect()

if __name__ == "__main__":
    asyncio.run(main())
