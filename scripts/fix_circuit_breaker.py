#!/usr/bin/env python3
"""
Fix circuit breaker false trigger caused by incorrect day_start_capital.

This script:
1. Connects to IBKR/Alpaca to get actual current equity
2. Updates risk_state.json with correct day_start_capital
3. Resets circuit breaker
"""

import json
import sys
from pathlib import Path
from datetime import datetime

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import ApexConfig
from execution.ibkr_connector import IBKRConnector

def get_current_equity_from_ibkr():
    """Get current equity from IBKR."""
    try:
        connector = IBKRConnector(
            host=ApexConfig.IBKR_HOST,
            port=ApexConfig.IBKR_PORT,
            client_id=ApexConfig.IBKR_CLIENT_ID + 100  # Use different client ID
        )
        
        # Get account summary
        equity = connector.get_total_equity()
        cash = connector.get_cash_balance()
        
        print("✅ IBKR Connection successful")
        print(f"   Total Equity: ${equity:,.2f}")
        print(f"   Cash Balance: ${cash:,.2f}")
        
        connector.ib.disconnect()
        return equity
        
    except Exception as e:
        print(f"❌ Failed to connect to IBKR: {e}")
        return None

def get_current_equity_from_alpaca():
    """Get current equity from Alpaca (fallback)."""
    try:
        from execution.alpaca_connector import AlpacaConnector
        
        connector = AlpacaConnector(
            api_key=getattr(ApexConfig, "ALPACA_API_KEY", ""),
            secret_key=getattr(ApexConfig, "ALPACA_SECRET_KEY", ""),
            base_url=getattr(ApexConfig, "ALPACA_BASE_URL", ""),
        )
        
        account = connector.api.get_account()
        equity = float(account.equity)
        
        print("✅ Alpaca Connection successful")
        print(f"   Total Equity: ${equity:,.2f}")
        
        return equity
        
    except Exception as e:
        print(f"❌ Failed to connect to Alpaca: {e}")
        return None

def fix_risk_state(actual_equity: float):
    """Fix risk_state.json with correct values."""
    risk_state_file = ApexConfig.DATA_DIR / "risk_state.json"
    
    if not risk_state_file.exists():
        print(f"❌ Risk state file not found: {risk_state_file}")
        return False
    
    # Load current state
    with open(risk_state_file, 'r') as f:
        state = json.load(f)
    
    print("\n📊 Current Risk State:")
    print(f"   Day Start Capital: ${state['day_start_capital']:,.2f}")
    print(f"   Peak Capital: ${state['peak_capital']:,.2f}")
    print(f"   Starting Capital: ${state['starting_capital']:,.2f}")
    print(f"   Circuit Breaker Tripped: {state['circuit_breaker']['is_tripped']}")
    if state['circuit_breaker']['is_tripped']:
        print(f"   Trip Reason: {state['circuit_breaker']['reason']}")
    
    # Create backup
    backup_file = risk_state_file.with_suffix(f".json.bak.{datetime.now().strftime('%Y%m%dT%H%M%SZ')}")
    with open(backup_file,  'w') as f:
        json.dump(state, f, indent=2)
    print(f"\n💾 Backup saved to: {backup_file}")
    
    # Fix the values
    today = datetime.now().strftime('%Y-%m-%d')
    state['day_start_capital'] = actual_equity
    state['peak_capital'] = max(actual_equity, state.get('peak_capital', 0))
    state['current_day'] = today
    state['circuit_breaker'] = {
        'is_tripped': False,
        'reason': None,
        'trip_time': None,
        'consecutive_losses': 0,
        'recent_trades': 0
    }
    
    # Save fixed state
    with open(risk_state_file, 'w') as f:
        json.dump(state, f, indent=2)
    
    print("\n✅ Fixed Risk State:")
    print(f"   Day Start Capital: ${state['day_start_capital']:,.2f}")
    print(f"   Peak Capital: ${state['peak_capital']:,.2f}")
    print("   Circuit Breaker: RESET")
    
    return True

def main():
    print("=" * 80)
    print("🔧 APEX Trading - Circuit Breaker Fix")
    print("=" * 80)
    print()
    
    # Try to get actual equity
    print("🔍 Fetching actual equity from broker...")
    actual_equity = None
    
    # Try IBKR first
    if ApexConfig.LIVE_TRADING:
        broker_mode = getattr(ApexConfig, "BROKER_MODE", "ibkr").lower()
        
        if broker_mode in ("ibkr", "both"):
            actual_equity = get_current_equity_from_ibkr()
        
        if actual_equity is None and broker_mode in ("alpaca", "both"):
            actual_equity = get_current_equity_from_alpaca()
    
    if actual_equity is None:
        print("\n❌ Could not fetch actual equity from broker")
        print("Please enter the actual current equity manually:")
        try:
            actual_equity = float(input("Current Equity ($): "))
        except (ValueError, KeyboardInterrupt):
            print("\n❌ Invalid input or cancelled")
            return 1
    
    print(f"\n📍 Using Actual Equity: ${actual_equity:,.2f}")
    
    # Fix risk state
    if fix_risk_state(actual_equity):
        print("\n✅ Circuit breaker has been reset!")
        print("\n⚠️  Next steps:")
        print("   1. Restart the trading system to pick up the corrected state")
        print("   2. Monitor logs to verify correct daily P&L calculation")
        return 0
    else:
        print("\n❌ Failed to fix risk state")
        return 1

if __name__ == "__main__":
    sys.exit(main())
