#!/usr/bin/env python3
"""
Simple manual circuit breaker reset script.
Allows you to manually set the day_start_capital and reset circuit breaker.
"""

import json
import sys
from pathlib import Path
from datetime import datetime

def main():
    print("=" * 80)
    print("🔧 APEX Trading - Manual Circuit Breaker Reset")
    print("=" * 80)
    print()
    
    risk_state_file = Path("data/risk_state.json")
    
    if not risk_state_file.exists():
        print(f"❌ Risk state file not found: {risk_state_file}")
        return 1
    
    # Load current state
    with open(risk_state_file, 'r') as f:
        state = json.load(f)
    
    print("📊 Current Risk State:")
    print(f"   Day Start Capital: ${state['day_start_capital']:,.2f}")
    print(f"   Peak Capital: ${state['peak_capital']:,.2f}")
    print(f"   Starting Capital: ${state['starting_capital']:,.2f}")
    print(f"   Current Day: {state['current_day']}")
    print()
    print(f"   Circuit Breaker Tripped: {state['circuit_breaker']['is_tripped']}")
    if state['circuit_breaker']['is_tripped']:
        print(f"   Trip Reason: {state['circuit_breaker']['reason']}")
        print(f"   Trip Time: {state['circuit_breaker']['trip_time']}")
    print()
    
    # Get user input for actual equity
    print("What is the ACTUAL current equity of your account?")
    print("(Check your broker - IBKR or Alpaca)")
    print()
    
    try:
        actual_equity_input = input("Enter actual current equity ($): ").strip().replace(',', '').replace('$', '')
        actual_equity = float(actual_equity_input)
        
        if actual_equity <= 0:
            print("❌ Invalid equity value (must be positive)")
            return 1
            
    except (ValueError, KeyboardInterrupt, EOFError):
        print("\n❌ Invalid input or cancelled")
        return 1
    
    print(f"\n📍 You entered: ${actual_equity:,.2f}")
    confirm = input("\nIs this correct? (yes/no): ").strip().lower()
    
    if confirm not in ('yes', 'y'):
        print("❌ Cancelled")
        return 1
    
    # Create backup
    backup_file = risk_state_file.with_suffix(f".json.bak.{datetime.now().strftime('%Y%m%dT%H%M%SZ')}")
    with open(backup_file, 'w') as f:
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
    
    print()
    print("✅ Fixed Risk State:")
    print(f"   Day Start Capital: ${state['day_start_capital']:,.2f}")
    print(f"   Peak Capital: ${state['peak_capital']:,.2f}")
    print(f"   Current Day: {today}")
    print(f"   Circuit Breaker: RESET")
    print()
    print("✅ Circuit breaker has been reset!")
    print()
    print("⚠️  The trading system will pick up this change automatically")
    print("   on the next cycle (within ~30 seconds)")
    print()
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
