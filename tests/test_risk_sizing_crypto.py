import sys
import os
from pathlib import Path

# Add project root to sys.path
sys.path.append(str(Path(__file__).parent.parent))

from risk.institutional_risk_manager import InstitutionalRiskManager, RiskConfig
from core.symbols import AssetClass
import pandas as pd
import numpy as np

def test_crypto_sizing():
    print("🚀 Testing Crypto Sizing Logic...")
    config = RiskConfig(min_position_pct=0.01) # 1% min default
    mgr = InstitutionalRiskManager(config)
    mgr.initialize(1100000) # $1.1M capital
    
    # 1. Test EQUITY - should be blocked if size < $11k
    # base_value = 5% of 1.1M = 55k. signal_factor = sqrt(0.1*1)=0.316. 55k * 0.316 = 17.4k. 
    # To get below 11k, we need signal_factor < 0.2. But there is a floor of 0.5.
    # Ah, the floor means min size is 2.5% = $27.5k.
    # Wait, if signal_factor floor is 0.5, then min_position_pct=1% is almost never hit unless capital is very small.
    # On $1.1M, 1% is $11k. 5% base is $55k. Floor 0.5 * 55k = $27.5k.
    # So $27.5k > $11k.
    
    # Let's test with a much smaller max_position_pct to hit the min.
    config.max_position_pct = 0.015 # 1.5% max = $16.5k
    # base_value = $16.5k. floor 0.5 * 16.5k = $8.25k. 
    # $8.25k < $11k (1% min). Should be blocked.
    
    mgr.current_capital = 1100000
    equity_size = mgr.calculate_position_size(
        symbol="AAPL",
        price=150.0,
        signal_strength=0.1,
        signal_confidence=1.0,
        current_positions={},
        price_cache={"AAPL": 150.0},
        asset_class=AssetClass.EQUITY
    )
    print(f"Equity target value ($8.25k target): ${equity_size.target_value:,.2f}")
    if equity_size.target_value == 0 and "below_minimum" in str(equity_size.constraints):
        print("✅ Equity correctly blocked below 1% ($11k)")
    else:
        print(f"❌ Equity NOT blocked as expected. Constraints: {equity_size.constraints}")

    # 2. Test CRYPTO - should ALLOW $8.25k trades (0.1% floor = $1.1k)
    crypto_size = mgr.calculate_position_size(
        symbol="CRYPTO:BTC/USD",
        price=50000.0,
        signal_strength=0.1,
        signal_confidence=1.0,
        current_positions={},
        price_cache={"CRYPTO:BTC/USD": 50000.0},
        asset_class=AssetClass.CRYPTO
    )
    
    print(f"Crypto target value ($8.25k target): ${crypto_size.target_value:,.2f}")
    if crypto_size.target_value == 8250.0:
        print("✅ Crypto correctly allowed below 1% ($11k) but above 0.1% ($1.1k)")
    else:
        print(f"❌ Crypto NOT allowed as expected. Value: ${crypto_size.target_value:,.2f}. Constraints: {crypto_size.constraints}")

    # 3. Test Sector Exposure
    # Add a mock position in 'Crypto' sector.
    price_cache = {"CRYPTO:ETH/USD": 3000.0, "CRYPTO:BTC/USD": 50000.0}
    current_positions = {"CRYPTO:ETH/USD": 100} # $300k in ETH
    
    mgr.current_capital = 1000000 # $1M for easier math
    # Exposure = 300k / 1M = 30%. Limit is 30%. Should block new entries.
    
    crypto_re_entry = mgr.calculate_position_size(
        symbol="CRYPTO:BTC/USD",
        price=50000.0,
        signal_strength=0.5,
        signal_confidence=1.0,
        current_positions=current_positions,
        price_cache=price_cache,
        sector="Crypto",
        asset_class=AssetClass.CRYPTO
    )
    
    print(f"Crypto re-entry value (30% already in sector): ${crypto_re_entry.target_value:,.2f}")
    if crypto_re_entry.target_value == 0 and "sector_limit" in crypto_re_entry.constraints:
        print("✅ Sector limit correctly enforced")
    else:
        print(f"❌ Sector limit NOT enforced. Value: ${crypto_re_entry.target_value:,.2f}. Constraints: {crypto_re_entry.constraints}")

if __name__ == "__main__":
    test_crypto_sizing()
