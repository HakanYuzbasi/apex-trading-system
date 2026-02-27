#!/usr/bin/env python3
"""
Emergency fix for zero-trading issue
Addresses:
1. Weak ML signals (add debug logging)
2. Client ID conflicts
3. Signal threshold analysis
"""
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import ApexConfig
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def analyze_config():
    """Analyze current config for trading issues"""
    print("=" * 80)
    print("APEX TRADING CONFIGURATION ANALYSIS")
    print("=" * 80)
    print()

    print("📊 Signal Thresholds:")
    print(f"   MIN_SIGNAL_THRESHOLD: {ApexConfig.MIN_SIGNAL_THRESHOLD}")
    print(f"   FX_SIGNAL_THRESHOLD: {ApexConfig.FX_SIGNAL_THRESHOLD}")
    print(f"   CRYPTO_SIGNAL_THRESHOLD_MULTIPLIER: {ApexConfig.CRYPTO_SIGNAL_THRESHOLD_MULTIPLIER}")
    print()

    print("🔧 Trading Controls:")
    print(f"   MAX_POSITIONS: {ApexConfig.MAX_POSITIONS}")
    print(f"   LIVE_TRADING: {ApexConfig.LIVE_TRADING}")
    print(f"   OPTIONS_ENABLED: {ApexConfig.OPTIONS_ENABLED}")
    print(f"   CRYPTO_ALWAYS_OPEN: {ApexConfig.CRYPTO_ALWAYS_OPEN}")
    print()

    print("🔌 IBKR Config:")
    print(f"   IBKR_HOST: {ApexConfig.IBKR_HOST}")
    print(f"   IBKR_PORT: {ApexConfig.IBKR_PORT}")
    print(f"   IBKR_CLIENT_ID: {ApexConfig.IBKR_CLIENT_ID}")
    print()

    print("🪙 Crypto Config:")
    print(f"   CRYPTO_PAIRS: {len(ApexConfig.CRYPTO_PAIRS)} pairs")
    print(f"   Sample: {', '.join(list(ApexConfig.CRYPTO_PAIRS)[:5])}")
    print()

    # Check for issues
    issues = []

    if ApexConfig.MIN_SIGNAL_THRESHOLD > 0.20:
        issues.append(f"⚠️  Signal threshold too high ({ApexConfig.MIN_SIGNAL_THRESHOLD}) - lowering to 0.15 recommended")

    if ApexConfig.IBKR_CLIENT_ID == 1:
        issues.append("🔴 IBKR_CLIENT_ID=1 will conflict with API service! Should be 10-99")

    if len(ApexConfig.CRYPTO_PAIRS) < 20:
        issues.append(f"⚠️  Only {len(ApexConfig.CRYPTO_PAIRS)} crypto pairs configured (expected 35+)")

    if issues:
        print("🚨 ISSUES FOUND:")
        for issue in issues:
            print(f"   {issue}")
    else:
        print("✅ No config issues detected")

    print()
    print("=" * 80)

def check_ml_models():
    """Check if ML models exist and are recent"""
    print("\n📚 ML MODEL STATUS:")
    print("=" * 80)

    import glob
    from pathlib import Path
    from datetime import datetime, timedelta

    model_dir = Path(ApexConfig.MODELS_DIR) / "saved_ultimate"

    if not model_dir.exists():
        print(f"❌ Model directory not found: {model_dir}")
        return

    regimes = ["bull", "bear", "neutral", "volatile"]

    for regime in regimes:
        regime_dir = model_dir / regime
        if not regime_dir.exists():
            print(f"⚠️  {regime}: Directory missing")
            continue

        model_files = list(regime_dir.glob("*.json")) + list(regime_dir.glob("*.pkl"))

        if not model_files:
            print(f"❌ {regime}: No models found")
            continue

        # Check most recent model
        latest = max(model_files, key=lambda p: p.stat().st_mtime)
        mod_time = datetime.fromtimestamp(latest.stat().st_mtime)
        age = datetime.now() - mod_time

        if age > timedelta(days=7):
            print(f"⚠️  {regime}: Models old ({age.days}d) - last updated {mod_time.strftime('%Y-%m-%d %H:%M')}")
        else:
            print(f"✅ {regime}: {len(model_files)} models, updated {mod_time.strftime('%Y-%m-%d %H:%M')}")

    print("=" * 80)

def suggest_fixes():
    """Suggest fixes for identified issues"""
    print("\n💡 RECOMMENDED FIXES:")
    print("=" * 80)

    print("\n1. **Add Signal Debug Logging** (Immediate)")
    print("   Edit: core/execution_loop.py line ~3729")
    print("   Change: if abs(signal) >= 0.30:")
    print("   To:     if abs(signal) >= 0.01:  # Log ALL signals for debugging")
    print()

    print("2. **Lower Signal Threshold** (Immediate)")
    print("   Add to .env:")
    print("   APEX_MIN_SIGNAL_THRESHOLD=0.12")
    print()

    print("3. **Fix IBKR Client ID Conflicts** (High Priority)")
    print("   Edit: services/broker/service.py lines 476, 739")
    print("   Change: client_id or 1")
    print("   To:     client_id or random.randint(100, 199)")
    print()

    print("4. **Increase Crypto Coverage** (Medium Priority)")
    print("   Check why only 10 crypto symbols are processing")
    print("   Expected: 35+ from Alpaca discovery")
    print()

    print("5. **Restart System** (After fixes)")
    print("   ./apex_ctl.sh restart")
    print()

    print("=" * 80)

if __name__ == "__main__":
    analyze_config()
    check_ml_models()
    suggest_fixes()
