#!/usr/bin/env python3
"""
Remove Dead Code from Apex Trading System
Based on comprehensive audit findings
"""
import os
import shutil
from pathlib import Path

# Track what we remove
removed = {"files": [], "dirs": [], "errors": []}

def safe_remove_file(path: str):
    """Remove file if exists"""
    p = Path(path)
    if p.exists():
        try:
            p.unlink()
            removed["files"].append(str(p))
            print(f"✓ Removed file: {p}")
        except Exception as e:
            removed["errors"].append(f"Failed to remove {p}: {e}")
            print(f"✗ Failed to remove {p}: {e}")

def safe_remove_dir(path: str):
    """Remove directory if exists"""
    p = Path(path)
    if p.exists() and p.is_dir():
        try:
            shutil.rmtree(p)
            removed["dirs"].append(str(p))
            print(f"✓ Removed dir: {p}")
        except Exception as e:
            removed["errors"].append(f"Failed to remove {p}: {e}")
            print(f"✗ Failed to remove {p}: {e}")

print("=" * 80)
print("🧹 APEX DEAD CODE CLEANUP")
print("=" * 80)
print()

# 1. UNUSED MODEL FILES
print("📊 Removing unused model files...")
unused_models = [
    "models/ensemble_signal_generator.py",
    "models/pairs_trader.py",
    "models/rl_environment.py",
    "models/signal_optimizer.py",
    "models/online_learner.py",
    "models/hyperparameter_tuner.py",
    "models/meta_labeler.py",
    "models/feature_store.py",
]
for f in unused_models:
    safe_remove_file(f)
print()

# 2. UNUSED EXECUTION MODULES
print("⚙️  Removing unused execution modules...")
unused_execution = [
    "execution/adaptive_twap.py",
    "execution/transaction_cost_optimizer.py",
    "execution/ibkr_adapter.py",
    "execution/ibkr_lease_manager.py",
    "execution/execution_analytics.py",
]
for f in unused_execution:
    safe_remove_file(f)
print()

# 3. UNUSED MONITORING MODULES
print("📈 Removing unused monitoring modules...")
unused_monitoring = [
    "monitoring/alert_manager.py",
    "monitoring/business_metrics.py",
    "monitoring/compliance_manager.py",
    "monitoring/dashboard.py",
    "monitoring/advanced_dashboard.py",
    "monitoring/feature_drift_detector.py",
    "monitoring/realtime_monitor.py",
    "monitoring/api_monitor.py",
    "monitoring/model_monitor.py",
    "monitoring/trade_journal.py",
    "monitoring/slack_notifier.py",
]
for f in unused_monitoring:
    safe_remove_file(f)
print()

# 4. UNUSED SAAS SERVICES (test-only)
print("☁️  Removing unused SaaS services...")
unused_services = [
    "services/social_trading/",
    "services/strategy_marketplace/",
    "services/mandate_copilot/",
    "services/compliance_copilot/",
]
for d in unused_services:
    safe_remove_dir(d)
print()

# 5. DUPLICATE/ORPHANED MODULES
print("🔄 Removing duplicate modules...")
duplicates = [
    "data/metrics_store.py",  # Duplicate of monitoring/metrics_collector.py
    "risk/drawdown_cascade_breaker.py",  # Overlaps with kill_switch.py
]
for f in duplicates:
    safe_remove_file(f)
print()

# 6. CLEANUP OLD SCRIPTS
print("🗑️  Removing old/emergency scripts...")
old_scripts = [
    "emergency_fix.sh",
    "fix_trading_issues.py",
]
for f in old_scripts:
    safe_remove_file(f)
print()

# Summary
print("=" * 80)
print("📋 CLEANUP SUMMARY")
print("=" * 80)
print(f"Files removed:       {len(removed['files'])}")
print(f"Directories removed: {len(removed['dirs'])}")
print(f"Errors:              {len(removed['errors'])}")
print()

if removed['errors']:
    print("⚠️  ERRORS:")
    for err in removed['errors']:
        print(f"  - {err}")
    print()

print("✅ Cleanup complete!")
print()
print("Next steps:")
print("1. Restart system: ./apex_ctl.sh restart")
print("2. Verify no import errors: python -c 'import main'")
print("3. Monitor logs: tail -f /private/tmp/apex_main.log")
