import os
import re

def patch_enhanced_filter():
    path = "models/enhanced_signal_filter.py"
    if not os.path.exists(path): return
    with open(path, "r", encoding="utf-8") as f: content = f.read()
    
    # Relax the net-edge cost filter by 50%
    if "abs(expected_return) < est_cost_pct" in content:
        content = content.replace(
            "abs(expected_return) < est_cost_pct",
            "abs(expected_return) < (est_cost_pct * 0.5)"
        )
        with open(path, "w", encoding="utf-8") as f: f.write(content)
        print("✅ Relaxed Net-Edge Filter in enhanced_signal_filter.py")

def patch_trading_excellence():
    path = "risk/trading_excellence.py"
    if not os.path.exists(path): return
    with open(path, "r", encoding="utf-8") as f: content = f.read()
    
    # Change exit threshold so it requires a negative signal, not just a weak one
    if "MODERATE_MISMATCH_THRESHOLD = 0.05" in content:
        content = content.replace(
            "MODERATE_MISMATCH_THRESHOLD = 0.05",
            "MODERATE_MISMATCH_THRESHOLD = -0.05"
        )
        with open(path, "w", encoding="utf-8") as f: f.write(content)
        print("✅ Fixed premature exits in trading_excellence.py")

if __name__ == "__main__":
    print("🔓 Unlocking final execution gatekeepers...")
    patch_enhanced_filter()
    patch_trading_excellence()
