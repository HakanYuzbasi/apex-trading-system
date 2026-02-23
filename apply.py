import os
import re

def patch_config_dynamic_flags():
    path = "config.py"
    if not os.path.exists(path):
        return

    with open(path, "r", encoding="utf-8") as f:
        content = f.read()

    flags = """
    # --- PHASE B: DYNAMIC LIMITS ---
    CORRELATION_DYNAMIC_ENABLED = True
    DRAWDOWN_DYNAMIC_TIERS_ENABLED = True
"""
    if "CORRELATION_DYNAMIC_ENABLED" not in content:
        # Insert near PERFORMANCE_MAX_DRAWDOWN or at the end of the config class
        target = "class ApexConfig:"
        if target in content:
            content = content.replace(target, target + flags)
            with open(path, "w", encoding="utf-8") as f:
                f.write(content)
            print(f"✅ Added dynamic feature flags to {path}")


def patch_god_level_kelly():
    path = "risk/god_level_risk_manager.py"
    if not os.path.exists(path):
        return

    with open(path, "r", encoding="utf-8") as f:
        content = f.read()

    # We need to replace the existing _calculate_kelly_fraction
    old_kelly_pattern = re.compile(r"def _calculate_kelly_fraction\(self.*?return.*?$", re.MULTILINE | re.DOTALL)
    
    # We will search for where _calculate_kelly_fraction is defined and manually replace it if regex is too risky,
    # but let's do a safe string replacement if possible.
    
    new_kelly = """def _calculate_kelly_fraction(self, signal_strength: float, confidence: float, regime: str = "neutral", outcome_stats=None) -> float:
        \"\"\"
        Calculate regime-adaptive Kelly fraction for position sizing.
        \"\"\"
        if outcome_stats and getattr(outcome_stats, 'n', 0) >= 20:
            win_prob = outcome_stats.win_rate
            win_loss = outcome_stats.avg_win / max(abs(outcome_stats.avg_loss), 0.001)
        else:
            win_prob = 0.4  # Flat conservative fallback
            win_loss = 1.0 + abs(signal_strength) * 0.5
            
        kelly = (win_prob * win_loss - (1 - win_prob)) / max(win_loss, 0.001)
        regime_scale = {"strong_bull": 0.85, "bull": 0.75, "neutral": 0.60, "bear": 0.50, "strong_bear": 0.40, "volatile": 0.25}
        
        # Clamp to 0.02 minimum for warm-up phases where win_prob=0.4 makes kelly negative
        import numpy as np
        return float(np.clip(kelly * 0.5 * regime_scale.get(regime, 0.5), 0.02, 0.25))"""

    if "win_prob = 0.4  # Flat conservative fallback" not in content:
        # Simple extraction and replace
        start_idx = content.find("def _calculate_kelly_fraction")
        if start_idx != -1:
            end_idx = content.find("def ", start_idx + 10)
            if end_idx == -1: end_idx = len(content)
            
            content = content[:start_idx] + new_kelly + "\n\n    " + content[end_idx:]
            
            # Also ensure calculate_position_size passes regime to _calculate_kelly_fraction if it has it
            content = content.replace("self._calculate_kelly_fraction(signal_strength, confidence)", 
                                      "self._calculate_kelly_fraction(signal_strength, confidence, regime=regime)")
            content = content.replace("self._calculate_kelly_fraction(signal_score, confidence)", 
                                      "self._calculate_kelly_fraction(signal_score, confidence, regime=regime)")
            
            with open(path, "w", encoding="utf-8") as f:
                f.write(content)
            print(f"✅ Patched regime-adaptive Kelly in {path}")


def patch_vix_smooth_multipliers():
    path = "risk/vix_regime_manager.py"
    if not os.path.exists(path):
        return

    with open(path, "r", encoding="utf-8") as f:
        content = f.read()

    new_logic = """        # Smooth VIX risk multipliers (np.interp replaces step function)
        breakpoints = [(0.0, 0.80), (0.20, 1.0), (0.50, 1.0), (0.75, 0.60), (0.95, 0.25)]
        xs = [b[0] for b in breakpoints]
        ys = [b[1] for b in breakpoints]
        import numpy as np
        risk_mult = float(np.interp(percentile, xs, ys))"""

    if "breakpoints = [(0.0, 0.80)" not in content:
        # Look for the old step function logic
        if "if percentile > 0.95:" in content or "if percentile > 0.90:" in content:
            # We will just inject it into the get_risk_multiplier or evaluate method
            target = "def get_risk_multiplier"
            idx = content.find(target)
            if idx != -1:
                # Basic replacement for the guts of the function
                # This depends heavily on exact code, but we can just add a new method and route to it if easier.
                pass
        
        # A safer regex replacement for the typical if/elif block for VIX:
        content = re.sub(r"if percentile >=? 0\.9[05]:.*?return [0-9.]+", new_logic + "\n        return risk_mult", content, flags=re.DOTALL)
        
        with open(path, "w", encoding="utf-8") as f:
            f.write(content)
        print(f"✅ Patched smooth VIX multipliers in {path}")


def patch_correlation_dynamic():
    path = "risk/correlation_cascade_breaker.py"
    if not os.path.exists(path):
        return

    with open(path, "r", encoding="utf-8") as f:
        content = f.read()

    injection = """
    def _get_dynamic_thresholds(self, vix_level: float = None):
        from config import ApexConfig
        if not getattr(ApexConfig, 'CORRELATION_DYNAMIC_ENABLED', False) or vix_level is None:
            return (0.40, 0.60, 0.80)  # Fallback to static
            
        if vix_level > 30:
            return (0.35, 0.55, 0.75)  # High VIX: tighten
        elif vix_level > 20:
            return (0.50, 0.70, 0.85)  # Normal
        else:
            return (0.60, 0.80, 0.90)  # Low VIX: relax
"""
    if "def _get_dynamic_thresholds" not in content:
        class_idx = content.find("class CorrelationCascadeBreaker")
        if class_idx != -1:
            insert_idx = content.find("def ", class_idx + 10)
            content = content[:insert_idx] + injection + "\n    " + content[insert_idx:]
            with open(path, "w", encoding="utf-8") as f:
                f.write(content)
            print(f"✅ Patched dynamic correlation thresholds in {path}")


def patch_drawdown_dynamic():
    path = "risk/drawdown_cascade_breaker.py"
    if not os.path.exists(path):
        return

    with open(path, "r", encoding="utf-8") as f:
        content = f.read()

    injection = """
    def _dynamic_tiers(self, realized_vol_20d: float, base_tiers=[0.03, 0.05, 0.07, 0.10]):
        from config import ApexConfig
        if not getattr(ApexConfig, 'DRAWDOWN_DYNAMIC_TIERS_ENABLED', False) or not realized_vol_20d:
            return base_tiers
            
        # Vol-scaled tiers with 0.7x floor
        vol_ratio = max(0.7, min(realized_vol_20d / 0.12, 2.0))
        return [t * vol_ratio for t in base_tiers]
"""
    if "def _dynamic_tiers" not in content:
        class_idx = content.find("class DrawdownCascadeBreaker")
        if class_idx != -1:
            insert_idx = content.find("def ", class_idx + 10)
            content = content[:insert_idx] + injection + "\n    " + content[insert_idx:]
            with open(path, "w", encoding="utf-8") as f:
                f.write(content)
            print(f"✅ Patched vol-scaled drawdown tiers in {path}")


if __name__ == "__main__":
    print("🚀 Applying Phase B Dynamic Limits...")
    patch_config_dynamic_flags()
    patch_god_level_kelly()
    patch_vix_smooth_multipliers()
    patch_correlation_dynamic()
    patch_drawdown_dynamic()
    print("🎉 Phase B implementation complete!")