import os
import re

def upgrade_atr_stops():
    path = "risk/god_level_risk_manager.py"
    if not os.path.exists(path):
        print(f"⚠️ Could not find {path}")
        return

    with open(path, "r", encoding="utf-8") as f:
        content = f.read()

    if "_calculate_chandelier_stops" in content:
        print("⚡ Chandelier Exits already applied!")
        return

    # 1. Inject the helper method right before calculate_position_size
    helper_method = """
    def _calculate_chandelier_stops(self, entry_price: float, atr: float, signal_strength: float, regime: str):
        \"\"\"
        Calculates Volatility-Adjusted Chandelier Exits.
        Allows massive runners to run while mathematically preventing noise-outs.
        \"\"\"
        atr_mult = 2.5 if regime in ["volatile", "high_volatility", "bear"] else 3.0
        strength_adj = 1.0 + (abs(signal_strength) * 0.5)
        final_mult = atr_mult * strength_adj

        if signal_strength > 0:  # LONG
            stop_loss = entry_price - (atr * final_mult)
            take_profit = entry_price + (atr * final_mult * 2.0)
        else:  # SHORT
            stop_loss = entry_price + (atr * final_mult)
            take_profit = entry_price - (atr * final_mult * 2.0)
            
        max_stop_pct = 0.15
        if signal_strength > 0:
            stop_loss = max(stop_loss, entry_price * (1.0 - max_stop_pct))
        else:
            stop_loss = min(stop_loss, entry_price * (1.0 + max_stop_pct))

        return round(stop_loss, 4), round(take_profit, 4)

    def calculate_position_size"""
    
    content = content.replace("def calculate_position_size", helper_method)

    # 2. Modify the return statement using regex to find the old static dict
    pattern = r"(return\s*\{\s*'target_shares':\s*target_shares,\s*'stop_loss':\s*.*?'atr':\s*atr\s*\})"
    
    match = re.search(pattern, content, re.DOTALL)
    if match:
        old_return = match.group(1)
        new_return = """sl, tp = self._calculate_chandelier_stops(entry_price, atr, signal_strength, regime)
        return {
            'target_shares': target_shares,
            'stop_loss': sl,
            'take_profit': tp,
            'trailing_stop_pct': round((atr * 1.5) / entry_price, 4) if entry_price > 0 else 0.03,
            'atr': atr
        }"""
        content = content.replace(old_return, new_return)
        
        with open(path, "w", encoding="utf-8") as f:
            f.write(content)
        print("✅ Upgraded GodLevelRiskManager to use Volatility-Adjusted Chandelier Exits!")
    else:
        # Fallback approach if regex fails due to custom formatting
        pattern_fallback = r"return\s*\{[^}]*'target_shares'[^}]*\}"
        match2 = re.search(pattern_fallback, content, re.DOTALL)
        if match2:
            new_return = """sl, tp = self._calculate_chandelier_stops(entry_price, atr, signal_strength, regime)
        return {
            'target_shares': target_shares,
            'stop_loss': sl,
            'take_profit': tp,
            'trailing_stop_pct': round((atr * 1.5) / entry_price, 4) if entry_price > 0 else 0.03,
            'atr': atr
        }"""
            content = content.replace(match2.group(0), new_return)
            with open(path, "w", encoding="utf-8") as f:
                f.write(content)
            print("✅ Upgraded GodLevelRiskManager to use Volatility-Adjusted Chandelier Exits! (Used fallback matcher)")
        else:
            print("⚠️ Could not locate the exact return block.")

if __name__ == "__main__":
    upgrade_atr_stops()
