import os

path = "risk/god_level_risk_manager.py"
if not os.path.exists(path):
    print(f"⚠️ Could not find {path}")
    exit(1)

with open(path, "r", encoding="utf-8") as f:
    content = f.read()

# 1. Inject the Helper Method if not present
if "def _calculate_chandelier_stops" not in content:
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
    
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)

# 2. Precisely overwrite the old return statement
with open(path, "r", encoding="utf-8") as f:
    content = f.read()

start_idx = content.find("def calculate_position_size")
if start_idx != -1:
    # Find where the next method starts so we only modify calculate_position_size
    end_idx = content.find("def ", start_idx + 10)
    if end_idx == -1:
        end_idx = len(content)
        
    method_body = content[start_idx:end_idx]
    
    # Isolate the final return statement and chop it off
    return_start = method_body.rfind("return")
    if return_start != -1:
        new_return = """sl, tp = self._calculate_chandelier_stops(entry_price, atr, signal_strength, regime)
        return {
            'target_shares': target_shares,
            'stop_loss': sl,
            'take_profit': tp,
            'trailing_stop_pct': round((atr * 1.5) / entry_price, 4) if entry_price > 0 else 0.03,
            'atr': atr
        }
"""
        new_method = method_body[:return_start] + new_return
        final_content = content[:start_idx] + new_method + content[end_idx:]
        
        with open(path, "w", encoding="utf-8") as f:
            f.write(final_content)
        print("✅ Chandelier Exits successfully forced into GodLevelRiskManager!")
    else:
        print("⚠️ Could not find 'return' in the method.")
