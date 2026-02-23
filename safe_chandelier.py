import os
import re

path = "risk/god_level_risk_manager.py"
if not os.path.exists(path):
    print(f"⚠️ Could not find {path}")
    exit(1)

with open(path, "r", encoding="utf-8") as f:
    content = f.read()

# Dynamically find the exact indentation of the target method
match = re.search(r"([ \t]+)def calculate_position_size", content)
if not match:
    print("⚠️ Could not find calculate_position_size")
    exit(1)

indent = match.group(1)
inner_indent = indent + "    "

helper_method = f"""{indent}def _calculate_chandelier_stops(self, entry_price: float, atr: float, signal_strength: float, regime: str):
{inner_indent}\"\"\"Calculates Volatility-Adjusted Chandelier Exits.\"\"\"
{inner_indent}atr_mult = 2.5 if regime in ["volatile", "high_volatility", "bear"] else 3.0
{inner_indent}strength_adj = 1.0 + (abs(signal_strength) * 0.5)
{inner_indent}final_mult = atr_mult * strength_adj
{inner_indent}if signal_strength > 0:
{inner_indent}    stop_loss = entry_price - (atr * final_mult)
{inner_indent}    take_profit = entry_price + (atr * final_mult * 2.0)
{inner_indent}else:
{inner_indent}    stop_loss = entry_price + (atr * final_mult)
{inner_indent}    take_profit = entry_price - (atr * final_mult * 2.0)
{inner_indent}max_stop_pct = 0.15
{inner_indent}if signal_strength > 0:
{inner_indent}    stop_loss = max(stop_loss, entry_price * (1.0 - max_stop_pct))
{inner_indent}else:
{inner_indent}    stop_loss = min(stop_loss, entry_price * (1.0 + max_stop_pct))
{inner_indent}return round(stop_loss, 4), round(take_profit, 4)

{indent}def calculate_position_size"""

# Inject the helper method
content = content.replace(f"{indent}def calculate_position_size", helper_method)

# Isolate the method body to replace the return statement safely
start_idx = content.find(f"{indent}def calculate_position_size")
end_idx = content.find(f"{indent}def ", start_idx + len(f"{indent}def calculate_position_size"))
if end_idx == -1: 
    end_idx = len(content)

method_body = content[start_idx:end_idx]

# Regex to find the returned dictionary
return_pattern = re.compile(r"return\s*\{[^\}]*'target_shares':[^\}]*\}", re.DOTALL)

new_return = f"""sl, tp = self._calculate_chandelier_stops(entry_price, atr, signal_strength, regime)
{inner_indent}return {{
{inner_indent}    'target_shares': target_shares,
{inner_indent}    'stop_loss': sl,
{inner_indent}    'take_profit': tp,
{inner_indent}    'trailing_stop_pct': round((atr * 1.5) / entry_price, 4) if entry_price > 0 else 0.03,
{inner_indent}    'atr': atr
{inner_indent}}}"""

new_method_body = return_pattern.sub(new_return, method_body)
final_content = content[:start_idx] + new_method_body + content[end_idx:]

with open(path, "w", encoding="utf-8") as f:
    f.write(final_content)

print("✅ Chandelier Exits successfully injected with perfect indentation!")
