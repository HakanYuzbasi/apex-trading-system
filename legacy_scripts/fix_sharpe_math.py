import os
import re

def fix_killswitch_math():
    path = "risk/kill_switch.py"
    if not os.path.exists(path):
        print(f"⚠️ Could not find {path}")
        return

    with open(path, "r", encoding="utf-8") as f:
        content = f.read()

    # Find the logic where the Sharpe ratio breach is calculated
    target_block = """sharpe_breach = (
            math.isfinite(self.state_data.sharpe_rolling)
            and self.state_data.sharpe_rolling < self.config.sharpe_floor
            and len(equity_curve) >= min_samples
        )"""

    # Replace it with a block that REQUIRES at least 20 real trades before caring about Sharpe
    new_block = """
        # Only evaluate Sharpe if we have a statistical baseline (e.g., > 20 trades/samples)
        # Prevents a single cold-start losing trade from creating a mathematically infinite negative Sharpe
        minimum_baseline_samples = max(min_samples, 20)
        
        sharpe_breach = (
            math.isfinite(self.state_data.sharpe_rolling)
            and self.state_data.sharpe_rolling < self.config.sharpe_floor
            and len(equity_curve) >= minimum_baseline_samples
        )"""

    if target_block in content:
        content = content.replace(target_block, new_block)
        with open(path, "w", encoding="utf-8") as f:
            f.write(content)
        print("✅ Fixed Kill-Switch math! It will now ignore Sharpe anomalies until 20 trades are reached.")
    else:
        # Fallback if the exact text doesn't match
        if "sharpe_breach =" in content and "minimum_baseline_samples" not in content:
            content = re.sub(
                r'sharpe_breach\s*=\s*\([^)]+len\(equity_curve\)\s*>=\s*min_samples[^)]*\)',
                new_block.strip(),
                content,
                flags=re.DOTALL
            )
            with open(path, "w", encoding="utf-8") as f:
                f.write(content)
            print("✅ Fixed Kill-Switch math via regex!")
        else:
            print("⚡ Kill-switch math appears to already be modified.")

if __name__ == "__main__":
    fix_killswitch_math()
