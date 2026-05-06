import os, json, math, warnings, time, sys
warnings.filterwarnings("ignore", category=Warning, module="requests")
import requests

def fetch_and_display():
    key = os.environ.get("APEX_ALPACA_API_KEY")
    secret = os.environ.get("APEX_ALPACA_SECRET_KEY")
    base = os.environ.get("APCA_API_BASE_URL", "https://paper-api.alpaca.markets")
    
    # Allow fallback to read from .env if running on host
    if not key or not secret:
        try:
            with open(".env", "r") as f:
                for line in f:
                    if line.startswith("APEX_ALPACA_API_KEY="):
                        key = line.strip().split("=")[1].strip().strip("\"'")
                    elif line.startswith("APEX_ALPACA_SECRET_KEY="):
                        secret = line.strip().split("=")[1].strip().strip("\"'")
        except:
            pass

    if not key or not secret:
        print("Error: Missing APEX_ALPACA_API_KEY or APEX_ALPACA_SECRET_KEY.")
        sys.exit(1)

    headers = {"APCA-API-KEY-ID": key, "APCA-API-SECRET-KEY": secret}

    # Clear the screen using ANSI escape sequence
    print("\033[H\033[J", end="")
    print("="*70)
    print("🚀 APEX LIVE PERFORMANCE & RISK DASHBOARD 🚀")
    print("="*70)

    # 1. Get Account
    r = requests.get(f"{base}/v2/account", headers=headers)
    if r.status_code == 200:
        d = r.json()
        equity = float(d.get('equity', 0))
        print("\n=== ACCOUNT ===")
        print(f"Equity       : ${equity:,.2f}")
        print(f"Buying Power : ${float(d.get('buying_power', 0)):,.2f}")
        print(f"Day Trades   : {d.get('daytrade_count')}")
    else:
        print("Failed to get account:", r.text)
        return
        
    # 2. Get Portfolio History for Sharpe/Drawdown
    print("\n=== METRICS (Rolling 14 Days) ===")
    r = requests.get(f"{base}/v2/account/portfolio/history?period=2W&timeframe=15Min", headers=headers)
    if r.status_code == 200:
        hist = r.json()
        equities = hist.get('equity', [])
        equities = [e for e in equities if e is not None]
        
        if len(equities) > 1:
            returns = []
            for i in range(1, len(equities)):
                if equities[i-1] > 0:
                    returns.append((equities[i] - equities[i-1]) / equities[i-1])
                else:
                    returns.append(0)
                    
            if len(returns) > 0:
                import statistics
                mean_ret = statistics.mean(returns)
                std_ret = statistics.stdev(returns) if len(returns) > 1 else 0
                
                annual_factor = math.sqrt(6552)
                sharpe = (mean_ret / std_ret * annual_factor) if std_ret > 0 else 0
                
                peak = equities[0]
                max_dd = 0
                for eq in equities:
                    if eq > peak:
                        peak = eq
                    dd = (peak - eq) / peak
                    if dd > max_dd:
                        max_dd = dd
                        
                downside_returns = [r for r in returns if r < 0]
                downside_std = statistics.stdev(downside_returns) if len(downside_returns) > 1 else 0
                sortino = (mean_ret / downside_std * annual_factor) if downside_std > 0 else 0
                
                print(f"Sharpe Ratio : {sharpe:.2f}")
                print(f"Sortino Ratio: {sortino:.2f}")
                print(f"Max Drawdown : {max_dd*100:.2f}%")
                print(f"Total Return : {((equities[-1] - equities[0])/equities[0])*100:.2f}%")
                
                tier = "🟢 GREEN (Healthy)"
                if max_dd >= 0.08 or sharpe < -0.25:
                    tier = "🔴 RED (Halted)"
                elif max_dd >= 0.064 or sharpe < 0.60:
                    tier = "🟠 ORANGE (Reduced)"
                elif max_dd >= 0.048 or sharpe < 1.12:
                    tier = "🟡 YELLOW (Caution)"
                    
                print(f"Current Tier : {tier}")
            else:
                print("Not enough return data.")
        else:
            print("Not enough equity history.")
    else:
        print("Failed to get portfolio history:", r.text)

    # 3. Get Positions
    print("\n=== OPEN POSITIONS ===")
    r = requests.get(f"{base}/v2/positions", headers=headers)
    if r.status_code == 200:
        pos = r.json()
        if not pos:
            print("No open positions.")
        else:
            print(f"{'SYMBOL':<10} | {'SIDE':<5} | {'QTY':>12} | {'VALUE':>12} | {'UNREALIZED P&L':>18}")
            print("-" * 70)
            for p in pos:
                pl_usd = float(p['unrealized_pl'])
                pl_pct = float(p['unrealized_plpc']) * 100
                pl_str = f"${pl_usd:.2f} ({pl_pct:+.2f}%)"
                val = f"${float(p['market_value']):.2f}"
                qty = f"{float(p['qty']):.6f}".rstrip('0').rstrip('.')
                print(f"{p['symbol']:<10} | {p['side']:<5} | {qty:>12} | {val:>12} | {pl_str:>18}")
            
            total_long = sum(float(p['market_value']) for p in pos if p['side'] == 'long')
            total_short = sum(float(p['market_value']) for p in pos if p['side'] == 'short')
            net = total_long - total_short
            print("-" * 70)
            print(f"Net Exposure: ${net:,.2f}  (Long: ${total_long:,.2f} / Short: ${total_short:,.2f})")

def main():
    try:
        while True:
            fetch_and_display()
            print("\nRefreshing in 5 minutes... (Press Ctrl+C to exit)")
            time.sleep(300)
    except KeyboardInterrupt:
        print("\nExited Dashboard.")
        sys.exit(0)

if __name__ == "__main__":
    main()
