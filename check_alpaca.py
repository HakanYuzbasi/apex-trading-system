import requests, os, json
key = os.environ.get("APEX_ALPACA_API_KEY")
secret = os.environ.get("APEX_ALPACA_SECRET_KEY")
base = os.environ.get("APCA_API_BASE_URL", "https://paper-api.alpaca.markets")
headers = {"APCA-API-KEY-ID": key, "APCA-API-SECRET-KEY": secret}

print("=== ACCOUNT ===")
r = requests.get(f"{base}/v2/account", headers=headers)
if r.status_code == 200:
    d = r.json()
    print(f"Equity: ${float(d.get('equity', 0)):.2f}")
    print(f"Buying Power: ${float(d.get('buying_power', 0)):.2f}")
    print(f"Day Trade Count: {d.get('daytrade_count')}")
else:
    print("Failed:", r.text)

print("\n=== POSITIONS ===")
r = requests.get(f"{base}/v2/positions", headers=headers)
if r.status_code == 200:
    pos = r.json()
    if not pos:
        print("No open positions.")
    else:
        for p in pos:
            print(f"{p['symbol']:<10} {p['side']:>5} | Qty: {p['qty']:>8} | Market Value: ${float(p['market_value']):>8.2f} | Unrealized P&L: ${float(p['unrealized_pl']):>8.2f} ({float(p['unrealized_plpc'])*100:.2f}%)")
