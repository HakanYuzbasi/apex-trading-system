import yfinance as yf
import pandas as pd

def test_vix():
    ticker = yf.Ticker("^VIX")
    hist = ticker.history(period="5d", interval="1d")
    print(f"VIX History:\n{hist}")
    if hist.empty:
        print("VIX data is EMPTY")
    else:
        vix = float(hist["Close"].iloc[-1])
        print(f"Current VIX: {vix}")

if __name__ == "__main__":
    test_vix()
