"""r18_backtest.py — R18: VolCircuitBreaker (FIX 1) + 2024-2025 OOS test.
Raises RuntimeError if yfinance fails.
"""
import json
from collections import defaultdict
import numpy as np
import pandas as pd

from r17_train import UNIVERSE, build_features, load_sym
from r18_train import (OUT_DIR, VOL_CB_ENABLED, VOL_CB_RATIO_SCALE,
                        VOL_CB_RATIO_HALT, VOL_CB_RATIO_CLOSE,
                        SIGNAL_THRESHOLD, INITIAL_CAPITAL, KELLY_FRACTION,
                        OOS_START, OOS_END, load_model)

CB_LOG = OUT_DIR / "vol_circuit_breaker_log.jsonl"


class VolCircuitBreaker:
    """Rolling ATR(5)/ATR(60) position-size gating.

    Thresholds → multiplier, mode:
      > VOL_CB_RATIO_CLOSE (4.0): 0.0, 'close'  — close all
      > VOL_CB_RATIO_HALT  (3.0): 0.0, 'halt'   — no new entries
      > VOL_CB_RATIO_SCALE (2.0): 0.25, 'scale'
      else                       : 1.0, 'normal'
    """
    _MULTS = {"normal": 1.0, "scale": 0.25, "halt": 0.0, "close": 0.0}

    def __init__(self, enabled: bool = VOL_CB_ENABLED,
                 ratio_scale: float = VOL_CB_RATIO_SCALE,
                 ratio_halt:  float = VOL_CB_RATIO_HALT,
                 ratio_close: float = VOL_CB_RATIO_CLOSE):
        self.enabled = enabled
        self.rs, self.rh, self.rc = ratio_scale, ratio_halt, ratio_close
        self.log: list[dict] = []
        self._prev = "normal"

    def evaluate(self, atr5: float, atr60: float, dt=None) -> tuple[float, str]:
        if not self.enabled or atr60 <= 0:
            return 1.0, "normal"
        r = atr5 / atr60
        mode = ("close" if r > self.rc else "halt" if r > self.rh
                else "scale" if r > self.rs else "normal")
        if mode != self._prev:
            self.log.append({"dt": str(dt), "from": self._prev, "to": mode,
                              "ratio": round(r, 3)})
            self._prev = mode
        return self._MULTS[mode], mode


def _atr_n(df: pd.DataFrame, n: int) -> pd.Series:
    h, l, c = df["High"], df["Low"], df["Close"]
    tr = pd.concat([h-l, (h-c.shift()).abs(), (l-c.shift()).abs()], axis=1).max(axis=1)
    return tr.rolling(n).mean()


def backtest(cb_enabled: bool = True,
             start: str = OOS_START, end: str = OOS_END) -> dict:
    """Score pre-trained model on [start,end] with CB on/off.
    Returns: Return% | Sharpe | MaxDD% | Trades | monthly {ym: pnl}
    """
    art   = load_model()
    gbm, scaler = art["gbm"], art["scaler"]

    spy_df  = load_sym("SPY", start, end)
    spy_ret = spy_df["Close"].pct_change(1)

    capital = INITIAL_CAPITAL
    bucket  = capital / len(UNIVERSE)
    trades: list[float] = []
    monthly: dict = defaultdict(float)
    cb_global = VolCircuitBreaker(enabled=cb_enabled)

    for sym in UNIVERSE:
        try:
            df = load_sym(sym, start, end)
        except RuntimeError:
            continue
        X = build_features(df, macro=False,
                           spy_ret=spy_ret.reindex(df.index, fill_value=0.0))
        X = X.replace([np.inf, -np.inf], np.nan).dropna()
        if len(X) < 30:
            continue
        prob  = gbm.predict_proba(scaler.transform(X))[:, 1]
        atr5  = _atr_n(df, 5)
        atr60 = _atr_n(df, 60)
        pos   = 0.0; entry_px = 0.0

        for i, (dt, p) in enumerate(zip(X.index, prob)):
            a5  = float(atr5.reindex([dt], method="ffill").fillna(0).iloc[0])
            a60 = float(atr60.reindex([dt], method="ffill").fillna(1).iloc[0])
            mult, mode = cb_global.evaluate(a5, a60, dt)
            close = float(df["Close"].reindex([dt], method="ffill").iloc[0])
            ym    = str(dt)[:7]

            if pos > 0 and mode == "close":          # FIX 1: force close
                pnl = pos * (close - entry_px); capital += pnl
                monthly[ym] += pnl; trades.append(pnl); pos = 0.0; continue
            if pos == 0 and mode not in ("halt", "close") and p > SIGNAL_THRESHOLD:
                size = bucket * KELLY_FRACTION * mult
                pos  = size / close if close > 0 else 0
                entry_px = close
            elif pos > 0 and (p < 0.50 or i == len(X) - 1):
                pnl = pos * (close - entry_px); capital += pnl
                monthly[ym] += pnl; trades.append(pnl); pos = 0.0

    # Persist CB log for forward test
    OUT_DIR.mkdir(exist_ok=True)
    with open(CB_LOG, "a") as f:
        for e in cb_global.log:
            f.write(json.dumps(e) + "\n")

    rets  = pd.Series([t / INITIAL_CAPITAL for t in trades])
    total = (capital - INITIAL_CAPITAL) / INITIAL_CAPITAL
    sh    = float(rets.mean() / rets.std() * np.sqrt(252)) if len(rets)>1 and rets.std()>0 else 0.0
    eq    = INITIAL_CAPITAL + rets.cumsum() * INITIAL_CAPITAL
    dd    = float(((eq - eq.cummax()) / eq.cummax()).min()) if len(eq) > 0 else 0.0
    return {"Return": round(total*100,2), "Sharpe": round(sh,3),
            "MaxDD": round(dd*100,2), "Trades": len(trades),
            "monthly": {k: round(v,2) for k,v in sorted(monthly.items())}}


if __name__ == "__main__":
    from r18_train import train_model
    train_model()
    res = backtest(cb_enabled=True, start=OOS_START, end=OOS_END)
    print(f"2024-2025 OOS (CB=ON): Ret={res['Return']}%  Sharpe={res['Sharpe']}  "
          f"MaxDD={res['MaxDD']}%  Trades={res['Trades']}")
    for ym, pnl in res["monthly"].items():
        print(f"  {ym}: {pnl:+.1f}")
    print("FILE COMPLETE: r18_backtest.py")
