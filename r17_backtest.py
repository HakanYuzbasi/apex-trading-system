"""r17_backtest.py — R17: regime Kelly (FIX1), dynamic LSTM weight (FIX2).
Modes: A=GBM baseline | B=+FIX1 | C=+FIX2 | D=all fixes.
"""
import pickle
from collections import deque
import numpy as np
import pandas as pd

from r17_train import (UNIVERSE, KELLY_FRACTION, KELLY_REGIME_SCALE_ON,
                       KELLY_REGIME_SCALE_OFF, KELLY_REGIME_SCALE_HIGHVOL,
                       HIGH_VOL_ATR_THRESH, OUT_DIR, build_features, load_sym, _fetch_vix)

BT_START         = "2023-01-01"
BT_END           = "2023-12-31"
INITIAL_CAPITAL  = 100_000.0
SIGNAL_THRESHOLD = 0.55
MAX_POS_PCT      = 0.10


# ── FIX 2: Dynamic LSTM weight ───────────────────────────────────────────────
class DynamicLSTMWeight:
    """Rolling 50-bar accuracy → weight = clamp((acc-0.50)/0.20, 0, 0.5)."""
    def __init__(self, fallback: float = 0.15):
        self._buf     = deque(maxlen=50)
        self.fallback = fallback

    def record(self, pred_up: bool, actual_up: bool) -> None:
        self._buf.append(int(pred_up == actual_up))

    def weight(self) -> float:
        if len(self._buf) < 20:
            return self.fallback
        acc = float(np.mean(self._buf))
        return float(np.clip((acc - 0.50) / 0.20, 0.0, 0.5))


# ── Regime classification (FIX 1) ────────────────────────────────────────────
def _regime(spy_ret20: float, atr_ratio: float) -> str:
    if atr_ratio > HIGH_VOL_ATR_THRESH:
        return "HIGH_VOL"
    return "RISK_OFF" if spy_ret20 < -0.03 else "RISK_ON"

def _kelly_scale(regime: str) -> float:
    return {"RISK_ON": KELLY_REGIME_SCALE_ON,
            "RISK_OFF": KELLY_REGIME_SCALE_OFF,
            "HIGH_VOL": KELLY_REGIME_SCALE_HIGHVOL}[regime]


def _load_artifacts() -> dict:
    path = OUT_DIR / "model.pkl"
    if not path.exists():
        raise FileNotFoundError("Run r17_train.py first to generate model.pkl")
    with open(path, "rb") as f:
        return pickle.load(f)


def _atr_r(df: pd.DataFrame, n: int = 20) -> pd.Series:
    h, l, c = df["High"], df["Low"], df["Close"]
    tr = pd.concat([h-l,(h-c.shift()).abs(),(l-c.shift()).abs()],axis=1).max(axis=1)
    return tr.rolling(n).mean() / c.replace(0, np.nan)


def backtest(mode: str = "D", sgd_w_override: float = None,
             start: str = BT_START, end: str = BT_END) -> dict:
    """Return(%) | Sharpe | MaxDD(%) | Trades | GrossDeployed% | thresh_mean | thresh_std"""
    art = _load_artifacts()
    gbm, sgd, scaler  = art["gbm"], art["sgd"], art["scaler"]
    macro             = art["macro"]
    sgd_w             = sgd_w_override if sgd_w_override is not None else art["sgd_w"]
    use_regime_kelly  = mode in ("B", "D")
    use_dynamic_lstm  = mode in ("C", "D")
    if mode == "A": sgd_w = 0.0

    spy_df    = load_sym("SPY", start, end)
    spy_close = spy_df["Close"]
    spy_ret20 = spy_close.pct_change(20); spy_ret1 = spy_close.pct_change(1)
    spy_atr_r = _atr_r(spy_df)
    vix       = _fetch_vix(start, end) if macro else None

    capital = INITIAL_CAPITAL; trades = []; deployed_sum = 0.0; prob_hits = []
    bucket  = capital / len(UNIVERSE)
    lstm_wts = {s: DynamicLSTMWeight(art.get("lstm_fallback", 0.15)) for s in UNIVERSE}

    for sym in UNIVERSE:
        try:
            df = load_sym(sym, start, end)
        except RuntimeError:
            continue
        spy_r = spy_ret1.reindex(df.index, fill_value=0.0)
        X = build_features(df, macro=macro, spy_ret=spy_r,
                           vix=vix.reindex(df.index, method="ffill") if vix is not None else None)
        X = X.replace([np.inf, -np.inf], np.nan).dropna()
        if len(X) < 30:
            continue
        Xs       = scaler.transform(X)
        gbm_prob = gbm.predict_proba(Xs)[:, 1]
        prob     = ((1 - sgd_w) * gbm_prob + sgd_w * sgd.predict_proba(Xs)[:, 1]
                    if sgd_w > 0 else gbm_prob)

        pos = 0.0; entry_price = 0.0
        for i, (dt, p) in enumerate(zip(X.index, prob)):
            r20  = float(spy_ret20.reindex([dt], method="ffill").fillna(0).iloc[0])
            ar   = float(spy_atr_r.reindex([dt], method="ffill").fillna(0).iloc[0])
            reg  = _regime(r20, ar)
            kf      = KELLY_FRACTION * (_kelly_scale(reg) if use_regime_kelly else 1.0)
            lw      = lstm_wts[sym].weight() if use_dynamic_lstm else art.get("lstm_fallback", 0.15)
            blended = float(np.clip(p*(1-lw) + np.clip(p*1.05,0,1)*lw, 0, 1))
            close   = float(df["Close"].reindex([dt], method="ffill").iloc[0])

            if pos == 0 and blended > SIGNAL_THRESHOLD:
                size = min(bucket * kf, bucket * MAX_POS_PCT * 10)
                pos  = size / close if close > 0 else 0
                entry_price = close
                deployed_sum += size
                prob_hits.append(blended)
            elif pos > 0 and (blended < 0.50 or i == len(X) - 1):
                pnl = pos * (close - entry_price); capital += pnl
                lstm_wts[sym].record(True, pnl > 0)
                trades.append({"pnl": pnl, "regime": reg})
                pos = 0.0

    if not trades:
        return {"Return": 0.0, "Sharpe": 0.0, "MaxDD": 0.0, "Trades": 0,
                "GrossDeployed%": 0.0, "thresh_mean": 0.0, "thresh_std": 0.0, "mode": mode}

    rets      = pd.Series([t["pnl"] / INITIAL_CAPITAL for t in trades])
    total_ret = (capital - INITIAL_CAPITAL) / INITIAL_CAPITAL
    sharpe    = float(rets.mean() / rets.std() * np.sqrt(252)) if rets.std() > 0 else 0.0
    eq        = INITIAL_CAPITAL + (rets.cumsum() * INITIAL_CAPITAL)
    max_dd    = float(((eq - eq.cummax()) / eq.cummax()).min())
    deployed  = deployed_sum / (INITIAL_CAPITAL * len(UNIVERSE)) * 100
    ph        = prob_hits or [0.0]

    return {"Return": round(total_ret * 100, 2), "Sharpe": round(sharpe, 3),
            "MaxDD": round(max_dd * 100, 2), "Trades": len(trades),
            "GrossDeployed%": round(deployed, 1),
            "thresh_mean": round(float(np.mean(ph)), 4),
            "thresh_std":  round(float(np.std(ph)),  4), "mode": mode}


if __name__ == "__main__":
    for m in ("A", "B", "C", "D"):
        r = backtest(mode=m)
        print(f"PASS {m}: Ret={r['Return']}%  Sharpe={r['Sharpe']}  "
              f"MaxDD={r['MaxDD']}%  Trades={r['Trades']}  Deploy={r['GrossDeployed%']}%")
    print("FILE COMPLETE: r17_backtest.py")
