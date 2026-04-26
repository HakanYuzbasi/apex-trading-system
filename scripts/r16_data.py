"""Round 16 data helper: patches yfinance with GBM synthetic fallback for offline envs."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

_PATCHED = False


def _synth(sym: str, start: str, end: str) -> pd.DataFrame:
    dates = pd.bdate_range(start, end)
    n = len(dates)
    if n == 0:
        return pd.DataFrame()
    rng = np.random.default_rng(abs(hash(sym)) % (2 ** 31))
    # Regime-aware GBM: trending first half, mean-reverting second half
    half = n // 2
    log_ret = np.concatenate([
        rng.normal(8e-5, 0.012, half),    # mild uptrend
        rng.normal(-2e-5, 0.008, n - half),  # low-vol mean reversion
    ])
    close = 100.0 * np.exp(np.cumsum(log_ret))
    noise = lambda s: np.abs(rng.normal(0, s, n))
    high   = close * (1 + noise(0.007))
    low    = np.maximum(close * (1 - noise(0.007)), close * 0.97)
    open_  = np.concatenate([[close[0]], close[:-1]]) * (1 + rng.normal(0, 0.003, n))
    volume = rng.integers(800_000, 6_000_000, n).astype(float)
    df = pd.DataFrame(
        {"Open": open_, "High": high, "Low": low, "Close": close, "Volume": volume},
        index=pd.to_datetime(dates).tz_localize(None).normalize(),
    )
    return df


_orig_history = yf.Ticker.history


def _patched_history(self, *, start, end, interval="1d", **kw):
    try:
        df = _orig_history(self, start=start, end=end, interval=interval, **kw)
        if df is not None and not df.empty:
            return df
    except Exception:
        pass
    return _synth(getattr(self, "ticker", "UNK"), str(start), str(end))


def apply_synth_fallback() -> None:
    global _PATCHED
    if not _PATCHED:
        yf.Ticker.history = _patched_history
        _PATCHED = True
