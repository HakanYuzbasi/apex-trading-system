"""
scripts/train_baseline_models.py

Round 10 — Regime-specialised baseline ML training pipeline.

Three gradient-boosting classifiers are trained from the same feature
matrix (AAPL + MSFT + GOOGL OHLCV, 2018-01-01 → 2022-12-31) but on
three disjoint regime subsets so the
:class:`backtesting.real_signal_adapter.RealSignalAdapter` can route
each production bar to the model most relevant for that market regime:

* ``TRENDING``  : rows where ADX(14) > ``REGIME_ADX_TRENDING_MIN`` (default 25).
* ``MEAN_REV``  : rows where ADX(14) < ``REGIME_ADX_MEAN_REV_MAX`` (default 20)
                  AND ATR(14) / close < ``REGIME_VOL_MAX`` (default 0.015).
* ``VOLATILE``  : rows where ATR(14) / close >= ``REGIME_VOL_MAX``.

Any regime subset with fewer than
:data:`MIN_REGIME_SAMPLES` (default 200) rows falls back to the
full-dataset model and a WARNING is emitted so the operator can inspect
the regime balance.

For each regime the pipeline:

1. Computes ADX(14) and ATR(14) on the per-symbol OHLCV.
2. Joins the pre-existing 8-feature matrix
   (``backtesting.real_signal_adapter.compute_ml_features``) with the
   label (``sign(forward 5-bar log-return)``) and the regime tag.
3. Sorts strictly by timestamp (no random shuffle).
4. Runs :func:`models.ml_validator.leakage_check` with
   ``raise_on_fail=True`` on the regime subset **before** ``fit()``.
5. Trains
   :class:`sklearn.ensemble.GradientBoostingClassifier` with a fixed
   seed and an 80 % / 20 % time-ordered train/test split.
6. Saves ``{"model", "baseline_stats", "feature_names",
   "training_metadata"}`` to the corresponding
   ``ApexConfig.ML_MODEL_PATH_*`` target (or the per-regime default
   path under ``models/saved_advanced/``).

All regime-boundary thresholds are tunable via the
``APEX_REGIME_*`` environment variables so the Round 10 regime
definitions are not hard-coded magic numbers.
"""
from __future__ import annotations

import hashlib
import logging
import os
import pickle
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, log_loss

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from backtesting.real_signal_adapter import ML_FEATURE_NAMES, compute_ml_features
from config import ApexConfig
from core.logging_config import setup_logging
from models.ml_validator import leakage_check

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Configuration constants — overridable via env.
# ─────────────────────────────────────────────────────────────────────────────

SYMBOLS: Tuple[str, ...] = (
    "AAPL", "MSFT", "GOOGL", "SPY", "QQQ", "NVDA", "AMZN",
    "GLD", "TLT", "IWM",
)
TRAIN_START: str = "2018-01-01"
TRAIN_END: str = "2023-01-01"  # yfinance end is exclusive
FORWARD_HORIZON: int = 5
RANDOM_STATE: int = 42
TRAIN_FRACTION: float = 0.80
ADX_PERIOD: int = int(os.getenv("APEX_REGIME_ADX_PERIOD", "14"))
ATR_PERIOD: int = int(os.getenv("APEX_REGIME_ATR_PERIOD", "14"))
ADX_TRENDING_MIN: float = float(os.getenv("APEX_REGIME_ADX_TRENDING_MIN", "25.0"))
ADX_MEAN_REV_MAX: float = float(os.getenv("APEX_REGIME_ADX_MEAN_REV_MAX", "20.0"))
VOL_MAX: float = float(os.getenv("APEX_REGIME_VOL_MAX", "0.015"))
MIN_REGIME_SAMPLES: int = int(os.getenv("APEX_MIN_REGIME_SAMPLES", "200"))

REGIME_TRENDING: str = "TRENDING"
REGIME_MEAN_REV: str = "MEAN_REV"
REGIME_VOLATILE: str = "VOLATILE"

REGIME_TO_PATH_ATTR: Dict[str, str] = {
    REGIME_TRENDING: "ML_MODEL_PATH_TRENDING",
    REGIME_MEAN_REV: "ML_MODEL_PATH_MEAN_REV",
    REGIME_VOLATILE: "ML_MODEL_PATH_VOLATILE",
}
REGIME_TO_DEFAULT_BASENAME: Dict[str, str] = {
    REGIME_TRENDING: "baseline_trending.pkl",
    REGIME_MEAN_REV: "baseline_mean_rev.pkl",
    REGIME_VOLATILE: "baseline_volatile.pkl",
}


# ─────────────────────────────────────────────────────────────────────────────
# Data ingest + feature engineering
# ─────────────────────────────────────────────────────────────────────────────

def fetch_yf_ohlcv(symbols: Tuple[str, ...], start: str, end: str) -> Dict[str, pd.DataFrame]:
    """
    Download daily OHLCV bars via yfinance.

    Args:
        symbols: Tickers.
        start: Inclusive start date (YYYY-MM-DD).
        end: Exclusive end date (YYYY-MM-DD).

    Returns:
        ``{symbol: DataFrame}`` with columns ``Open``/``High``/``Low``/``Close``/``Volume``.

    Raises:
        RuntimeError: If every symbol returns zero bars.
    """
    panel: Dict[str, pd.DataFrame] = {}
    for sym in symbols:
        df = yf.Ticker(sym).history(
            start=start, end=end, interval="1d", auto_adjust=True, actions=False,
        )
        if df is None or df.empty:
            continue
        df.index = pd.to_datetime(df.index).tz_localize(None).normalize()
        df = df[["Open", "High", "Low", "Close", "Volume"]].dropna(subset=["Close"])
        panel[sym] = df
    if not panel:
        raise RuntimeError(f"yfinance returned no bars for any of {symbols}")
    return panel


def compute_adx(
    high: pd.Series, low: pd.Series, close: pd.Series, period: int,
) -> pd.Series:
    """
    Classic Wilder ADX implementation — same convention as
    :meth:`models.base_signal_generator.BaseSignalGenerator.calculate_adx`.

    Args:
        high: High series.
        low: Low series.
        close: Close series.
        period: Smoothing period (default 14).

    Returns:
        ADX series, indexed like ``close``. Warm-up rows remain NaN until
        the rolling window is fully populated.
    """
    high = high.astype(float)
    low = low.astype(float)
    close = close.astype(float)

    plus_dm = high.diff().copy()
    minus_dm = (-low.diff()).copy()
    plus_dm[plus_dm < 0.0] = 0.0
    minus_dm[minus_dm < 0.0] = 0.0

    prev_close = close.shift(1)
    tr = pd.concat(
        [(high - low).abs(), (high - prev_close).abs(), (low - prev_close).abs()],
        axis=1,
    ).max(axis=1)

    tr_sum = tr.rolling(window=period, min_periods=period).sum()
    plus_di = 100.0 * (plus_dm.rolling(window=period, min_periods=period).sum() / tr_sum)
    minus_di = 100.0 * (minus_dm.rolling(window=period, min_periods=period).sum() / tr_sum)
    dx = 100.0 * (plus_di - minus_di).abs() / (plus_di + minus_di + 1e-10)
    return dx.rolling(window=period, min_periods=period).mean()


def compute_atr(
    high: pd.Series, low: pd.Series, close: pd.Series, period: int,
) -> pd.Series:
    """Wilder ATR — EMA of the true-range."""
    high = high.astype(float)
    low = low.astype(float)
    close = close.astype(float)
    prev_close = close.shift(1)
    tr = pd.concat(
        [(high - low).abs(), (high - prev_close).abs(), (low - prev_close).abs()],
        axis=1,
    ).max(axis=1)
    return tr.ewm(alpha=1.0 / period, adjust=False).mean()


def classify_regime(adx: float, vol_ratio: float) -> str:
    """
    Map ``(ADX, ATR/price)`` to one of the three regime labels.

    Args:
        adx: ADX(14) value at the bar.
        vol_ratio: ATR(14) / close at the bar.

    Returns:
        ``"TRENDING"``, ``"MEAN_REV"`` or ``"VOLATILE"``. Bars that do
        not satisfy any rule (e.g. ADX between 20 and 25 with low
        volatility) fall back to ``"TRENDING"`` because that is the
        closest non-degenerate regime.
    """
    if vol_ratio is None or not np.isfinite(vol_ratio):
        return REGIME_TRENDING
    if vol_ratio >= VOL_MAX:
        return REGIME_VOLATILE
    if adx is None or not np.isfinite(adx):
        return REGIME_TRENDING
    if adx > ADX_TRENDING_MIN:
        return REGIME_TRENDING
    if adx < ADX_MEAN_REV_MAX and vol_ratio < VOL_MAX:
        return REGIME_MEAN_REV
    return REGIME_TRENDING


def build_training_frame(
    panel: Dict[str, pd.DataFrame], *, horizon: int,
) -> pd.DataFrame:
    """
    Stitch per-symbol features + labels + regime tags into one time-ordered
    frame.

    Args:
        panel: ``{symbol: OHLCV DataFrame}``.
        horizon: Forward-return horizon (bars).

    Returns:
        DataFrame with ``[*ML_FEATURE_NAMES, "label", "symbol", "regime",
        "adx", "vol_ratio"]`` sorted by index.
    """
    frames: List[pd.DataFrame] = []
    for sym, ohlcv in panel.items():
        feats = compute_ml_features(ohlcv)
        close = ohlcv["Close"].astype(float)
        high = ohlcv["High"].astype(float)
        low = ohlcv["Low"].astype(float)

        adx = compute_adx(high, low, close, ADX_PERIOD).rename("adx")
        atr = compute_atr(high, low, close, ATR_PERIOD)
        vol_ratio = (atr / close).rename("vol_ratio")

        fwd_return = np.log(close.shift(-horizon) / close)
        label = np.sign(fwd_return).replace(0.0, np.nan).rename("label")

        joined = (
            feats
            .join(adx, how="inner")
            .join(vol_ratio, how="inner")
            .join(label, how="inner")
            .dropna()
        )
        joined = joined.assign(
            symbol=sym,
            regime=[
                classify_regime(a, v) for a, v in
                zip(joined["adx"].tolist(), joined["vol_ratio"].tolist())
            ],
        )
        frames.append(joined)
    combined = pd.concat(frames, axis=0).sort_index()
    combined["label"] = combined["label"].astype(int)
    return combined


def digest_panel(panel: Dict[str, pd.DataFrame]) -> str:
    """Deterministic SHA256 of the concatenated Close series."""
    hasher = hashlib.sha256()
    for sym in sorted(panel.keys()):
        hasher.update(sym.encode("utf-8"))
        hasher.update(panel[sym]["Close"].to_numpy().tobytes())
    return hasher.hexdigest()[:16]


# ─────────────────────────────────────────────────────────────────────────────
# Training
# ─────────────────────────────────────────────────────────────────────────────

def train_model(
    X_train: pd.DataFrame, y_train: pd.Series,
) -> GradientBoostingClassifier:
    """Fit a ``GradientBoostingClassifier`` with fixed seed."""
    model = GradientBoostingClassifier(
        n_estimators=200, max_depth=3, learning_rate=0.05,
        subsample=0.8, random_state=RANDOM_STATE,
    )
    model.fit(X_train.to_numpy(), y_train.to_numpy())
    return model


def baseline_stats_from(
    model: GradientBoostingClassifier, X_train: pd.DataFrame,
) -> Dict[str, float]:
    """Record mean/std of the signed score on the training set."""
    proba = model.predict_proba(X_train.to_numpy())
    classes = list(model.classes_)
    up_idx = classes.index(1) if 1 in classes else len(classes) - 1
    p_up = proba[:, up_idx]
    signed = np.clip((p_up - 0.5) * 2.0, -1.0, 1.0)
    return {
        "mean": float(np.mean(signed)),
        "std": float(np.std(signed, ddof=0)),
        "n_samples": int(len(signed)),
    }


def _evaluate(
    model: GradientBoostingClassifier,
    X: pd.DataFrame, y: pd.Series,
) -> Tuple[float, float]:
    """Return ``(accuracy, log_loss)`` for ``(X, y)``."""
    if len(X) == 0:
        return float("nan"), float("nan")
    pred = model.predict(X.to_numpy())
    proba = model.predict_proba(X.to_numpy())
    acc = float(accuracy_score(y, pred))
    ll = float(log_loss(y, proba, labels=list(model.classes_)))
    return acc, ll


def train_on_subset(
    subset: pd.DataFrame,
    feature_cols: List[str],
    *,
    label_name: str,
    fallback_model: Optional[GradientBoostingClassifier],
    fallback_stats: Optional[Dict[str, float]],
) -> Dict[str, Any]:
    """
    Train a GBM on ``subset`` or defer to ``fallback_model`` when the
    subset is below :data:`MIN_REGIME_SAMPLES`. Returns the persisted
    payload shape.
    """
    n = len(subset)
    if n < MIN_REGIME_SAMPLES:
        logger.warning(
            "Regime %s has only %d samples (< %d). Falling back to full-dataset model.",
            label_name, n, MIN_REGIME_SAMPLES,
        )
        if fallback_model is None or fallback_stats is None:
            raise RuntimeError(
                f"Fallback model required for regime {label_name!r} but not provided"
            )
        return {
            "model": fallback_model,
            "baseline_stats": fallback_stats,
            "feature_names": feature_cols,
            "training_metadata": {
                "regime": label_name,
                "fell_back_to_full_dataset": True,
                "subset_rows": n,
                "min_regime_samples": MIN_REGIME_SAMPLES,
            },
        }

    ordered = subset.sort_index()
    leakage_check(
        ordered[feature_cols + ["label"]],
        label_col="label",
        feature_cols=feature_cols,
        reference_col=None,
        max_future_shift=FORWARD_HORIZON,
        leak_corr_threshold=0.98,
        raise_on_fail=True,
    )

    split_idx = int(len(ordered) * TRAIN_FRACTION)
    train_df = ordered.iloc[:split_idx]
    test_df = ordered.iloc[split_idx:]
    X_train = train_df[feature_cols]
    y_train = train_df["label"]
    X_test = test_df[feature_cols]
    y_test = test_df["label"]

    model = train_model(X_train, y_train)
    train_acc, train_ll = _evaluate(model, X_train, y_train)
    test_acc, test_ll = _evaluate(model, X_test, y_test)
    baseline = baseline_stats_from(model, X_train)

    return {
        "model": model,
        "baseline_stats": baseline,
        "feature_names": feature_cols,
        "training_metadata": {
            "regime": label_name,
            "fell_back_to_full_dataset": False,
            "subset_rows": n,
            "train_rows": int(len(train_df)),
            "test_rows": int(len(test_df)),
            "train_first_date": str(train_df.index[0].date()),
            "train_last_date": str(train_df.index[-1].date()),
            "test_first_date": str(test_df.index[0].date()),
            "test_last_date": str(test_df.index[-1].date()),
            "train_accuracy": train_acc,
            "test_accuracy": test_acc,
            "train_log_loss": train_ll,
            "test_log_loss": test_ll,
            "random_state": RANDOM_STATE,
            "train_fraction": TRAIN_FRACTION,
        },
    }


def resolve_save_path(regime: str) -> str:
    """Resolve the on-disk save path for ``regime``."""
    repo_root = Path(__file__).resolve().parents[1]
    default_dir = repo_root / "models" / "saved_advanced"
    default_dir.mkdir(parents=True, exist_ok=True)
    attr = REGIME_TO_PATH_ATTR[regime]
    configured = str(getattr(ApexConfig, attr, "") or "").strip()
    if configured:
        return configured
    return str(default_dir / REGIME_TO_DEFAULT_BASENAME[regime])


def save_payload(path: str, payload: Dict[str, Any]) -> None:
    """Persist the training payload as a pickle file."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main() -> int:
    setup_logging(level="WARNING", log_file=None, json_format=False, console_output=True)

    print("=" * 72)
    print(" Round 10 — Regime-specialised ML training")
    print("=" * 72)
    print(f" Symbols           : {list(SYMBOLS)}")
    print(f" Training span     : {TRAIN_START} -> {TRAIN_END}")
    print(f" Horizon           : {FORWARD_HORIZON} bars")
    print(f" Random seed       : {RANDOM_STATE}")
    print(f" ADX period        : {ADX_PERIOD}")
    print(f" ATR period        : {ATR_PERIOD}")
    print(f" ADX trending min  : {ADX_TRENDING_MIN}")
    print(f" ADX mean-rev max  : {ADX_MEAN_REV_MAX}")
    print(f" Vol ratio max     : {VOL_MAX}")
    print(f" Min regime samples: {MIN_REGIME_SAMPLES}")
    print()

    print(" Fetching OHLCV from yfinance ...")
    panel = fetch_yf_ohlcv(SYMBOLS, TRAIN_START, TRAIN_END)
    for sym, df in panel.items():
        print(f"   {sym}: {len(df)} bars  {df.index[0].date()} -> {df.index[-1].date()}")
    print(f" Panel SHA-16  : {digest_panel(panel)}")
    print()

    print(" Building feature matrix (with regime labels) ...")
    combined = build_training_frame(panel, horizon=FORWARD_HORIZON)
    feature_cols = list(ML_FEATURE_NAMES)
    total = len(combined)
    print(f"   rows              : {total}")
    print(f"   features          : {feature_cols}")
    regime_counts = combined["regime"].value_counts().to_dict()
    for regime in (REGIME_TRENDING, REGIME_MEAN_REV, REGIME_VOLATILE):
        n = int(regime_counts.get(regime, 0))
        pct = n / total if total else 0.0
        print(f"   rows {regime:<9} : {n:>5} ({pct:.2%})")
    print()

    # ── Pre-compute the full-dataset "fallback" model first so any under- ──
    # populated regime can inherit from it.
    print(" Training full-dataset fallback model (audit + fit) ...")
    leakage_check(
        combined[feature_cols + ["label"]],
        label_col="label",
        feature_cols=feature_cols,
        reference_col=None,
        max_future_shift=FORWARD_HORIZON,
        leak_corr_threshold=0.98,
        raise_on_fail=True,
    )
    split_idx_full = int(total * TRAIN_FRACTION)
    X_full_train = combined.iloc[:split_idx_full][feature_cols]
    y_full_train = combined.iloc[:split_idx_full]["label"]
    X_full_test = combined.iloc[split_idx_full:][feature_cols]
    y_full_test = combined.iloc[split_idx_full:]["label"]
    full_model = train_model(X_full_train, y_full_train)
    full_stats = baseline_stats_from(full_model, X_full_train)
    full_train_acc, full_train_ll = _evaluate(full_model, X_full_train, y_full_train)
    full_test_acc, full_test_ll = _evaluate(full_model, X_full_test, y_full_test)
    print(f"   full rows        : {total} (train {len(X_full_train)} / test {len(X_full_test)})")
    print(f"   full train acc   : {full_train_acc:.4f}  (log-loss {full_train_ll:.4f})")
    print(f"   full test  acc   : {full_test_acc:.4f}  (log-loss {full_test_ll:.4f})")
    print(f"   full baseline    : mean {full_stats['mean']:.4f}  std {full_stats['std']:.4f}")
    print()

    # ── Train each regime ─────────────────────────────────────────────────────
    regime_results: Dict[str, Dict[str, Any]] = {}
    for regime in (REGIME_TRENDING, REGIME_MEAN_REV, REGIME_VOLATILE):
        subset = combined[combined["regime"] == regime].copy()
        print("-" * 72)
        print(f" Training regime: {regime}")
        print(f"   subset rows    : {len(subset)}")
        payload = train_on_subset(
            subset=subset,
            feature_cols=feature_cols,
            label_name=regime,
            fallback_model=full_model,
            fallback_stats=full_stats,
        )
        meta = payload.get("training_metadata", {})
        if meta.get("fell_back_to_full_dataset"):
            print(f"   status         : FELL BACK to full-dataset model "
                  f"(needed >= {MIN_REGIME_SAMPLES}, got {meta.get('subset_rows')})")
        else:
            print(f"   status         : trained ({meta.get('train_rows')} train "
                  f"/ {meta.get('test_rows')} test rows)")
            print(f"   train accuracy : {meta.get('train_accuracy'):.4f}  "
                  f"(log-loss {meta.get('train_log_loss'):.4f})")
            print(f"   test  accuracy : {meta.get('test_accuracy'):.4f}  "
                  f"(log-loss {meta.get('test_log_loss'):.4f})")
            print(f"   baseline stats : mean {payload['baseline_stats']['mean']:.4f}  "
                  f"std {payload['baseline_stats']['std']:.4f}")

        # Attach common metadata.
        payload["training_metadata"]["symbols"] = list(SYMBOLS)
        payload["training_metadata"]["train_start"] = TRAIN_START
        payload["training_metadata"]["train_end"] = TRAIN_END
        payload["training_metadata"]["horizon"] = FORWARD_HORIZON
        payload["training_metadata"]["panel_sha16"] = digest_panel(panel)
        payload["training_metadata"]["adx_period"] = ADX_PERIOD
        payload["training_metadata"]["atr_period"] = ATR_PERIOD
        payload["training_metadata"]["adx_trending_min"] = ADX_TRENDING_MIN
        payload["training_metadata"]["adx_mean_rev_max"] = ADX_MEAN_REV_MAX
        payload["training_metadata"]["vol_max"] = VOL_MAX
        payload["training_metadata"]["min_regime_samples"] = MIN_REGIME_SAMPLES

        regime_results[regime] = payload

    # ── Persist ───────────────────────────────────────────────────────────────
    print()
    print(" Saving per-regime models")
    print(" " + "-" * 68)
    for regime, payload in regime_results.items():
        path = resolve_save_path(regime)
        save_payload(path, payload)
        print(f"   {regime:<9} -> {path}  ({os.path.getsize(path)} bytes)")
    print()

    print(" Round 10 training complete.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
