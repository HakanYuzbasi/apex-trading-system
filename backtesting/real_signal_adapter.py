"""
backtesting/real_signal_adapter.py — Backtest-side wrapper around the
production signal stack (PatternSignal + ORBSignal + SignalAggregator +
optional ML classifier).

The production execution loop at :mod:`core.execution_loop` builds a list
of :class:`~signals.signal_aggregator.SignalVote` objects and feeds them
into ``SignalAggregator.combine(...)`` together with a primary ML
score. The backtester previously bypassed that wiring and called a
placeholder momentum z-score. This adapter reproduces the production
path so backtest results reflect the same decisioning logic as live
trading:

  * primary ML signal   → loaded from one of ``ApexConfig.ML_MODEL_PATH_*``
    (falls back to rule-based momentum when no ``.pkl`` is found)
  * candlestick vote    → :class:`~signals.pattern_signal.PatternSignal`
  * ORB vote            → :class:`~signals.orb_signal.ORBSignal` (silently
    neutral on daily OHLCV since it is time-gated to intraday ET hours)
  * combination         → :meth:`~signals.signal_aggregator.SignalAggregator.combine`

The adapter exposes a ``generate_ml_signal(symbol, prices)`` method
matching the backtester's expected contract (see
``backtesting/advanced_backtester.py:615``) and additionally a richer
``get_signal(symbol, ohlcv_df, bar_idx)`` API returning a signed
confidence in ``[-1.0, 1.0]`` along with per-source diagnostics.

All thresholds — including entry magnitude, confidence floors and ML
input-feature windows — are sourced from :class:`config.ApexConfig`.
"""
from __future__ import annotations

import logging
import os
import pickle
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from config import ApexConfig
from signals.signal_aggregator import SignalAggregator, SignalVote
from signals.pattern_signal import PatternSignal
from signals.orb_signal import ORBSignal

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# ML feature engineering (matches scripts/train_baseline_models.py)
# ─────────────────────────────────────────────────────────────────────────────

_ML_RSI_PERIOD: int = int(getattr(ApexConfig, "ML_FEATURE_RSI_PERIOD", 14))
_ML_MACD_FAST: int = int(getattr(ApexConfig, "ML_FEATURE_MACD_FAST", 12))
_ML_MACD_SLOW: int = int(getattr(ApexConfig, "ML_FEATURE_MACD_SLOW", 26))
_ML_MACD_SIGNAL: int = int(getattr(ApexConfig, "ML_FEATURE_MACD_SIGNAL", 9))
_ML_ATR_PERIOD: int = int(getattr(ApexConfig, "ML_FEATURE_ATR_PERIOD", 14))
_ML_VOLUME_LOOKBACK: int = int(getattr(ApexConfig, "ML_FEATURE_VOLUME_LOOKBACK", 20))
_ML_SMA_FAST: int = int(getattr(ApexConfig, "ML_FEATURE_SMA_FAST", 20))
_ML_SMA_SLOW: int = int(getattr(ApexConfig, "ML_FEATURE_SMA_SLOW", 50))

# Regime classification thresholds — identical to those in
# ``scripts.train_baseline_models`` so bars are classified the same way
# at training time and at inference time.
_REGIME_ADX_PERIOD: int = int(os.getenv("APEX_REGIME_ADX_PERIOD", "14"))
_REGIME_ATR_PERIOD: int = int(os.getenv("APEX_REGIME_ATR_PERIOD", "14"))
_REGIME_ADX_TRENDING_MIN: float = float(
    os.getenv("APEX_REGIME_ADX_TRENDING_MIN", "25.0")
)
_REGIME_ADX_MEAN_REV_MAX: float = float(
    os.getenv("APEX_REGIME_ADX_MEAN_REV_MAX", "20.0")
)
_REGIME_VOL_MAX: float = float(os.getenv("APEX_REGIME_VOL_MAX", "0.015"))


def _last_adx(high: pd.Series, low: pd.Series, close: pd.Series) -> float:
    """Return the latest ADX(14) value or NaN when warm-up is incomplete."""
    period = _REGIME_ADX_PERIOD
    if len(close) < period * 2 + 1:
        return float("nan")
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
    adx = dx.rolling(window=period, min_periods=period).mean()
    if adx.empty:
        return float("nan")
    val = float(adx.iloc[-1])
    return val if np.isfinite(val) else float("nan")


def _last_atr_ratio(high: pd.Series, low: pd.Series, close: pd.Series) -> float:
    """Return the latest ATR(14) / Close or NaN when warm-up is incomplete."""
    period = _REGIME_ATR_PERIOD
    if len(close) < period + 1:
        return float("nan")
    high = high.astype(float)
    low = low.astype(float)
    close = close.astype(float)
    prev_close = close.shift(1)
    tr = pd.concat(
        [(high - low).abs(), (high - prev_close).abs(), (low - prev_close).abs()],
        axis=1,
    ).max(axis=1)
    atr = tr.ewm(alpha=1.0 / period, adjust=False).mean()
    last_atr = float(atr.iloc[-1])
    last_close = float(close.iloc[-1])
    if not np.isfinite(last_atr) or last_close <= 0.0:
        return float("nan")
    return last_atr / last_close


def classify_regime(adx: float, vol_ratio: float) -> str:
    """
    Map ``(ADX, ATR/price)`` to a regime label.

    Mirrors :func:`scripts.train_baseline_models.classify_regime`. NaN
    inputs default to ``"TRENDING"`` — the same fallback used in training
    when the warm-up window has not yet elapsed.
    """
    if vol_ratio is not None and np.isfinite(vol_ratio) and vol_ratio >= _REGIME_VOL_MAX:
        return "VOLATILE"
    if adx is not None and np.isfinite(adx) and adx > _REGIME_ADX_TRENDING_MIN:
        return "TRENDING"
    if (
        adx is not None and np.isfinite(adx) and adx < _REGIME_ADX_MEAN_REV_MAX
        and vol_ratio is not None and np.isfinite(vol_ratio)
        and vol_ratio < _REGIME_VOL_MAX
    ):
        return "MEAN_REV"
    return "TRENDING"


def compute_ml_features(ohlcv: pd.DataFrame) -> pd.DataFrame:
    """
    Build the feature matrix expected by the baseline ML classifier.

    Args:
        ohlcv: DataFrame with columns ``Open``, ``High``, ``Low``, ``Close``,
            ``Volume`` and a monotonic DatetimeIndex.

    Returns:
        DataFrame with columns ``rsi``, ``macd``, ``macd_signal``,
        ``macd_hist``, ``atr``, ``volume_ratio``, ``price_vs_sma20``,
        ``price_vs_sma50``. Rows before the indicator warm-up window are
        dropped. The index is inherited from ``ohlcv``.

    Raises:
        KeyError: If ``ohlcv`` is missing any of the required columns.
    """
    required = ("Open", "High", "Low", "Close", "Volume")
    missing = [c for c in required if c not in ohlcv.columns]
    if missing:
        raise KeyError(f"compute_ml_features: missing columns {missing}")

    close = ohlcv["Close"].astype(float)
    high = ohlcv["High"].astype(float)
    low = ohlcv["Low"].astype(float)
    volume = ohlcv["Volume"].astype(float)

    # RSI(14) — Wilder-style exponential smoothing.
    delta = close.diff()
    gain = delta.where(delta > 0.0, 0.0)
    loss = (-delta).where(delta < 0.0, 0.0)
    avg_gain = gain.ewm(alpha=1.0 / _ML_RSI_PERIOD, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1.0 / _ML_RSI_PERIOD, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0.0, np.nan)
    rsi = 100.0 - (100.0 / (1.0 + rs))
    rsi = rsi.fillna(50.0)

    # MACD(12, 26, 9)
    ema_fast = close.ewm(span=_ML_MACD_FAST, adjust=False).mean()
    ema_slow = close.ewm(span=_ML_MACD_SLOW, adjust=False).mean()
    macd = ema_fast - ema_slow
    macd_signal = macd.ewm(span=_ML_MACD_SIGNAL, adjust=False).mean()
    macd_hist = macd - macd_signal

    # ATR(14) — Wilder's true-range smoothing.
    prev_close = close.shift(1)
    tr = pd.concat(
        [
            (high - low).abs(),
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    atr = tr.ewm(alpha=1.0 / _ML_ATR_PERIOD, adjust=False).mean()

    # Volume ratio vs trailing mean.
    vol_mean = volume.rolling(_ML_VOLUME_LOOKBACK, min_periods=_ML_VOLUME_LOOKBACK).mean()
    volume_ratio = volume / vol_mean.replace(0.0, np.nan)

    # Price vs moving averages (normalised deviation).
    sma_fast = close.rolling(_ML_SMA_FAST, min_periods=_ML_SMA_FAST).mean()
    sma_slow = close.rolling(_ML_SMA_SLOW, min_periods=_ML_SMA_SLOW).mean()
    price_vs_sma20 = (close - sma_fast) / sma_fast
    price_vs_sma50 = (close - sma_slow) / sma_slow

    feats = pd.DataFrame(
        {
            "rsi": rsi,
            "macd": macd,
            "macd_signal": macd_signal,
            "macd_hist": macd_hist,
            "atr": atr,
            "volume_ratio": volume_ratio,
            "price_vs_sma20": price_vs_sma20,
            "price_vs_sma50": price_vs_sma50,
        },
        index=ohlcv.index,
    )
    return feats.dropna()


ML_FEATURE_NAMES: Tuple[str, ...] = (
    "rsi",
    "macd",
    "macd_signal",
    "macd_hist",
    "atr",
    "volume_ratio",
    "price_vs_sma20",
    "price_vs_sma50",
)


# ─────────────────────────────────────────────────────────────────────────────
# Return type
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class SignalDecision:
    """Aggregated signal decision emitted to the backtester."""

    symbol: str
    signal: float
    confidence: float
    primary: float
    votes: List[SignalVote] = field(default_factory=list)
    contributing_sources: List[str] = field(default_factory=list)
    blocked: bool = False


# ─────────────────────────────────────────────────────────────────────────────
# Adapter
# ─────────────────────────────────────────────────────────────────────────────

class RealSignalAdapter:
    """
    Backtest-compatible wrapper that reproduces the production signal stack.

    Construction loads any available per-regime ML classifier
    (``ML_MODEL_PATH_TRENDING`` → ``MEAN_REV`` → ``VOLATILE`` in order of
    preference) and records whether ML is active. When all three paths
    are unset or the ``.pkl`` files are missing, the adapter falls back
    to a rule-based primary signal: a capped 20-day z-score of close
    prices. The rule-based fallback itself is deterministic and
    documented — it is NOT the same as the placeholder
    ``MomentumZScoreSignalGenerator`` because it now feeds through the
    production ``SignalAggregator`` alongside votes from
    ``PatternSignal`` and ``ORBSignal``.

    Thread-safe for reads. Not thread-safe for the internal source-tally
    counter used for reporting (single-writer expected).
    """

    _DEFAULT_ENTRY_THRESHOLD: float = 0.19
    _FALLBACK_Z_WINDOW: int = 20
    _DEFAULT_PRIMARY_CONF: float = 0.60

    def __init__(
        self,
        *,
        ml_model_path: Optional[str] = None,
        entry_threshold: Optional[float] = None,
        enable_ml: bool = True,
    ) -> None:
        """
        Args:
            ml_model_path: Explicit path to a baseline ML ``.pkl``. When
                ``None``, falls through ``ApexConfig.ML_MODEL_PATH_*`` in
                the order ``TRENDING`` → ``MEAN_REV`` → ``VOLATILE``.
            entry_threshold: Minimum ``|signal|`` for a long/short entry.
                When ``None``, resolves to ``ApexConfig.ML_CONFIDENCE_THRESHOLD``
                (default 0.19 — P40 of the Round 10 calibration run). Falls
                back to the legacy ``SIGNAL_ENTRY_THRESHOLD`` config field
                and then the class default ``_DEFAULT_ENTRY_THRESHOLD``.
            enable_ml: When ``False``, skips all ML-model discovery and
                forces the rule-based momentum primary.
        """
        if entry_threshold is not None:
            resolved_threshold = float(entry_threshold)
        else:
            resolved_threshold = float(
                getattr(
                    ApexConfig,
                    "ML_CONFIDENCE_THRESHOLD",
                    getattr(
                        ApexConfig,
                        "SIGNAL_ENTRY_THRESHOLD",
                        self._DEFAULT_ENTRY_THRESHOLD,
                    ),
                )
            )
        self._entry_threshold: float = resolved_threshold

        self._aggregator = SignalAggregator()
        self._pattern_signal = PatternSignal()
        self._orb_signal = ORBSignal()

        self._ml_model: Optional[Any] = None
        self._ml_baseline: Optional[Dict[str, float]] = None
        self._ml_model_path: Optional[str] = None
        self._ml_active: bool = False
        if enable_ml:
            self._load_ml_model(explicit_path=ml_model_path)

        if not self._ml_active:
            logger.info(
                "RealSignalAdapter: no trained ML model loaded — "
                "falling back to rule-based primary signal (20-day z-score); "
                "pattern + ORB votes still flow through SignalAggregator."
            )

        # Diagnostics: tally how often each source contributes to a non-zero
        # final decision. Reported at the end of the backtest.
        self._source_hits: Dict[str, int] = {
            "ml": 0,
            "pattern": 0,
            "orb": 0,
            "momentum_fallback": 0,
        }
        self._entry_count: int = 0
        # Per-regime hit counter populated by :meth:`_ml_primary` each time
        # a non-zero ML score is returned for a specific regime.
        self._regime_hits: Dict[str, int] = {}

    # ── ML model loading ──────────────────────────────────────────────────────

    _REGIME_TRENDING: str = "TRENDING"
    _REGIME_MEAN_REV: str = "MEAN_REV"
    _REGIME_VOLATILE: str = "VOLATILE"

    _REGIME_ATTR_MAP: Tuple[Tuple[str, str], ...] = (
        ("TRENDING", "ML_MODEL_PATH_TRENDING"),
        ("MEAN_REV", "ML_MODEL_PATH_MEAN_REV"),
        ("VOLATILE", "ML_MODEL_PATH_VOLATILE"),
    )

    def _load_ml_model(self, explicit_path: Optional[str]) -> None:
        """
        Load up to three regime-specialised models (one per regime key).

        When ``explicit_path`` is provided, that single model is used for
        every regime (back-compat with callers that want the legacy
        "one model everywhere" behaviour).
        """
        self._regime_models: Dict[str, Any] = {}
        self._regime_baselines: Dict[str, Optional[Dict[str, float]]] = {}
        self._regime_model_paths: Dict[str, str] = {}

        path_entries: List[Tuple[str, str]]
        if explicit_path:
            path_entries = [(regime, explicit_path) for regime, _ in self._REGIME_ATTR_MAP]
        else:
            path_entries = []
            for regime, attr in self._REGIME_ATTR_MAP:
                p = str(getattr(ApexConfig, attr, "") or "").strip()
                if p:
                    path_entries.append((regime, p))

        for regime, path in path_entries:
            if not os.path.isfile(path):
                continue
            try:
                with open(path, "rb") as f:
                    payload = pickle.load(f)
                if isinstance(payload, dict) and "model" in payload:
                    model_obj = payload["model"]
                    baseline = payload.get("baseline_stats")
                else:
                    model_obj = payload
                    baseline = None
                self._regime_models[regime] = model_obj
                self._regime_baselines[regime] = baseline
                self._regime_model_paths[regime] = path
                logger.info(
                    "RealSignalAdapter: loaded %s model from %s (baseline=%s)",
                    regime, path, bool(baseline),
                )
            except Exception as exc:
                logger.warning(
                    "RealSignalAdapter: failed to load %s model %s (%s)",
                    regime, path, exc,
                )

        # Legacy single-model attributes mirror whichever regime loaded
        # first; callers that haven't migrated to per-regime routing still
        # see a usable model.
        if self._regime_models:
            first_regime = next(iter(self._regime_models))
            self._ml_model = self._regime_models[first_regime]
            self._ml_baseline = self._regime_baselines[first_regime]
            self._ml_model_path = self._regime_model_paths[first_regime]
            self._ml_active = True

    # ── Primary signal computation ────────────────────────────────────────────

    def _momentum_zscore(self, closes: pd.Series) -> float:
        if closes is None or len(closes) < self._FALLBACK_Z_WINDOW + 1:
            return 0.0
        window = closes.tail(self._FALLBACK_Z_WINDOW)
        mean = float(window.mean())
        std = float(window.std(ddof=0))
        if std <= 0.0 or not np.isfinite(std):
            return 0.0
        last = float(closes.iloc[-1])
        z = (last - mean) / std
        sig = float(np.clip(z / 2.0, -1.0, 1.0))
        return sig if np.isfinite(sig) else 0.0

    def _select_regime_model(
        self, ohlcv: pd.DataFrame,
    ) -> Tuple[str, Any, Optional[Dict[str, float]]]:
        """
        Classify the most recent bar's regime and pick the matching model.

        Returns:
            ``(regime_label, model, baseline_stats)``. If no model is
            registered for the computed regime, falls back to the first
            available model with an ``"_FALLBACK"`` suffix appended to
            the regime label so diagnostics record the fallback.
        """
        adx = _last_adx(ohlcv["High"], ohlcv["Low"], ohlcv["Close"])
        vol_ratio = _last_atr_ratio(ohlcv["High"], ohlcv["Low"], ohlcv["Close"])
        regime = classify_regime(adx, vol_ratio)

        model = self._regime_models.get(regime)
        baseline = self._regime_baselines.get(regime)
        if model is not None:
            return regime, model, baseline

        # Fallback: any other loaded regime.
        if self._regime_models:
            first = next(iter(self._regime_models))
            return (
                f"{regime}_FALLBACK",
                self._regime_models[first],
                self._regime_baselines.get(first),
            )
        return regime, None, None

    def _ml_primary(self, ohlcv: pd.DataFrame) -> Tuple[float, float]:
        """
        Return ``(signal, confidence)`` from the regime-appropriate ML
        classifier evaluated on the most recent bar.

        The bar's regime is derived from ADX(14) / ATR(14)-to-close using
        the same rules as :mod:`scripts.train_baseline_models`, then the
        matching model from ``self._regime_models`` is invoked.
        """
        if (
            not self._ml_active
            or ohlcv is None
            or len(ohlcv) < _ML_SMA_SLOW + 5
            or not self._regime_models
        ):
            return 0.0, 0.0
        try:
            regime_used, model, baseline = self._select_regime_model(ohlcv)
            if model is None:
                return 0.0, 0.0

            feats = compute_ml_features(ohlcv)
            if feats.empty:
                return 0.0, 0.0
            x = feats.iloc[[-1]][list(ML_FEATURE_NAMES)].to_numpy()
            if hasattr(model, "predict_proba"):
                proba = model.predict_proba(x)
                classes = list(getattr(model, "classes_", [-1, 1]))
                up_idx = classes.index(1) if 1 in classes else len(classes) - 1
                p_up = float(proba[0, up_idx])
            else:
                raw = model.predict(x)
                p_up = float(np.clip(raw[0], 0.0, 1.0))

            signal = float(np.clip((p_up - 0.5) * 2.0, -1.0, 1.0))
            confidence = float(abs(signal))
            if baseline is not None:
                mu = float(baseline.get("mean", 0.0))
                sd = float(baseline.get("std", 1.0)) or 1.0
                z_drift = (signal - mu) / sd
                if not np.isfinite(z_drift) or abs(z_drift) > 4.0:
                    return 0.0, 0.0

            # Per-regime hit counter (diagnostics).
            self._regime_hits[regime_used] = (
                self._regime_hits.get(regime_used, 0) + 1
            )
            return signal, confidence
        except Exception as exc:
            logger.debug("RealSignalAdapter._ml_primary error: %s", exc)
            return 0.0, 0.0

    # ── Public contracts ──────────────────────────────────────────────────────

    def get_signal(
        self,
        symbol: str,
        ohlcv_df: pd.DataFrame,
        bar_idx: Optional[int] = None,
    ) -> SignalDecision:
        """
        Compute a combined directional signal for ``symbol`` up to ``bar_idx``.

        Args:
            symbol: Ticker (e.g. ``"AAPL"`` for equities, ``"BTC/USD"`` for
                crypto). Used to route the asset class through the
                aggregator and skip ORB on non-equities.
            ohlcv_df: DataFrame with an OHLCV schema and a DatetimeIndex.
                Must contain every bar up to and including the decision
                point; the adapter slices to ``bar_idx`` internally.
            bar_idx: Zero-based integer index of the decision bar. When
                ``None``, the last row is used.

        Returns:
            :class:`SignalDecision` with a signed ``signal`` in
            ``[-1.0, 1.0]``, the aggregator's ``confidence`` in
            ``[0.0, 1.0]``, the originating ``primary`` score, the
            applied votes and a list of contributing sources for
            diagnostics.
        """
        if ohlcv_df is None or ohlcv_df.empty:
            return SignalDecision(symbol=symbol, signal=0.0, confidence=0.0, primary=0.0)

        n = len(ohlcv_df)
        if bar_idx is None:
            bar_idx = n - 1
        bar_idx = max(0, min(int(bar_idx), n - 1))
        window = ohlcv_df.iloc[: bar_idx + 1]

        asset_class = "crypto" if "/" in symbol else "equity"

        # Primary signal: ML preferred, rule-based momentum as fallback.
        if self._ml_active:
            primary_signal, primary_conf = self._ml_primary(window)
            if primary_signal == 0.0 and primary_conf == 0.0:
                # ML returned a truly neutral score — still feed rule-based
                # momentum so the aggregator has something to gate votes on.
                primary_signal = self._momentum_zscore(window["Close"])
                primary_conf = self._DEFAULT_PRIMARY_CONF * abs(primary_signal)
                source_primary = "momentum_fallback"
            else:
                source_primary = "ml"
        else:
            primary_signal = self._momentum_zscore(window["Close"])
            primary_conf = self._DEFAULT_PRIMARY_CONF * abs(primary_signal)
            source_primary = "momentum_fallback"

        # Votes.
        votes: List[SignalVote] = []
        sources: List[str] = []
        if primary_signal != 0.0:
            sources.append(source_primary)

        # Candlestick pattern vote — needs ≥3 bars, any asset class.
        if len(window) >= 3:
            try:
                pat = self._pattern_signal.get_signal(symbol, window)
                if pat.confidence > 0.15 and pat.signal != 0.0:
                    votes.append(
                        SignalVote(
                            signal=float(pat.signal),
                            confidence=float(pat.confidence),
                            source="pattern",
                            applies_to="all",
                        )
                    )
                    sources.append("pattern")
            except Exception as exc:
                logger.debug("RealSignalAdapter pattern error %s: %s", symbol, exc)

        # ORB vote — equities only, and intrinsically intraday.
        if asset_class == "equity":
            try:
                orb_ctx = self._orb_signal.get_signal(
                    symbol=symbol,
                    current_price=float(window["Close"].iloc[-1]),
                    rvol=1.0,
                )
                if orb_ctx.signal != 0.0 and orb_ctx.confidence > 0.0:
                    votes.append(
                        SignalVote(
                            signal=float(orb_ctx.signal),
                            confidence=float(orb_ctx.confidence),
                            source="orb",
                            applies_to="equity",
                        )
                    )
                    sources.append("orb")
            except Exception as exc:
                logger.debug("RealSignalAdapter ORB error %s: %s", symbol, exc)

        # Aggregate.
        adj_conf, blocked = self._aggregator.combine(
            primary_signal=float(primary_signal),
            votes=votes,
            primary_confidence=float(max(0.0, min(1.0, primary_conf))),
            asset_class=asset_class,
        )

        if blocked or abs(primary_signal) < self._entry_threshold:
            final_signal = 0.0
        else:
            final_signal = float(
                np.clip(
                    float(np.sign(primary_signal)) * float(adj_conf),
                    -1.0,
                    1.0,
                )
            )

        # Diagnostics tally.
        if final_signal != 0.0:
            self._entry_count += 1
            for s in sources:
                self._source_hits[s] = self._source_hits.get(s, 0) + 1

        return SignalDecision(
            symbol=symbol,
            signal=final_signal,
            confidence=float(adj_conf),
            primary=float(primary_signal),
            votes=votes,
            contributing_sources=sources,
            blocked=bool(blocked),
        )

    # ── Legacy contract used by AdvancedBacktester ────────────────────────────

    def generate_ml_signal(
        self,
        symbol: str,
        prices: pd.Series,
    ) -> Dict[str, float]:
        """
        Backtester-compatible shim.

        ``AdvancedBacktester._check_entries`` calls
        ``signal_generator.generate_ml_signal(symbol, prices)`` with a
        close-price Series. To expose pattern + ORB votes we reconstruct a
        full OHLCV frame by looking up the panel registered via
        :meth:`attach_panel`; if no panel is attached we degrade
        gracefully to a close-only primary signal.

        Args:
            symbol: Ticker.
            prices: Close-price ``pd.Series`` up to the decision bar.

        Returns:
            ``{"signal": float}`` with the combined directional score.
        """
        panel = getattr(self, "_panel", None)
        ohlcv = None
        if panel is not None and symbol in panel:
            full = panel[symbol]
            if prices is not None and len(prices) > 0:
                try:
                    ohlcv = full.loc[: prices.index[-1]]
                except Exception:
                    ohlcv = full.iloc[: len(prices)]
            else:
                ohlcv = full
        if ohlcv is None or ohlcv.empty:
            # Build a minimal frame so PatternSignal validation passes.
            if prices is None or len(prices) == 0:
                return {"signal": 0.0}
            ohlcv = pd.DataFrame(
                {
                    "Open": prices,
                    "High": prices,
                    "Low": prices,
                    "Close": prices,
                    "Volume": 0.0,
                },
                index=prices.index,
            )

        decision = self.get_signal(symbol, ohlcv)
        return {"signal": float(decision.signal)}

    # ── Panel registration + diagnostics ──────────────────────────────────────

    def attach_panel(self, panel: Dict[str, pd.DataFrame]) -> None:
        """Register the full OHLCV panel so ``generate_ml_signal`` can
        route through the rich ``get_signal`` code path."""
        self._panel = panel

    def source_hit_report(self) -> Dict[str, int]:
        """Return a shallow copy of the per-source contribution tally."""
        report = dict(self._source_hits)
        report["_entries_fired"] = self._entry_count
        report["_ml_active"] = int(self._ml_active)
        return report

    def regime_hit_report(self) -> Dict[str, int]:
        """Per-regime counter for non-zero ML scores (diagnostics)."""
        return dict(self._regime_hits)

    def loaded_regime_models(self) -> List[str]:
        """List the regime labels for which a model is currently loaded."""
        return list(self._regime_models.keys())
