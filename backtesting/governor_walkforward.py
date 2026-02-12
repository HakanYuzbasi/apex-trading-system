"""
backtesting/governor_walkforward.py

Walk-forward tuner for governor policy controls with asset-class + regime scope.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from itertools import product
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import pandas as pd

from core.symbols import AssetClass, parse_symbol
from risk.governor_policy import GovernorPolicy, TierControls


@dataclass
class WalkForwardTuningConfig:
    cadence: str = "weekly"  # weekly | daily
    min_samples_per_fold: int = 30
    sharpe_floor_63d: float = 0.2
    historical_mdd_default: float = 0.08
    signal_threshold_grid: Sequence[float] = (0.0, 0.02, 0.03, 0.05, 0.07)
    confidence_boost_grid: Sequence[float] = (0.0, 0.03, 0.05, 0.08, 0.10)
    size_multiplier_grid: Sequence[float] = (1.0, 0.9, 0.8, 0.75, 0.65)


def _asset_class_from_symbol(symbol: str) -> str:
    try:
        parsed = parse_symbol(symbol)
        return parsed.asset_class.value
    except ValueError:
        return AssetClass.EQUITY.value


def _risk_adjusted_score(returns: np.ndarray) -> Tuple[float, float, float]:
    if returns.size < 5:
        return -999.0, 0.0, 1.0
    vol = float(np.std(returns))
    sharpe = float(np.mean(returns) / vol * np.sqrt(252)) if vol > 1e-12 else 0.0
    equity = np.cumprod(1.0 + returns)
    peaks = np.maximum.accumulate(equity)
    drawdowns = (equity - peaks) / np.maximum(peaks, 1e-9)
    mdd = abs(float(np.min(drawdowns)))
    score = sharpe - (2.0 * mdd)
    return score, sharpe, mdd


def _infer_strategy_return(row: pd.Series) -> float:
    r10 = row.get("return_10d")
    if r10 is None or pd.isna(r10):
        return 0.0
    direction = str(row.get("signal_direction", "BUY")).upper()
    if direction == "SELL":
        return float(-r10)
    return float(r10)


def _build_folds(df: pd.DataFrame, cadence: str) -> List[pd.DataFrame]:
    if df.empty:
        return []
    cadence = cadence.lower().strip()
    freq = "D" if cadence == "daily" else "W"
    grouped: List[pd.DataFrame] = []
    for _, part in df.groupby(pd.Grouper(key="signal_time", freq=freq)):
        if not part.empty:
            grouped.append(part.copy())
    return grouped


def tune_policy_for_key(
    signal_df: pd.DataFrame,
    asset_class: str,
    regime: str,
    config: WalkForwardTuningConfig,
) -> GovernorPolicy | None:
    if signal_df.empty:
        return None

    local = signal_df.copy()
    local["signal_time"] = pd.to_datetime(local["signal_time"], errors="coerce")
    local = local.dropna(subset=["signal_time"])
    local["asset_class"] = local["symbol"].map(_asset_class_from_symbol)
    local = local[
        (local["asset_class"] == asset_class.upper())
        & (local["regime"].astype(str).str.lower() == regime.lower())
    ].copy()
    if len(local) < config.min_samples_per_fold:
        return None

    folds = _build_folds(local, config.cadence)
    if len(folds) < 2:
        return None

    best_params = None
    best_score = -1e9
    best_sharpe = 0.0
    best_mdd = 1.0

    for signal_boost, confidence_boost, size_mult in product(
        config.signal_threshold_grid,
        config.confidence_boost_grid,
        config.size_multiplier_grid,
    ):
        fold_returns: List[float] = []
        for fold in folds:
            selected = fold[
                (fold["signal_value"].abs() >= (0.25 + signal_boost))
                & (fold["confidence"] >= (0.35 + confidence_boost))
            ]
            if len(selected) < config.min_samples_per_fold:
                continue
            simulated = selected.apply(_infer_strategy_return, axis=1).to_numpy(dtype=float)
            fold_returns.extend((simulated * size_mult).tolist())

        if len(fold_returns) < config.min_samples_per_fold:
            continue

        score, sharpe, mdd = _risk_adjusted_score(np.array(fold_returns, dtype=float))
        if score > best_score:
            best_score = score
            best_params = (signal_boost, confidence_boost, size_mult)
            best_sharpe = sharpe
            best_mdd = mdd

    if best_params is None:
        return None

    signal_boost, confidence_boost, size_mult = best_params
    version = f"wf-{datetime.utcnow().strftime('%Y%m%d%H%M%S')}"
    controls: Dict[str, TierControls] = {
        "green": TierControls(size_mult, signal_boost, confidence_boost, False),
        "yellow": TierControls(max(0.25, size_mult * 0.85), signal_boost + 0.02, confidence_boost + 0.02, False),
        "orange": TierControls(max(0.20, size_mult * 0.65), signal_boost + 0.05, confidence_boost + 0.05, False),
        "red": TierControls(max(0.15, size_mult * 0.40), signal_boost + 0.08, confidence_boost + 0.08, True),
    }
    return GovernorPolicy(
        asset_class=asset_class.upper(),
        regime=regime.lower(),
        version=version,
        oos_sharpe=float(best_sharpe),
        oos_drawdown=float(best_mdd),
        historical_mdd=max(float(best_mdd), config.historical_mdd_default),
        sharpe_floor_63d=float(config.sharpe_floor_63d),
        tier_controls=controls,
        metadata={
            "cadence": config.cadence,
            "samples": int(len(local)),
            "folds": int(len(folds)),
            "objective_score": float(best_score),
        },
    )


def tune_policies(
    signal_df: pd.DataFrame,
    config: WalkForwardTuningConfig,
    regimes_by_asset_class: Dict[str, Iterable[str]],
) -> List[GovernorPolicy]:
    tuned: List[GovernorPolicy] = []
    for asset_class, regimes in regimes_by_asset_class.items():
        for regime in regimes:
            policy = tune_policy_for_key(signal_df, asset_class, regime, config)
            if policy:
                tuned.append(policy)
    return tuned
