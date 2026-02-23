#!/usr/bin/env python3
"""
scripts/calibrate_social_shock_thresholds.py

Calibrate social-shock thresholds by asset class + regime using walk-forward windows.
Persists versioned snapshots in data/governor_policies/.
"""

from __future__ import annotations

import argparse
import logging
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple
import sys
from typing import Optional

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import ApexConfig
from core.symbols import parse_symbol
from monitoring.signal_outcome_tracker import SignalOutcomeTracker
from risk.social_governor_policy import SocialGovernorPolicy, SocialGovernorPolicyRepository


logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


@dataclass(frozen=True)
class Candidate:
    reduce_threshold: float
    block_threshold: float
    verified_event_weight: float
    max_probability_divergence: float
    max_source_disagreement: float


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def _asset_class_from_symbol(symbol: object) -> str:
    try:
        parsed = parse_symbol(str(symbol))
        return parsed.asset_class.value
    except Exception:
        return "EQUITY"


def _map_regime(asset_class: str, regime: object) -> str:
    reg = str(regime or "default").strip().lower()
    asset = str(asset_class or "EQUITY").upper()
    if asset == "EQUITY":
        if reg in {"bear", "strong_bear"}:
            return "risk_off"
        if reg in {"bull", "strong_bull"}:
            return "risk_on"
        if reg in {"volatile", "high_volatility"}:
            return "volatile"
        return "default"
    if asset == "FOREX":
        if reg in {"bear", "strong_bear", "volatile", "high_volatility"}:
            return "carry_crash"
        if reg in {"bull", "strong_bull"}:
            return "carry"
        return "default"
    if asset == "CRYPTO":
        if reg in {"bear", "strong_bear"}:
            return "crash"
        if reg in {"bull", "strong_bull"}:
            return "trend"
        if reg in {"volatile", "high_volatility"}:
            return "high_vol"
        return "default"
    return "default"


def _strategy_return(row: pd.Series) -> float:
    r10 = row.get("return_10d")
    if r10 is None or pd.isna(r10):
        return 0.0
    direction = str(row.get("signal_direction", "BUY")).upper()
    if direction == "SELL":
        return float(-r10)
    return float(r10)


def _build_model_frame(df: pd.DataFrame) -> pd.DataFrame:
    local = df.copy()
    local["signal_time"] = pd.to_datetime(local["signal_time"], errors="coerce")
    local = local.dropna(subset=["signal_time"])
    local["asset_class"] = local["symbol"].map(_asset_class_from_symbol)
    local["governor_regime"] = local.apply(
        lambda r: _map_regime(str(r.get("asset_class")), r.get("regime")),
        axis=1,
    )
    local["signal_value"] = pd.to_numeric(local.get("signal_value"), errors="coerce").fillna(0.0).clip(-1.0, 1.0)
    local["confidence"] = pd.to_numeric(local.get("confidence"), errors="coerce").fillna(0.0).clip(0.0, 1.0)
    local["strategy_return"] = local.apply(_strategy_return, axis=1).astype(float)
    local["return_10d"] = pd.to_numeric(local.get("return_10d"), errors="coerce").fillna(0.0)
    local["mae_10d"] = pd.to_numeric(local.get("mae_10d"), errors="coerce").fillna(0.0)
    local["mfe_10d"] = pd.to_numeric(local.get("mfe_10d"), errors="coerce").fillna(0.0)

    stress_bonus = local["governor_regime"].map(
        {
            "risk_off": 0.22,
            "carry_crash": 0.22,
            "crash": 0.24,
            "high_vol": 0.18,
            "volatile": 0.16,
            "risk_on": 0.06,
            "carry": 0.06,
            "trend": 0.06,
            "default": 0.10,
        }
    ).fillna(0.10)

    local["risk_proxy"] = (
        0.52 * local["signal_value"].abs()
        + 0.28 * (1.0 - local["confidence"])
        + stress_bonus
    ).clip(0.0, 1.0)
    local["event_proxy"] = (
        (local["return_10d"].clip(upper=0.0).abs() * 6.0)
        + (local["mae_10d"].abs() * 2.0)
    ).clip(0.0, 1.0)
    local["divergence_proxy"] = (
        (local["signal_value"].abs() * (1.0 - local["confidence"]))
        + local["return_10d"].abs() * 0.2
    ).clip(0.0, 1.0)
    local["source_disagreement_proxy"] = (
        local["mae_10d"].abs() / (local["mfe_10d"].abs() + 1e-6)
    ).clip(0.0, 1.0)
    return local


def _walk_forward_windows(
    df: pd.DataFrame,
    *,
    train_days: int,
    test_days: int,
    step_days: int,
    min_samples: int,
) -> List[Tuple[pd.DataFrame, pd.DataFrame]]:
    if df.empty:
        return []
    local = df.sort_values("signal_time").copy()
    start = local["signal_time"].min()
    end = local["signal_time"].max()
    if pd.isna(start) or pd.isna(end):
        return []

    windows: List[Tuple[pd.DataFrame, pd.DataFrame]] = []
    cursor = start
    while cursor + timedelta(days=train_days + test_days) <= end:
        train_end = cursor + timedelta(days=train_days)
        test_end = train_end + timedelta(days=test_days)
        train = local[(local["signal_time"] >= cursor) & (local["signal_time"] < train_end)]
        test = local[(local["signal_time"] >= train_end) & (local["signal_time"] < test_end)]
        if len(train) >= min_samples and len(test) >= max(15, min_samples // 2):
            windows.append((train, test))
        cursor = cursor + timedelta(days=step_days)

    if windows:
        return windows

    # Fallback split when date range is short.
    split_idx = int(len(local) * 0.7)
    if split_idx <= 0 or split_idx >= len(local):
        return []
    train = local.iloc[:split_idx]
    test = local.iloc[split_idx:]
    if len(train) >= min_samples and len(test) >= max(10, min_samples // 3):
        return [(train, test)]
    return []


def _simulate_policy(df: pd.DataFrame, candidate: Candidate, *, base_notional: float = 20_000.0) -> Dict[str, float]:
    if df.empty:
        return {
            "score": -1e9,
            "blocked_alpha_opportunity": 0.0,
            "avoided_drawdown_estimate": 0.0,
            "hedge_cost_drag": 0.0,
            "verification_fail_rate": 0.0,
            "block_rate": 0.0,
            "reduce_rate": 0.0,
        }

    combined = (
        df["risk_proxy"].to_numpy(dtype=float)
        + candidate.verified_event_weight * df["event_proxy"].to_numpy(dtype=float)
    )
    combined = np.clip(combined, 0.0, 1.0)
    returns = df["strategy_return"].to_numpy(dtype=float)

    block_mask = combined >= candidate.block_threshold
    reduce_mask = (~block_mask) & (combined >= candidate.reduce_threshold)
    # Social governor reduce curve mirrors runtime behavior.
    progress = np.zeros_like(combined, dtype=float)
    denom = max(0.01, candidate.block_threshold - candidate.reduce_threshold)
    progress[reduce_mask] = (combined[reduce_mask] - candidate.reduce_threshold) / denom
    min_gross = float(ApexConfig.SOCIAL_SHOCK_MIN_GROSS_MULTIPLIER)
    exposure_multiplier = np.ones_like(combined, dtype=float)
    exposure_multiplier[block_mask] = min_gross
    exposure_multiplier[reduce_mask] = 1.0 - progress[reduce_mask] * (1.0 - min_gross)
    suppression = np.clip(1.0 - exposure_multiplier, 0.0, 1.0)

    blocked_alpha = float(np.sum(np.where(returns > 0.0, returns * base_notional * suppression, 0.0)))
    avoided_drawdown = float(np.sum(np.where(returns < 0.0, abs(returns) * base_notional * suppression, 0.0)))
    hedge_drag = float(
        np.sum(
            np.where(
                suppression > 0.0,
                base_notional * suppression * (0.0008 + df["event_proxy"].to_numpy(dtype=float) * 0.0006),
                0.0,
            )
        )
    )

    divergence = df["divergence_proxy"].to_numpy(dtype=float)
    disagreement = df["source_disagreement_proxy"].to_numpy(dtype=float)
    verification_fails = (divergence > candidate.max_probability_divergence) | (
        disagreement > candidate.max_source_disagreement
    )
    fail_rate = float(np.mean(verification_fails.astype(float)))
    score = avoided_drawdown - (1.15 * blocked_alpha) - (0.8 * hedge_drag) - (fail_rate * base_notional * 0.05)

    return {
        "score": float(score),
        "blocked_alpha_opportunity": blocked_alpha,
        "avoided_drawdown_estimate": avoided_drawdown,
        "hedge_cost_drag": hedge_drag,
        "verification_fail_rate": fail_rate,
        "block_rate": float(np.mean(block_mask.astype(float))),
        "reduce_rate": float(np.mean(reduce_mask.astype(float))),
    }


def _candidate_grid() -> Sequence[Candidate]:
    rows: List[Candidate] = []
    for reduce_threshold in (0.45, 0.52, 0.60, 0.67):
        for block_threshold in (0.72, 0.80, 0.86, 0.92):
            if block_threshold <= reduce_threshold + 0.08:
                continue
            for event_weight in (0.15, 0.25, 0.35, 0.45):
                for max_div in (0.10, 0.15, 0.20, 0.25):
                    for max_dis in (0.12, 0.18, 0.24):
                        rows.append(
                            Candidate(
                                reduce_threshold=reduce_threshold,
                                block_threshold=block_threshold,
                                verified_event_weight=event_weight,
                                max_probability_divergence=max_div,
                                max_source_disagreement=max_dis,
                            )
                        )
    return rows


def _tune_group(
    *,
    group_df: pd.DataFrame,
    version: str,
    asset_class: str,
    regime: str,
    windows: List[Tuple[pd.DataFrame, pd.DataFrame]],
) -> Optional[SocialGovernorPolicy]:
    grid = _candidate_grid()
    if not windows:
        return None

    selected_train_best: List[Candidate] = []
    scored_candidates: Dict[Candidate, List[float]] = {}
    diagnostics: Dict[Candidate, List[Dict[str, float]]] = {}

    for train_df, test_df in windows:
        train_scores: List[Tuple[Candidate, float]] = []
        for candidate in grid:
            train_score = _simulate_policy(train_df, candidate)["score"]
            train_scores.append((candidate, train_score))
        train_scores.sort(key=lambda x: x[1], reverse=True)
        top = [c for c, _ in train_scores[:8]]
        if not top:
            continue
        selected_train_best.append(top[0])
        for candidate in top:
            stats = _simulate_policy(test_df, candidate)
            scored_candidates.setdefault(candidate, []).append(float(stats["score"]))
            diagnostics.setdefault(candidate, []).append(stats)

    if not scored_candidates:
        return None

    winner = max(
        scored_candidates.items(),
        key=lambda item: (float(np.mean(item[1])), float(np.median(item[1]))),
    )[0]
    winner_stats = diagnostics.get(winner, [])
    avg_stats = {
        key: float(np.mean([row.get(key, 0.0) for row in winner_stats])) if winner_stats else 0.0
        for key in (
            "score",
            "blocked_alpha_opportunity",
            "avoided_drawdown_estimate",
            "hedge_cost_drag",
            "verification_fail_rate",
            "block_rate",
            "reduce_rate",
        )
    }

    return SocialGovernorPolicy(
        asset_class=asset_class.upper(),
        regime=regime.lower(),
        version=version,
        reduce_threshold=float(winner.reduce_threshold),
        block_threshold=float(winner.block_threshold),
        verified_event_weight=float(winner.verified_event_weight),
        verified_event_probability_floor=float(ApexConfig.SOCIAL_SHOCK_VERIFIED_EVENT_FLOOR),
        max_probability_divergence=float(winner.max_probability_divergence),
        max_source_disagreement=float(winner.max_source_disagreement),
        min_independent_sources=int(ApexConfig.PREDICTION_VERIFY_MIN_SOURCES),
        minimum_market_probability=float(ApexConfig.PREDICTION_VERIFY_MIN_MARKET_PROB),
        metadata={
            "samples": int(len(group_df)),
            "windows": int(len(windows)),
            "train_best_mode": {
                "reduce_threshold": float(np.median([c.reduce_threshold for c in selected_train_best]))
                if selected_train_best
                else winner.reduce_threshold,
                "block_threshold": float(np.median([c.block_threshold for c in selected_train_best]))
                if selected_train_best
                else winner.block_threshold,
            },
            "walk_forward_avg": avg_stats,
        },
    )


def _bootstrap_defaults(version: str) -> List[SocialGovernorPolicy]:
    defaults: List[SocialGovernorPolicy] = []
    for asset in ("GLOBAL", "EQUITY", "FOREX", "CRYPTO"):
        defaults.append(
            SocialGovernorPolicy(
                asset_class=asset,
                regime="default",
                version=version,
                reduce_threshold=float(ApexConfig.SOCIAL_SHOCK_REDUCE_THRESHOLD),
                block_threshold=float(ApexConfig.SOCIAL_SHOCK_BLOCK_THRESHOLD),
                verified_event_weight=float(ApexConfig.SOCIAL_SHOCK_VERIFIED_EVENT_WEIGHT),
                verified_event_probability_floor=float(ApexConfig.SOCIAL_SHOCK_VERIFIED_EVENT_FLOOR),
                max_probability_divergence=float(ApexConfig.PREDICTION_VERIFY_MAX_PROB_DIVERGENCE),
                max_source_disagreement=float(ApexConfig.PREDICTION_VERIFY_MAX_SOURCE_DISAGREEMENT),
                min_independent_sources=int(ApexConfig.PREDICTION_VERIFY_MIN_SOURCES),
                minimum_market_probability=float(ApexConfig.PREDICTION_VERIFY_MIN_MARKET_PROB),
                metadata={"source": "config_default_bootstrap"},
            )
        )
    return defaults


def _iter_group_keys(df: pd.DataFrame) -> Iterable[Tuple[str, str]]:
    grouped = df.groupby(["asset_class", "governor_regime"], dropna=False)
    for (asset_class, regime), part in grouped:
        if len(part) < 40:
            continue
        yield (str(asset_class).upper(), str(regime).lower())


def main() -> int:
    parser = argparse.ArgumentParser(description="Calibrate social-shock policy thresholds via walk-forward.")
    parser.add_argument("--data-dir", default=str(ApexConfig.DATA_DIR))
    parser.add_argument("--train-days", type=int, default=180)
    parser.add_argument("--test-days", type=int, default=45)
    parser.add_argument("--step-days", type=int, default=30)
    parser.add_argument("--min-samples", type=int, default=60)
    parser.add_argument("--activate", action="store_true", default=True)
    parser.add_argument("--no-activate", dest="activate", action="store_false")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    tracker = SignalOutcomeTracker(data_dir=str(data_dir))
    signal_df = tracker.get_signals_for_ml()
    if signal_df.empty:
        logger.warning("No completed signals found; writing default social policy snapshot.")
        version = f"sshock-{datetime.utcnow().strftime('%Y%m%d%H%M%S')}"
        defaults = _bootstrap_defaults(version)
        repo = SocialGovernorPolicyRepository(data_dir / "governor_policies")
        snapshot = repo.save_snapshot(
            version=version,
            policies=defaults,
            metadata={"source": "no_data_fallback"},
        )
        if args.activate:
            repo.activate_snapshot(snapshot)
            logger.info("Activated default social policy snapshot: %s", snapshot.name)
        return 0

    model_df = _build_model_frame(signal_df)
    version = f"sshock-{datetime.utcnow().strftime('%Y%m%d%H%M%S')}"
    policies: List[SocialGovernorPolicy] = []
    for asset_class, regime in sorted(_iter_group_keys(model_df)):
        part = model_df[
            (model_df["asset_class"] == asset_class)
            & (model_df["governor_regime"] == regime)
        ].copy()
        windows = _walk_forward_windows(
            part,
            train_days=max(30, int(args.train_days)),
            test_days=max(15, int(args.test_days)),
            step_days=max(7, int(args.step_days)),
            min_samples=max(20, int(args.min_samples)),
        )
        policy = _tune_group(
            group_df=part,
            version=version,
            asset_class=asset_class,
            regime=regime,
            windows=windows,
        )
        if policy:
            policies.append(policy)
            logger.info(
                "social policy tuned %s/%s reduce=%.2f block=%.2f event_w=%.2f max_div=%.2f max_dis=%.2f windows=%d samples=%d",
                policy.asset_class,
                policy.regime,
                policy.reduce_threshold,
                policy.block_threshold,
                policy.verified_event_weight,
                policy.max_probability_divergence,
                policy.max_source_disagreement,
                int(policy.metadata.get("windows", 0)),
                int(policy.metadata.get("samples", 0)),
            )

    if not policies:
        logger.warning("No tunable groups met sample thresholds; using default social policy set.")
        policies = _bootstrap_defaults(version)
    else:
        key_set = {(p.asset_class, p.regime) for p in policies}
        # Ensure deterministic fallback keys exist.
        for default in _bootstrap_defaults(version):
            if (default.asset_class, default.regime) not in key_set:
                policies.append(default)

    repo = SocialGovernorPolicyRepository(data_dir / "governor_policies")
    snapshot = repo.save_snapshot(
        version=version,
        policies=policies,
        metadata={
            "train_days": int(args.train_days),
            "test_days": int(args.test_days),
            "step_days": int(args.step_days),
            "min_samples": int(args.min_samples),
            "policy_count": len(policies),
        },
    )
    logger.info("Saved social policy snapshot: %s (%d policies)", snapshot.name, len(policies))
    if args.activate:
        repo.activate_snapshot(snapshot)
        logger.info("Activated social policy snapshot: %s", snapshot.name)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
