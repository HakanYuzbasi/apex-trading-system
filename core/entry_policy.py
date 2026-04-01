from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Mapping, Optional


def normalize_generator_signals(
    generator_signals: Optional[Mapping[str, Any]],
) -> Dict[str, float]:
    normalized: Dict[str, float] = {}
    if not isinstance(generator_signals, Mapping):
        return normalized
    for raw_name, raw_value in generator_signals.items():
        name = str(raw_name or "").strip().lower()
        if not name:
            continue
        try:
            normalized[name] = float(raw_value or 0.0)
        except Exception:
            continue
    return normalized


def select_dominant_generator(
    generator_signals: Optional[Mapping[str, Any]],
) -> str:
    normalized = normalize_generator_signals(generator_signals)
    if not normalized:
        return "composite"
    best_name = "composite"
    best_abs_signal = 0.0
    for name, value in normalized.items():
        abs_value = abs(float(value))
        if abs_value > best_abs_signal:
            best_name = name
            best_abs_signal = abs_value
    return best_name if best_abs_signal > 1e-9 else "composite"


def expectancy_bucket_key(asset_class: str, regime: str, dominant_generator: str) -> str:
    return (
        f"{str(asset_class or 'UNKNOWN').upper()}|"
        f"{str(regime or 'unknown').lower()}|"
        f"{str(dominant_generator or 'composite').lower()}"
    )


def lookup_expectancy_bucket(
    ledger: Optional[Mapping[str, Any]],
    *,
    asset_class: str,
    regime: str,
    dominant_generator: str,
) -> Dict[str, Any]:
    if not isinstance(ledger, Mapping):
        return {}
    by_bucket = ledger.get("by_bucket", {})
    if not isinstance(by_bucket, Mapping):
        return {}
    return dict(
        by_bucket.get(
            expectancy_bucket_key(asset_class, regime, dominant_generator),
            {},
        )
    )


@dataclass(frozen=True)
class GeneratorDemotionDecision:
    action: str
    dominant_generator: str
    trades: int
    avg_pnl_bps: float
    signal_multiplier: float = 1.0
    confidence_multiplier: float = 1.0
    ledger_key: str = ""


def evaluate_generator_demotion(
    ledger: Optional[Mapping[str, Any]],
    *,
    asset_class: str,
    regime: str,
    generator_signals: Optional[Mapping[str, Any]],
    enabled: bool,
    min_trades: int,
    block_pnl_bps: float,
    signal_multiplier: float,
    confidence_multiplier: float,
) -> GeneratorDemotionDecision:
    dominant_generator = select_dominant_generator(generator_signals)
    ledger_key = expectancy_bucket_key(asset_class, regime, dominant_generator)
    if not enabled:
        return GeneratorDemotionDecision(
            action="none",
            dominant_generator=dominant_generator,
            trades=0,
            avg_pnl_bps=0.0,
            ledger_key=ledger_key,
        )

    bucket = lookup_expectancy_bucket(
        ledger,
        asset_class=asset_class,
        regime=regime,
        dominant_generator=dominant_generator,
    )
    trades = int(bucket.get("trades", 0) or 0)
    avg_pnl_bps = float(bucket.get("avg_pnl_bps", 0.0) or 0.0)
    if trades < max(1, int(min_trades)) or avg_pnl_bps >= 0.0:
        return GeneratorDemotionDecision(
            action="none",
            dominant_generator=dominant_generator,
            trades=trades,
            avg_pnl_bps=avg_pnl_bps,
            ledger_key=ledger_key,
        )

    if avg_pnl_bps <= float(block_pnl_bps):
        return GeneratorDemotionDecision(
            action="block",
            dominant_generator=dominant_generator,
            trades=trades,
            avg_pnl_bps=avg_pnl_bps,
            ledger_key=ledger_key,
        )

    return GeneratorDemotionDecision(
        action="scale",
        dominant_generator=dominant_generator,
        trades=trades,
        avg_pnl_bps=avg_pnl_bps,
        signal_multiplier=float(signal_multiplier),
        confidence_multiplier=float(confidence_multiplier),
        ledger_key=ledger_key,
    )


def should_block_no_trade_band(
    *,
    signal: float,
    effective_signal_threshold: float,
    slope: Optional[float],
    band_ratio: float,
    max_slope: float,
) -> bool:
    if slope is None:
        return False
    threshold = abs(float(effective_signal_threshold))
    abs_signal = abs(float(signal))
    if threshold <= 0.0 or abs_signal < threshold:
        return False
    upper_band = threshold * (1.0 + max(0.0, float(band_ratio)))
    if abs_signal >= upper_band:
        return False
    return float(slope) <= float(max_slope)


def get_expectancy_sizing_multiplier(
    bucket: Optional[Mapping[str, Any]],
    *,
    min_trades: int,
    loss_floor_bps: float,
    size_floor: float,
) -> float:
    if not isinstance(bucket, Mapping):
        return 1.0
    trades = int(bucket.get("trades", 0) or 0)
    avg_pnl_bps = float(bucket.get("avg_pnl_bps", 0.0) or 0.0)
    if trades < max(1, int(min_trades)) or avg_pnl_bps >= 0.0:
        return 1.0

    penalty_span = max(1e-6, abs(float(loss_floor_bps)))
    loss_ratio = min(1.0, abs(avg_pnl_bps) / penalty_span)
    floor = min(1.0, max(0.1, float(size_floor)))
    multiplier = 1.0 - ((1.0 - floor) * loss_ratio)
    return max(floor, round(multiplier, 4))
