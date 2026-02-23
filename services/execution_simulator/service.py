"""
Execution Simulator Service - compare execution costs across algo types.

Reuses SmartOrderRouter.select_algorithm and MarketImpactModel.calculate_execution_costs.
"""

import logging
from typing import Any, Dict

logger = logging.getLogger(__name__)

try:
    from execution.smart_order_router import SmartOrderRouter
    from backtesting.market_impact import MarketImpactModel, MarketConditions
    ROUTER_AVAILABLE = True
except ImportError:
    ROUTER_AVAILABLE = False


# Default market conditions when no live data
def _default_conditions(daily_volume: float = 2_000_000) -> "MarketConditions":
    return MarketConditions(
        avg_daily_volume=daily_volume,
        avg_daily_turnover=daily_volume * 300.0,
        volatility=0.02,
        bid_ask_spread_bps=5.0,
        current_volume_ratio=1.0,
    )


def simulate_execution(
    symbol: str,
    quantity: int,
    urgency: str = "medium",
    time_horizon_minutes: int | None = None,
    reference_price: float = 100.0,
    daily_volume: float = 2_000_000,
) -> Dict[str, Any]:
    """
    Simulate execution costs for different algo types.
    Returns dict with results list and best_algorithm.
    """
    if not ROUTER_AVAILABLE:
        return _fallback_simulate(symbol, quantity, reference_price)

    router = SmartOrderRouter()
    impact_model = MarketImpactModel(
        base_spread_bps=5.0,
        impact_multiplier=1.0,
        random_slippage_std=2.0,
    )
    conditions = _default_conditions(daily_volume)
    notional = quantity * reference_price
    side = "BUY"

    # Map urgency to algo types we want to compare
    algos = [
        ("MARKET", "critical", 0.1),
        ("TWAP", "low", time_horizon_minutes or 30),
        ("VWAP", "medium", 60),
        ("POV", "medium", time_horizon_minutes or 120),
        ("ICEBERG", "low", 45),
    ]

    # Base cost from market impact model (single evaluation)
    try:
        costs = impact_model.calculate_execution_costs(
            order_size_shares=quantity,
            price=reference_price,
            side=side,
            conditions=conditions,
        )
        base_bps = costs.total_cost_bps
        base_impact = costs.temporary_impact_bps + costs.permanent_impact_bps
    except Exception:
        base_bps = 10.0 + (quantity / daily_volume) ** 0.5 * 20
        base_impact = base_bps * 0.7

    # Algo-specific multipliers (spread execution = lower impact)
    algo_multipliers = {"MARKET": 1.8, "TWAP": 0.65, "VWAP": 0.75, "POV": 0.7, "ICEBERG": 0.8}
    results = []
    for algo_name, urg, default_min in algos:
        rec = router.select_algorithm(
            symbol=symbol,
            side=side,
            quantity=quantity,
            urgency=urg,
            daily_volume=daily_volume,
        )
        duration = rec.get("estimated_duration_minutes", default_min or 30)
        mult = algo_multipliers.get(algo_name, 1.0)
        cost_bps = base_bps * mult
        impact_bps = base_impact * mult
        cost_usd = notional * (cost_bps / 10000.0)
        results.append({
            "algorithm": algo_name,
            "estimated_cost_bps": round(cost_bps, 2),
            "estimated_cost_usd": round(cost_usd, 2),
            "time_to_complete_minutes": round(duration, 1),
            "market_impact_bps": round(impact_bps, 2),
            "recommendation": rec.get("reason"),
        })

    # Best = lowest cost
    best = min(results, key=lambda x: x["estimated_cost_bps"])
    return {
        "symbol": symbol,
        "quantity": quantity,
        "side": side,
        "reference_price": reference_price,
        "notional_usd": notional,
        "results": results,
        "best_algorithm": best["algorithm"],
    }


def _fallback_simulate(symbol: str, quantity: int, reference_price: float) -> Dict[str, Any]:
    """Fallback when execution modules are not available."""
    notional = quantity * reference_price
    # Simple heuristic: cost scales with sqrt(quantity/1e6)
    import math
    base_bps = 5 + math.sqrt(quantity / 1000) * 2
    results = [
        {"algorithm": "MARKET", "estimated_cost_bps": base_bps * 2, "estimated_cost_usd": notional * base_bps * 2 / 10000, "time_to_complete_minutes": 0.1, "market_impact_bps": base_bps * 1.5, "recommendation": "Immediate execution"},
        {"algorithm": "TWAP", "estimated_cost_bps": base_bps * 0.8, "estimated_cost_usd": notional * base_bps * 0.8 / 10000, "time_to_complete_minutes": 30, "market_impact_bps": base_bps * 0.6, "recommendation": "Time-weighted"},
        {"algorithm": "VWAP", "estimated_cost_bps": base_bps * 0.9, "estimated_cost_usd": notional * base_bps * 0.9 / 10000, "time_to_complete_minutes": 60, "market_impact_bps": base_bps * 0.7, "recommendation": "Volume-weighted"},
    ]
    best = min(results, key=lambda x: x["estimated_cost_bps"])
    return {
        "symbol": symbol,
        "quantity": quantity,
        "side": "BUY",
        "reference_price": reference_price,
        "notional_usd": notional,
        "results": results,
        "best_algorithm": best["algorithm"],
    }
