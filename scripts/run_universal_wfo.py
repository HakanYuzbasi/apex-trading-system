from __future__ import annotations

import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from quant_system.analytics.optimizer import GridSearchOptimizer, OptimizationConfig
from quant_system.analytics.wfo import WalkForwardOptimizer
from quant_system.strategies.pairs_stat_arb import PairsStatArbStrategy


PAIRS: tuple[tuple[str, str], ...] = (
    ("AAPL", "MSFT"),
    ("NVDA", "AMD"),
    ("KO", "PEP"),
    ("V", "MA"),
    ("JPM", "GS"),
    ("XOM", "CVX"),
)


def main() -> None:
    end_ts = datetime.now(timezone.utc)
    start_ts = end_ts - timedelta(days=365)

    parameter_grid = {
        "lookback_window": [20, 50, 100],
        "entry_z_score": [1.5, 2.0, 2.5],
        "exit_z_score": [0.0, 0.5],
        "leg_notional": [5_000.0],
    }

    rows: list[dict[str, object]] = []

    for instrument_a, instrument_b in PAIRS:
        print()
        print("=" * 88)
        print(f"RUNNING WALK-FORWARD OPTIMIZATION FOR {instrument_a} / {instrument_b}")
        print("=" * 88)

        pair_grid = {
            "instrument_a": [instrument_a],
            "instrument_b": [instrument_b],
            **parameter_grid,
        }
        optimizer = WalkForwardOptimizer(
            GridSearchOptimizer,
            PairsStatArbStrategy,
            pair_grid,
            base_config=OptimizationConfig(
                instrument_ids=(instrument_a, instrument_b),
                start_ts=start_ts,
                end_ts=end_ts,
                include_bars=True,
                include_trade_ticks=False,
                chunk_size=10_000,
                starting_cash=50_000.0,
                simulated_latency_ms=0.0,
                synchronous_broker=True,
            ),
            is_window_days=90,
            oos_window_days=30,
        )

        result = optimizer.run()
        rows.append(
            {
                "pair": f"{instrument_a}/{instrument_b}",
                "instrument_a": instrument_a,
                "instrument_b": instrument_b,
                "oos_total_return_pct": float(result.oos_metrics["total_return_pct"]),
                "oos_max_drawdown_pct": float(result.oos_metrics["max_drawdown_pct"]),
                "oos_annualized_sharpe": float(result.oos_metrics["annualized_sharpe"]),
                "oos_annualized_sortino": float(result.oos_metrics["annualized_sortino"]),
                "wfo_windows": int(len(result.window_results)),
            }
        )

    results = pd.DataFrame(rows)
    if results.empty:
        print("No walk-forward results were produced.")
        return

    leaderboard = results.sort_values("oos_annualized_sharpe", ascending=False, kind="stable").reset_index(drop=True)

    print()
    print("=" * 88)
    print("PAIRS LEADERBOARD")
    print("=" * 88)
    print(leaderboard.to_string(index=False))
    print("=" * 88)


if __name__ == "__main__":
    main()
