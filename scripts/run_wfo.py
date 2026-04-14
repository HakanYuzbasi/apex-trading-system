from __future__ import annotations

import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from quant_system.analytics.optimizer import GridSearchOptimizer, OptimizationConfig
from quant_system.analytics.wfo import WalkForwardOptimizer
from quant_system.strategies.pairs_stat_arb import PairsStatArbStrategy


def main() -> None:
    end_ts = datetime.now(timezone.utc)
    start_ts = end_ts - timedelta(days=182)
    instrument_ids = ("AAPL", "MSFT")

    parameter_grid = {
        "instrument_a": [instrument_ids[0]],
        "instrument_b": [instrument_ids[1]],
        "lookback_window": [20, 50, 100],
        "entry_z_score": [1.5, 2.0, 2.5],
        "exit_z_score": [0.0, 0.5],
        "leg_notional": [5_000.0],
    }

    optimizer = WalkForwardOptimizer(
        GridSearchOptimizer,
        PairsStatArbStrategy,
        parameter_grid,
        base_config=OptimizationConfig(
            instrument_ids=instrument_ids,
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

    print()
    print(f"WALK-FORWARD PAIRS STRATEGY: {instrument_ids[0]} / {instrument_ids[1]}")
    print()
    print("WALK-FORWARD WINDOW PARAMETER HISTORY")
    print(result.window_results.to_string(index=False))
    print()
    print(
        "Before running this, make sure your TimescaleDB contains historical bars for both symbols. "
        "If you only loaded AAPL previously, update and rerun "
        "scripts/load_alpaca_history.py for MSFT first."
    )


if __name__ == "__main__":
    main()
