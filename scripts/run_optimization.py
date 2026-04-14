from __future__ import annotations

import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from quant_system.analytics.optimizer import GridSearchOptimizer, OptimizationConfig
from quant_system.strategies.sma_crossover import SMACrossoverStrategy


def main() -> None:
    end_ts = datetime.now(timezone.utc)
    start_ts = end_ts - timedelta(days=182)

    parameter_grid = {
        "instrument_id": ["AAPL"],
        "short_window": [10, 20, 30],
        "long_window": [50, 80, 100],
        "long_notional": [5_000.0],
    }

    optimizer = GridSearchOptimizer(
        SMACrossoverStrategy,
        parameter_grid,
        optimization_config=OptimizationConfig(
            instrument_ids=("AAPL",),
            start_ts=start_ts,
            end_ts=end_ts,
            include_bars=True,
            include_trade_ticks=False,
            chunk_size=10_000,
            starting_cash=50_000.0,
            simulated_latency_ms=0.0,
            synchronous_broker=True,
        ),
    )

    results = optimizer.run()
    print()
    print("FULL OPTIMIZATION RESULTS")
    print(results.to_string(index=False))


if __name__ == "__main__":
    main()
