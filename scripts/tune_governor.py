#!/usr/bin/env python3
"""
scripts/tune_governor.py

Run governor walk-forward tuning and submit candidate policies.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Dict, Iterable

import pandas as pd

from backtesting.governor_walkforward import WalkForwardTuningConfig, tune_policies
from config import ApexConfig
from monitoring.signal_outcome_tracker import SignalOutcomeTracker
from risk.governor_policy import (
    GovernorPolicyRepository,
    PolicyPromotionService,
)

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


DEFAULT_REGIMES: Dict[str, Iterable[str]] = {
    "EQUITY": ("default", "risk_on", "risk_off", "volatile"),
    "FOREX": ("default", "carry", "carry_crash", "volatile"),
    "CRYPTO": ("default", "trend", "crash", "high_vol"),
}


def _crypto_is_unstable(df: pd.DataFrame) -> bool:
    if df.empty:
        return False
    local = df[df["symbol"].map(lambda s: "CRYPTO" if "/" in str(s) else "EQUITY") == "CRYPTO"]
    if len(local) < 50:
        return False
    returns = pd.to_numeric(local["return_10d"], errors="coerce").dropna()
    if len(returns) < 20:
        return False
    return float(returns.std()) > 0.08


def main() -> None:
    parser = argparse.ArgumentParser(description="Tune governor policies via walk-forward.")
    parser.add_argument("--cadence", choices=["weekly", "daily"], default=ApexConfig.GOVERNOR_TUNE_CADENCE_DEFAULT)
    parser.add_argument("--data-dir", default=str(ApexConfig.DATA_DIR))
    parser.add_argument("--environment", default=ApexConfig.ENVIRONMENT)
    parser.add_argument("--live-trading", action="store_true", default=ApexConfig.LIVE_TRADING)
    args = parser.parse_args()

    tracker = SignalOutcomeTracker(data_dir=args.data_dir)
    signal_df = tracker.get_signals_for_ml()
    if signal_df.empty:
        logger.warning("No completed signal outcomes found; skipping governor tuning.")
        return

    cadence = args.cadence
    if cadence == "weekly" and _crypto_is_unstable(signal_df):
        logger.warning("Detected crypto instability; escalating cadence to daily for CRYPTO policies")

    tuning_config = WalkForwardTuningConfig(cadence=cadence)
    policies = tune_policies(signal_df=signal_df, config=tuning_config, regimes_by_asset_class=DEFAULT_REGIMES)
    if not policies:
        logger.warning("No tunable policies generated from current signal outcomes.")
        return

    repository = GovernorPolicyRepository(Path(args.data_dir) / "governor_policies")
    promotion = PolicyPromotionService(
        repository=repository,
        environment=args.environment,
        live_trading=args.live_trading,
        auto_promote_non_prod=ApexConfig.GOVERNOR_AUTO_PROMOTE_NON_PROD,
    )

    accepted = 0
    for policy in policies:
        decision = promotion.submit_candidate(policy)
        logger.info(
            "Policy %s -> %s (accepted=%s, manual=%s, reason=%s)",
            policy.policy_id(),
            decision.status.value,
            decision.accepted,
            decision.manual_approval_required,
            decision.reason,
        )
        if decision.accepted:
            accepted += 1

    logger.info("Governor tuning complete: %d candidates accepted out of %d", accepted, len(policies))


if __name__ == "__main__":
    main()
