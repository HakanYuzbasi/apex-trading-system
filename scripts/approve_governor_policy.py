#!/usr/bin/env python3
"""Approve a staged governor policy for production activation."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from config import ApexConfig
from risk.governor_policy import GovernorPolicyRepository, PolicyPromotionService

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="Approve staged governor policy")
    parser.add_argument("--policy-id", required=True, help="Policy id: ASSET:regime:version")
    parser.add_argument("--approver", required=True, help="Approver name/email")
    parser.add_argument("--environment", default=ApexConfig.ENVIRONMENT)
    parser.add_argument("--live-trading", action="store_true", default=ApexConfig.LIVE_TRADING)
    parser.add_argument("--data-dir", default=str(ApexConfig.DATA_DIR))
    args = parser.parse_args()

    repo = GovernorPolicyRepository(Path(args.data_dir) / "governor_policies")
    service = PolicyPromotionService(
        repository=repo,
        environment=args.environment,
        live_trading=args.live_trading,
        auto_promote_non_prod=ApexConfig.GOVERNOR_AUTO_PROMOTE_NON_PROD,
    )
    decision = service.approve_staged(policy_id=args.policy_id, approver=args.approver)
    if decision.accepted:
        logger.info("Approved: %s", decision.reason)
    else:
        logger.error("Approval failed: %s", decision.reason)


if __name__ == "__main__":
    main()
