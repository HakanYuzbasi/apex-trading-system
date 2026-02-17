#!/usr/bin/env python3
"""
scripts/build_social_risk_inputs.py

Build normalized social ingestion payload and write data/social_risk_inputs.json.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import ApexConfig
from data.social.contract import write_social_risk_inputs


logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


def main() -> int:
    parser = argparse.ArgumentParser(description="Build normalized social risk input contract.")
    parser.add_argument("--data-dir", default=str(ApexConfig.DATA_DIR))
    parser.add_argument("--output", default="")
    parser.add_argument("--freshness-sla-seconds", type=int, default=1800)
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    out = Path(args.output) if args.output.strip() else (data_dir / "social_risk_inputs.json")
    path, _, report = write_social_risk_inputs(
        data_dir=data_dir,
        output_path=out,
        freshness_sla_seconds=max(60, int(args.freshness_sla_seconds)),
    )

    logger.info(
        "social_risk_inputs written: %s (valid=%s usable_feeds=%s sources=%s)",
        path,
        report.valid,
        report.has_usable_feeds,
        report.source_status,
    )
    for issue in report.warnings:
        logger.warning("social_ingestion_warning code=%s msg=%s", issue.code, issue.message)
    for issue in report.errors:
        logger.error("social_ingestion_error code=%s msg=%s", issue.code, issue.message)

    # Contract/schema errors should fail the job. Missing feeds should not.
    return 1 if not report.valid else 0


if __name__ == "__main__":
    raise SystemExit(main())
