#!/usr/bin/env python3
"""
scripts/run_daily_social_risk_cycle.py

Daily social-risk runner:
1) Build normalized social ingestion contract
2) Calibrate social shock policy thresholds
"""

from __future__ import annotations

import argparse
import json
import logging
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import ApexConfig


logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


def _run(cmd: List[str]) -> None:
    logger.info("running: %s", " ".join(cmd))
    proc = subprocess.run(cmd, cwd=PROJECT_ROOT, capture_output=True, text=True)
    if proc.stdout:
        logger.info(proc.stdout.strip())
    if proc.stderr:
        logger.warning(proc.stderr.strip())
    if proc.returncode != 0:
        raise RuntimeError(f"command failed ({proc.returncode}): {' '.join(cmd)}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Run daily social-risk ingestion + calibration cycle.")
    parser.add_argument("--data-dir", default=str(ApexConfig.DATA_DIR))
    parser.add_argument("--artifacts-dir", default="artifacts/social-risk-daily")
    parser.add_argument("--freshness-sla-seconds", type=int, default=1800)
    parser.add_argument("--activate", action="store_true", default=True)
    parser.add_argument("--no-activate", dest="activate", action="store_false")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    artifacts_dir = Path(args.artifacts_dir)
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    py = sys.executable
    _run(
        [
            py,
            "scripts/build_social_risk_inputs.py",
            "--data-dir",
            str(data_dir),
            "--freshness-sla-seconds",
            str(max(60, int(args.freshness_sla_seconds))),
        ]
    )
    calibration_cmd = [
        py,
        "scripts/calibrate_social_shock_thresholds.py",
        "--data-dir",
        str(data_dir),
    ]
    if not args.activate:
        calibration_cmd.append("--no-activate")
    _run(calibration_cmd)

    social_inputs_path = data_dir / "social_risk_inputs.json"
    social_inputs: Dict[str, object] = {}
    if social_inputs_path.exists():
        social_inputs = json.loads(social_inputs_path.read_text(encoding="utf-8"))

    policies_dir = data_dir / "governor_policies"
    snapshots = sorted(policies_dir.glob("social_shock_policy_*.json"))
    latest_snapshot = snapshots[-1].name if snapshots else None
    active_path = policies_dir / "social_shock_active.json"

    summary = {
        "ran_at": datetime.utcnow().isoformat(),
        "data_dir": str(data_dir),
        "social_inputs_file": str(social_inputs_path),
        "social_inputs_validation": social_inputs.get("validation", {}),
        "latest_snapshot": latest_snapshot,
        "active_snapshot_present": active_path.exists(),
        "activate_requested": bool(args.activate),
    }
    summary_path = artifacts_dir / "social_risk_daily_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    logger.info("daily social-risk cycle completed. summary=%s", summary_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
