"""
TCA Service - wraps monitoring.tca_report.build_tca_report for API use.
"""

import logging
from pathlib import Path
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


class TCAService:
    """Thin wrapper around build_tca_report with a configurable data_dir."""

    def __init__(self, data_dir: Optional[str] = None):
        self.data_dir = Path(data_dir) if data_dir else None

    def get_report(self) -> Dict[str, Any]:
        """Build and return the full TCA report dict."""
        from monitoring.tca_report import build_tca_report

        return build_tca_report(data_dir=self.data_dir)

    def get_status(self) -> Dict[str, Any]:
        """Return a condensed execution health summary."""
        report = self.get_report()
        summary = report["summary"]
        return {
            "execution_health_score": summary["execution_health_score"],
            "closed_trades": summary["closed_trades"],
            "win_rate_pct": summary["win_rate_pct"],
            "total_net_pnl": summary["total_net_pnl"],
            "total_rejections": summary["total_rejections"],
        }
