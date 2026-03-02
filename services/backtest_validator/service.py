"""
Backtest Validator Service - runs robustness validation on uploaded equity/backtest results.

Reuses:
- Monte Carlo logic (bootstrap on returns)
- PortfolioStressTest for scenario analysis
- PDF report generation
"""

import logging
import os
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np

from services.common.file_upload import ParsedUpload, validate_equity_curve
from services.common.pdf_report import PDFReport, is_available as pdf_available
from services.common.schemas import JobStatus

logger = logging.getLogger(__name__)

# Optional imports for existing Apex modules
try:
    from risk.portfolio_stress_test import PortfolioStressTest
    STRESS_AVAILABLE = True
except ImportError:
    STRESS_AVAILABLE = False


def run_monte_carlo_on_returns(returns: np.ndarray, n_sims: int = 500, start_equity: float = 1.0) -> Dict[str, float]:
    """Bootstrap Monte Carlo on return series. Returns metrics dict."""
    if returns is None or len(returns) < 10:
        return {}
    returns = np.asarray(returns, dtype=float)
    days = len(returns)
    final_values = []
    for _ in range(n_sims):
        sim_returns = np.random.choice(returns, size=days, replace=True)
        cum = np.cumprod(1.0 + sim_returns)
        final_values.append(start_equity * cum[-1])
    final_values = np.array(final_values)
    sorted_outcomes = np.sort(final_values)
    return {
        "mc_min_equity": float(sorted_outcomes[0]),
        "mc_median_equity": float(np.median(final_values)),
        "mc_95_pct_equity": float(sorted_outcomes[max(0, int(n_sims * 0.05))]),
        "mc_99_pct_equity": float(sorted_outcomes[max(0, int(n_sims * 0.01))]),
        "mc_max_equity": float(sorted_outcomes[-1]),
    }


def run_stress_tests(last_equity: float) -> Dict[str, Any]:
    """Run portfolio stress scenarios on a synthetic single-asset portfolio."""
    if not STRESS_AVAILABLE or last_equity <= 0:
        return {}
    try:
        # Synthetic portfolio: one "asset" with value = last_equity
        stress = PortfolioStressTest(
            positions={"PORTFOLIO": 1},
            prices={"PORTFOLIO": last_equity},
        )
        results = stress.run_all_scenarios()
        summary = {}
        for sid, res in results.items():
            summary[sid] = {
                "portfolio_return": getattr(res, "portfolio_return", 0),
                "portfolio_pnl": getattr(res, "portfolio_pnl", 0),
                "max_drawdown": getattr(res, "max_drawdown", 0),
            }
        return summary
    except Exception as e:
        logger.warning("Stress test failed: %s", e)
        return {}


def compute_robustness_score(
    monte_carlo: Dict[str, float],
    stress_summary: Dict[str, Any],
    max_drawdown: float,
) -> float:
    """Compute a 0-100 robustness score from MC and stress results."""
    score = 50.0
    if monte_carlo:
        # Prefer higher 5th percentile relative to median
        mc_95 = monte_carlo.get("mc_95_pct_equity") or 0
        mc_median = monte_carlo.get("mc_median_equity") or 1
        if mc_median > 0:
            ratio = mc_95 / mc_median
            score += min(25, max(-25, (ratio - 0.8) * 100))
    if max_drawdown > 0:
        # Penalize large drawdowns (max_drawdown is a fraction, so multiply by 100 for penalty logic)
        score -= min(30, (max_drawdown * 100) / 2)
    if stress_summary:
        worst_return = 0
        for v in stress_summary.values():
            if isinstance(v, dict):
                r = v.get("portfolio_return", 0)
                if r < worst_return:
                    worst_return = r
        score += min(20, max(-20, worst_return * 100))
    return max(0.0, min(100.0, round(score, 1)))


def build_validation_pdf(
    job_id: str,
    robustness_score: float,
    monte_carlo: Dict[str, float],
    stress_summary: Dict[str, Any],
    equity_rows: int,
    max_drawdown: float,
) -> Optional[bytes]:
    """Generate PDF report bytes."""
    if not pdf_available():
        return None
    report = PDFReport(title="Backtest Validation Report")
    report.add_header("Validation summary")
    report.add_key_value_section("Overview", {
        "Job ID": job_id,
        "Robustness Score": f"{robustness_score:.1f} / 100",
        "Equity data points": equity_rows,
        "Max drawdown (approx %)": f"{max_drawdown * 100:.2f}%",
    })
    if monte_carlo:
        report.add_header("Monte Carlo (bootstrap)")
        report.add_key_value_section("Metrics", {k: f"{v:.2f}" for k, v in monte_carlo.items()})
    if stress_summary:
        report.add_header("Stress test summary")
        rows = []
        for sid, data in list(stress_summary.items())[:8]:
            ret = data.get("portfolio_return", 0)
            rows.append([sid, f"{ret:.2%}"])
        report.add_table(headers=["Scenario", "Portfolio return"], rows=rows)
    return report.build()


class BacktestValidatorService:
    """Orchestrates validation: parse upload -> MC -> stress -> score -> PDF -> job record."""

    def __init__(self, jobs_dir: Optional[Path] = None):
        self.jobs_dir = jobs_dir or Path(os.getenv("APEX_JOBS_DIR", "data/saas_jobs"))
        self.jobs_dir.mkdir(parents=True, exist_ok=True)

    def validate(
        self,
        upload: ParsedUpload,
        user_id: str,
        db_session,
        job_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Run full validation pipeline. Creates a service_job record and optionally saves PDF.
        """
        from services.auth.models import ServiceJobModel

        job_id = job_id or str(uuid.uuid4())
        job = ServiceJobModel(
            id=job_id,
            user_id=user_id,
            feature_key="backtest_validator",
            status=JobStatus.RUNNING,
            input_params={"filename": upload.filename, "row_count": upload.row_count},
        )
        db_session.add(job)
        try:
            equity = validate_equity_curve(upload)
        except ValueError as e:
            job.status = JobStatus.FAILED
            job.error_message = str(e)
            job.completed_at = datetime.utcnow()
            return {"job_id": job_id, "status": JobStatus.FAILED, "error": str(e)}

        try:
            values = equity.values
            returns = np.array(equity.returns) if equity.returns else np.diff(values) / (np.array(values[:-1]) + 1e-12)
            if len(returns) < 10:
                job.status = JobStatus.FAILED
                job.error_message = "Insufficient data points for validation"
                job.completed_at = datetime.utcnow()
                return {"job_id": job_id, "status": JobStatus.FAILED, "error": job.error_message}

            start_equity = values[0] if values[0] > 0 else 1.0
            last_equity = values[-1]
            monte_carlo = run_monte_carlo_on_returns(returns, n_sims=500, start_equity=start_equity)
            stress_summary = run_stress_tests(last_equity)

            peak = values[0]
            max_dd = 0.0
            for v in values:
                if v > peak:
                    peak = v
                dd = (peak - v) / peak if peak > 0 else 0
                if dd > max_dd:
                    max_dd = dd
            max_dd_pct = max_dd * 100

            robustness = compute_robustness_score(monte_carlo, stress_summary, max_dd)

            pdf_bytes = build_validation_pdf(
                job_id, robustness, monte_carlo, stress_summary, len(values), max_dd
            )
            pdf_path = None
            if pdf_bytes:
                pdf_path = self.jobs_dir / f"{job_id}.pdf"
                pdf_path.write_bytes(pdf_bytes)
                pdf_path = str(pdf_path)

            result_summary = {
                "robustness_score": robustness,
                "monte_carlo": monte_carlo,
                "stress_test_summary": stress_summary,
                "max_drawdown": max_dd,
                "equity_points": len(values),
            }
            job.status = JobStatus.COMPLETED
            job.result_summary = result_summary
            job.result_file_path = pdf_path
            job.completed_at = datetime.utcnow()

            return {
                "job_id": job_id,
                "status": JobStatus.COMPLETED,
                "robustness_score": robustness,
                "monte_carlo": monte_carlo,
                "stress_test_summary": stress_summary,
                "pdf_url": f"/api/v1/backtest-validator/jobs/{job_id}/report" if pdf_path else None,
            }
        except Exception as e:
            logger.exception("Validation failed")
            job.status = JobStatus.FAILED
            job.error_message = str(e)
            job.completed_at = datetime.utcnow()
            return {"job_id": job_id, "status": JobStatus.FAILED, "error": str(e)}
