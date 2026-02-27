import uuid
import time
import logging
from datetime import datetime, timedelta, timezone
from fastapi import APIRouter, HTTPException, BackgroundTasks
from fastapi.responses import FileResponse
from pydantic import BaseModel
import pandas as pd
import numpy as np

from config import ApexConfig
from backtesting.advanced_backtester import AdvancedBacktester

logger = logging.getLogger("api.backtest")

router = APIRouter(prefix="/backtest", tags=["Backtest"])

BACKTEST_REPORTS_DIR = ApexConfig.DATA_DIR / "backtests" / "reports"
BACKTEST_REPORTS_DIR.mkdir(parents=True, exist_ok=True)

class BacktestRunRequest(BaseModel):
    start_date: str = "2023-01-01"
    end_date: str = "2023-12-31"
    position_size_usd: float = 5000.0
    use_mock_data: bool = True

class MockSignalGenerator:
    def generate_ml_signal(self, symbol, prices):
        return {'signal': np.random.randn() * 0.5}

def cleanup_expired_reports():
    """Background task to clean up reports older than 7 days."""
    try:
        now = time.time()
        retention_seconds = 7 * 24 * 3600
        for report_file in BACKTEST_REPORTS_DIR.glob("*.html"):
            if report_file.is_file():
                mtime = report_file.stat().st_mtime
                if now - mtime > retention_seconds:
                    report_file.unlink()
                    logger.info(f"Cleaned up expired backtest report: {report_file.name}")
    except Exception as e:
        logger.error(f"Error cleaning up expired backtest reports: {e}")

@router.post("/run")
async def run_backtest(
    request: BacktestRunRequest,
    background_tasks: BackgroundTasks,
    # user=Depends(require_user) # TODO: binding report UUID to authenticated user_id in v2
):
    """
    Trigger a backtest and generate a professional static HTML tear sheet.
    Returns the report UUID and URL.
    """
    if not request.use_mock_data:
        raise HTTPException(status_code=501, detail="Historical data backtesting not yet implemented via API.")

    report_id = str(uuid.uuid4())
    output_filename = BACKTEST_REPORTS_DIR / f"{report_id}.html"

    try:
        # Generate mock data
        np.random.seed(42)
        dates = pd.date_range(request.start_date, request.end_date, freq='D')
        
        data = {}
        for symbol in ['AAPL', 'MSFT', 'GOOGL']:
            close = 100 + np.random.randn(len(dates)).cumsum()
            data[symbol] = pd.DataFrame({
                'Open': close + np.random.randn(len(dates)) * 0.5,
                'High': close + abs(np.random.randn(len(dates))),
                'Low': close - abs(np.random.randn(len(dates))),
                'Close': close,
                'Volume': np.random.randint(1000000, 10000000, len(dates))
            }, index=dates)

        backtester = AdvancedBacktester(initial_capital=100000)
        results = backtester.run_backtest(
            data=data,
            signal_generator=MockSignalGenerator(),
            start_date=request.start_date,
            end_date=request.end_date,
            position_size_usd=request.position_size_usd,
            max_positions=5
        )

        success = backtester.generate_tear_sheet(results, str(output_filename))
        if not success:
            raise HTTPException(status_code=500, detail="Failed to generate tear sheet.")

        # Schedule cleanup job to run in background just to occasionally prune
        background_tasks.add_task(cleanup_expired_reports)

        expires_at = datetime.now(timezone.utc) + timedelta(days=7)
        
        return {
            "report_id": report_id,
            "report_url": f"/backtest/reports/{report_id}",
            "expires_at": expires_at.isoformat() + "Z"
        }

    except Exception as e:
        logger.error(f"Backtest run failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/reports/{report_id}")
async def get_report(report_id: str):
    """
    Serve the generated HTML tear sheet directly to the browser.
    """
    # Sanitize inputs to prevent directory traversal
    safe_report_id = "".join(c for c in report_id if c.isalnum() or c == '-')
    report_path = BACKTEST_REPORTS_DIR / f"{safe_report_id}.html"

    if not report_path.exists() or not report_path.is_file():
        raise HTTPException(status_code=404, detail="Report not found or has expired.")

    return FileResponse(
        path=report_path,
        media_type="text/html",
        filename=f"Apex_Tearsheet_{safe_report_id}.html"
    )
