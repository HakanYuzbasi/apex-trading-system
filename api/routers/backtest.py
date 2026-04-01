import uuid
import time
import logging
import importlib
from functools import lru_cache
from types import ModuleType
from datetime import datetime, timedelta, timezone
from fastapi import APIRouter, HTTPException, BackgroundTasks
from fastapi.responses import FileResponse
from pydantic import BaseModel

from config import ApexConfig

logger = logging.getLogger("api.backtest")

router = APIRouter(prefix="/backtest", tags=["Backtest"])

BACKTEST_REPORTS_DIR = ApexConfig.DATA_DIR / "backtests" / "reports"
BACKTEST_REPORTS_DIR.mkdir(parents=True, exist_ok=True)

class BacktestRunRequest(BaseModel):
    start_date: str = "2023-01-01"
    end_date: str = "2023-12-31"
    position_size_usd: float = 5000.0
    use_mock_data: bool = True


@lru_cache(maxsize=1)
def _load_backtest_runtime() -> tuple[ModuleType, ModuleType, type]:
    """Import heavy backtest dependencies only when the API endpoint is used."""
    pandas_module = importlib.import_module("pandas")
    numpy_module = importlib.import_module("numpy")
    backtester_module = importlib.import_module("backtesting.advanced_backtester")
    return pandas_module, numpy_module, backtester_module.AdvancedBacktester


class MockSignalGenerator:
    def generate_ml_signal(self, symbol, prices):
        _, numpy_module, _ = _load_backtest_runtime()
        return {"signal": float(numpy_module.random.randn() * 0.5)}

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
        pandas_module, numpy_module, advanced_backtester_cls = _load_backtest_runtime()

        # Generate mock data
        numpy_module.random.seed(42)
        dates = pandas_module.date_range(request.start_date, request.end_date, freq='D')
        
        data = {}
        for symbol in ['AAPL', 'MSFT', 'GOOGL']:
            close = 100 + numpy_module.random.randn(len(dates)).cumsum()
            data[symbol] = pandas_module.DataFrame({
                'Open': close + numpy_module.random.randn(len(dates)) * 0.5,
                'High': close + abs(numpy_module.random.randn(len(dates))),
                'Low': close - abs(numpy_module.random.randn(len(dates))),
                'Close': close,
                'Volume': numpy_module.random.randint(1000000, 10000000, len(dates))
            }, index=dates)

        backtester = advanced_backtester_cls(initial_capital=100000)
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
