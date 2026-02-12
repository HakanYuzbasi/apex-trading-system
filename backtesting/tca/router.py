from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks
from typing import List
from .schemas import TradeExecution, SlippageMetrics, MarketImpactEstimate, TCAReport
from .service import TCAService
from api.auth import get_current_user

router = APIRouter(prefix=\"/tca\", tags=[\"Transaction Cost Analysis\"])
tca_service = TCAService()

@router.post(\"/analyze-trade\", response_model=SlippageMetrics)
async def analyze_trade(trade: TradeExecution, current_user: str = Depends(get_current_user)):
    \"\"\"Analyze slippage for a single trade execution.\"\"\"
    return await tca_service.analyze_trade_slippage(trade)

@router.post(\"/estimate-impact\", response_model=MarketImpactEstimate)
async def estimate_impact(
    symbol: str, 
    quantity: float, 
    avg_daily_volume: float,
    current_user: str = Depends(get_current_user)
):
    \"\"\"Estimate market impact for a potential trade.\"\"\"
    return await tca_service.estimate_market_impact(symbol, quantity, avg_daily_volume)

@router.post(\"/generate-report\", response_model=TCAReport)
async def generate_report(trades: List[TradeExecution], current_user: str = Depends(get_current_user)):
    \"\"\"Generate a full TCA report for a batch of trades.\"\"\"
    try:
        return await tca_service.generate_tca_report(trades)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

@router.get(\"/benchmark-types\")
async def get_benchmarks():
    \"\"\"Get supported benchmark types.\"\"\"
    return [\"VWAP\", \"TWAP\", \"Arrival\", \"Close\", \"Open\"]
