from pydantic import BaseModel, Field
from typing import List, Optional, Dict
from datetime import datetime
from enum import Enum

class BenchmarkType(str, Enum):
    VWAP = \"VWAP\"
    TWAP = \"TWAP\"
    ARRIVAL = \"ARRIVAL\"
    CLOSE = \"CLOSE\"
    OPEN = \"OPEN\"

class TradeExecution(BaseModel):
    trade_id: str
    symbol: str
    side: str
    quantity: float
    execution_price: float
    arrival_price: float
    benchmark_price: float
    timestamp: datetime
    fees: float = 0.0
    venue: str = \"OTC\"

class SlippageMetrics(BaseModel):
    arrival_slippage_bps: float
    benchmark_slippage_bps: float
    total_cost_bps: float
    opportunity_cost_bps: Optional[float] = None

class MarketImpactEstimate(BaseModel):
    estimated_impact_bps: float
    participation_rate: float
    market_volatility: float

class TCAReport(BaseModel):
    report_id: str
    symbol: str
    period_start: datetime
    period_end: datetime
    total_volume: float
    avg_slippage_bps: float
    execution_efficiency: float
    detailed_metrics: List[SlippageMetrics]
    market_impact: MarketImpactEstimate
    recommendations: List[str]
