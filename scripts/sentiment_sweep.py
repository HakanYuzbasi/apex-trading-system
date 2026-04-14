#!/usr/bin/env python3
import asyncio
import os
import logging
import sys
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from quant_system.risk.sentiment_warden import SentimentWarden

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("sentiment_sweep")

# Active Universe for Sentiment Scanning
UNIVERSE = [
    "AAPL", "MSFT", "NVDA", "AMD", "V", "MA", "KO", "PEP", 
    "WMT", "TGT", "XOM", "CVX", "BTC/USD", "ETH/USD"
]

async def main():
    api_key = os.getenv("ALPACA_API_KEY")
    secret_key = os.getenv("ALPACA_SECRET_KEY")
    gemini_key = os.getenv("GEMINI_API_KEY")

    if not api_key or not secret_key:
        logger.error("ALPACA_API_KEY and ALPACA_SECRET_KEY must be set.")
        sys.exit(1)

    logger.info("🚀 Starting Pre-Flight Sentiment Sweep...")
    
    warden = SentimentWarden(
        api_key=api_key,
        secret_key=secret_key,
        gemini_api_key=gemini_key,
        state_path="run_state/v3/sentiment_vetoes.json"
    )

    await warden.scan_universe(UNIVERSE)
    
    logger.info("✅ Sentiment Sweep Complete.")

if __name__ == "__main__":
    asyncio.run(main())
