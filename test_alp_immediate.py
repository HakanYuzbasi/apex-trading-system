import asyncio
import logging
import sys
import os

sys.path.append("/Users/hakanyuzbasioglu/apex-trading")

from execution.alpaca_connector import AlpacaConnector

logging.basicConfig(level=logging.INFO)

async def main():
    connector = AlpacaConnector()
    print("Connecting...")
    await connector.connect()
    
    print("Getting portfolio value immediately...")
    val = await connector.get_portfolio_value()
    print(f"Portfolio value: {val}")

if __name__ == "__main__":
    asyncio.run(main())
