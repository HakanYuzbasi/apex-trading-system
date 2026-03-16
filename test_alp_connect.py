import asyncio
import logging
import sys
import os

sys.path.append("/Users/hakanyuzbasioglu/apex-trading")

from execution.alpaca_connector import AlpacaConnector

logging.basicConfig(level=logging.DEBUG)

async def main():
    connector = AlpacaConnector()
    print("Initialising connector...")
    try:
        await connector.connect()
        print("Connected.")
        val = await connector.get_portfolio_value()
        print(f"Portfolio value: {val}")
    except Exception as e:
        print(f"Exception: {e}")

if __name__ == "__main__":
    asyncio.run(main())
