import asyncio
import logging
import sys
import os

sys.path.append("/Users/hakanyuzbasioglu/apex-trading")

from execution.alpaca_connector import AlpacaConnector

logging.basicConfig(level=logging.DEBUG)

async def main():
    connector = AlpacaConnector()
    print("Connecting...")
    await connector.connect()
    
    print("Starting stream_quotes...")
    await connector.stream_quotes(["CRYPTO:BTC/USD", "CRYPTO:ETH/USD"])
    
    print("Waiting 1 second...")
    await asyncio.sleep(1)
    
    print("Getting portfolio value...")
    val = await connector.get_portfolio_value()
    print(f"Portfolio value: {val}")
    
    await connector.disconnect()

if __name__ == "__main__":
    asyncio.run(main())
