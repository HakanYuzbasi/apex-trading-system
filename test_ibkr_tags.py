import asyncio
from ib_insync import IB
import sys

async def main():
    ib = IB()
    for port in [4001, 7497, 4002]:
        try:
            print(f"Trying port {port}...")
            await ib.connectAsync('127.0.0.1', port, 999)
            await asyncio.sleep(2)
            avs = ib.accountValues()
            print("--- IB Account Values ---")
            for av in avs:
                if av.currency == 'USD':
                    print(f"Tag: {av.tag}, Value: {av.value}")
            ib.disconnect()
            sys.exit(0)
        except Exception as e:
            print(f"Port {port} failed:", e)

asyncio.run(main())
