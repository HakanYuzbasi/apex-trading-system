
import asyncio
from config import ApexConfig
from execution.ibkr_connector import IBKRConnector

async def check_positions():
    print("Connecting to IBKR...")
    ibkr = IBKRConnector(
        host=ApexConfig.IBKR_HOST,
        port=ApexConfig.IBKR_PORT,
        client_id=ApexConfig.IBKR_CLIENT_ID + 99 # Use different ID to avoid conflict
    )
    
    try:
        await ibkr.connect()
        await asyncio.sleep(2)
        
        positions = await ibkr.get_all_positions()
        print(f"Total Open Positions: {len(positions)}")
        
        off_universe = []
        for symbol in positions.keys():
            if symbol not in ApexConfig.SYMBOLS:
                off_universe.append(symbol)
                
        print(f"Positions NOT in SYMBOLS list: {len(off_universe)}")
        if off_universe:
            print(f"Off-universe symbols: {off_universe}")
            
        print(f"Configured SYMBOLS count: {len(ApexConfig.SYMBOLS)}")
        print(f"Total Potential Streams (SYMBOLS + Off-Universe): {len(ApexConfig.SYMBOLS) + len(off_universe)}")
        
    except Exception as e:
        print(f"Error: {e}")
    finally:
        ibkr.disconnect()

if __name__ == "__main__":
    asyncio.run(check_positions())
