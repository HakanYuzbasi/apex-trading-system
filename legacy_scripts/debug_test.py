import asyncio
from unittest.mock import AsyncMock
from execution.ibkr_adapter import IBKRAdapter

async def test():
    class MockConn:
        def __init__(self):
            # If we assign AsyncMock(side_effect=Exception) to a method...
            self.get_all_positions = AsyncMock(side_effect=Exception("foo"))
    
    adapter = IBKRAdapter(MockConn())
    adapter._failure_threshold = 3
    print("Initial error count:", adapter._error_count)
    
    for i in range(3):
        try:
            await adapter.get_all_positions()
        except Exception as e:
            print(f"Call {i} failed with:", type(e))
        print("Error count:", adapter._error_count)
        print("Circuit open:", adapter._circuit_open)

asyncio.run(test())
