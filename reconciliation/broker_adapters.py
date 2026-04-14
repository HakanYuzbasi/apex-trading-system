from __future__ import annotations

import asyncio
from typing import Dict
from alpaca.trading.client import TradingClient

class AlpacaReconcilerAdapter:
    """
    Adapter that allows PositionReconciler to work with Alpaca TradingClient.
    """
    def __init__(self, trading_client: TradingClient) -> None:
        self._client = trading_client

    async def get_all_positions(self) -> Dict[str, int]:
        """Fetch all positions from Alpaca and convert to {symbol: qty} dict."""
        # Alpaca positions are fetchable via get_all_positions
        # This needs to be run in a thread or use an async client if available.
        # The TradingClient in alpaca-py is synchronous for most methods.
        positions = await asyncio.to_thread(self._client.get_all_positions)
        
        return {
            p.symbol: int(float(p.qty)) 
            for p in positions
        }
    async def get_account_equity(self) -> float:
        """Fetch the current account equity from Alpaca."""
        account = await asyncio.to_thread(self._client.get_account)
        return float(account.equity)
