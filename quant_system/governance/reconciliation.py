"""
quant_system/governance/reconciliation.py
================================================================================
PHYSICAL BROKER RECONCILIATION DAEMON
================================================================================
"""

import os
import logging
import asyncio
import aiohttp

risk_logger = logging.getLogger('RiskManager')

class PositionReconciler:
    def __init__(self, engine_ref):
        self.api_key = os.environ.get("APCA_API_KEY_ID", "MISSING_KEY")
        self.api_secret = os.environ.get("APCA_API_SECRET_KEY", "MISSING_SECRET")
        self.base_url = "https://paper-api.alpaca.markets/v2"
        self.engine = engine_ref # Direct state access required mathematically protecting drift targets

    async def sync_ledger(self, target_symbol: str = "BTCUSD"):
        """
        Background AsyncIO Daemon actively polling exchange values tracking mathematical integrity 
        preventing desynchronization traps.
        """
        headers = {
            "APCA-API-KEY-ID": self.api_key,
            "APCA-API-SECRET-KEY": self.api_secret
        }
        
        while True:
            await asyncio.sleep(60) # 60s Polling delay prevents rate limits mapping API limits safely
            
            try:
                async with aiohttp.ClientSession() as session:
                    async with session.get(f"{self.base_url}/positions/{target_symbol}", headers=headers) as resp:
                        if resp.status == 200:
                            pos_data = await resp.json()
                            broker_qty = float(pos_data.get('qty', 0.0))
                        elif resp.status == 404: # Target asset implicitly closed/zero
                            broker_qty = 0.0 
                        else:
                            risk_logger.warning(f"Reconciliation Query Blocked: HTTP {resp.status}")
                            continue
                            
                # Engine strictly controls the mathematical truth value locally    
                internal_qty = self.engine.asset_quantity
                
                drift = 0.0
                if max(internal_qty, broker_qty) > 0:
                    drift = abs(internal_qty - broker_qty) / max(internal_qty, broker_qty)
                    
                if drift > 0.01:
                    risk_logger.critical(f"DESYNC DETECTED: Internal Node [{internal_qty:.6f}] vs Broker L2 [{broker_qty:.6f}] | Drift Deviation: {drift:.2%}")
                else:
                    risk_logger.info(f"Reconciliation Clean: Internal Ledger [{internal_qty:.6f}] == Broker State [{broker_qty:.6f}]")
                    
            except Exception as e:
                risk_logger.error(f"Reconciliation Daemon Execution Trap -> {e}")
