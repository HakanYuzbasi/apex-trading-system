"""
quant_system/execution/live_oems.py
================================================================================
PHYSICAL PAPER ORDER EXECUTION ROUTER (OEMS)
================================================================================
"""

import os
import logging
import asyncio
import aiohttp
from datetime import datetime

os.makedirs('quant_system/logs', exist_ok=True)
oems_logger = logging.getLogger('AlpacaOEMS')
oems_logger.setLevel(logging.INFO)

class AlpacaOEMS:
    def __init__(self, log_path: str = "quant_system/logs/shadow_oems_blotter.csv", shadow_mode: bool = True):
        self.log_path = log_path
        self.shadow_mode = shadow_mode
        self.api_key = os.environ.get("APCA_API_KEY_ID", "MISSING_KEY")
        self.api_secret = os.environ.get("APCA_API_SECRET_KEY", "MISSING_SECRET")
        self.base_url = "https://paper-api.alpaca.markets/v2"
        
        if not os.path.exists(self.log_path):
            with open(self.log_path, "w") as f:
                f.write("timestamp,side,symbol,quantity,intended_price,estimated_cost,order_id\n")

    def resolve_action(self, target_allocation_fraction: float, current_physical_quantity: float, current_price: float, total_portfolio_usd: float) -> dict:
        """Translates mathematical targets strictly into quantities scaling slippage variables natively."""
        physical_qty_to_execute = target_allocation_fraction * current_physical_quantity
        side = "sell" if physical_qty_to_execute > 0 else "hold"
        
        return {
            "side": side,
            "quantity": physical_qty_to_execute,
            "price": current_price
        }

    async def route_order(self, side: str, quantity: float, symbol: str, current_price: float, slippage_bps: float = 2.0):
        if side == "hold" or quantity <= 0.0001:
            return
            
        estimated_cost = quantity * current_price * (slippage_bps / 10000.0)
        ts = datetime.now().isoformat()
        
        # Format target symbol explicitly for Rest IO
        alpaca_symbol = symbol.replace("/", "")
        
        if self.shadow_mode:
            with open(self.log_path, "a") as f:
                f.write(f"{ts},{side},{alpaca_symbol},{quantity:.8f},{current_price:.4f},{estimated_cost:.4f},SHADOW_MODE\n")
            oems_logger.info(f"OEMS SHADOW ROUTED -> {side.upper()} {quantity:.8f} units {alpaca_symbol} @ ${current_price:.2f}")
        else:
            headers = {
                "APCA-API-KEY-ID": self.api_key,
                "APCA-API-SECRET-KEY": self.api_secret,
                "Content-Type": "application/json"
            }
            
            payload = {
                "symbol": alpaca_symbol,
                "qty": str(round(quantity, 8)),
                "side": side.lower(),
                "type": "market",
                "time_in_force": "ioc" # Immediate or Cancel ensuring clean slippage bounding
            }
            
            try:
                async with aiohttp.ClientSession() as session:
                    async with session.post(f"{self.base_url}/orders", json=payload, headers=headers) as resp:
                        if resp.status in [200, 201]:
                            res = await resp.json()
                            order_id = res.get("id", "ERROR_ID")
                            
                            with open(self.log_path, "a") as f:
                                f.write(f"{ts},{side},{alpaca_symbol},{quantity:.8f},{current_price:.4f},{estimated_cost:.4f},{order_id}\n")
                            oems_logger.info(f"PHYSICAL MARKET ROUTED -> {side.upper()} {quantity:.8f} units {alpaca_symbol} [Order ID: {order_id}]")
                        else:
                            txt = await resp.text()
                            oems_logger.error(f"Alpaca Rejection -> Code {resp.status} : {txt}")
                            
            except Exception as e:
                oems_logger.error(f"FATAL AIOHTTP Fault tracking API Order: {e}")
