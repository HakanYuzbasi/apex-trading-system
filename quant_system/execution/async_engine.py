"""
quant_system/execution/async_engine.py
================================================================================
REAL-TIME ASYNC MASTER EXECUTION ENGINE (ALPACA LIVE HOOKS)
================================================================================
"""

import asyncio
import numpy as np
import logging
import os

from quant_system.data.streaming_buffers import TickBuffer
from quant_system.data.live_websocket import AlpacaDataStream
from quant_system.governance.risk_manager import SystemHealthMonitor
from quant_system.infrastructure.persistence import StateManager
from quant_system.execution.live_oems import AlpacaOEMS
from quant_system.features.state_scaler import OnlineStateScaler
from quant_system.models.dqn_agent import ContinuousExecutionAgent
from quant_system.governance.reconciliation import PositionReconciler

os.makedirs('quant_system/logs', exist_ok=True)
logger = logging.getLogger("AsyncEngine")
logger.setLevel(logging.INFO)
if not logger.handlers:
    ch = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - ASYNC_ENGINE - %(message)s')
    ch.setFormatter(formatter)
    logger.addHandler(ch)

class AsyncExecutionEngine:
    def __init__(self):
        # 1. Pipeline Objects
        self.tick_buffer = TickBuffer(maxlen=200)
        self.health_monitor = SystemHealthMonitor(drawdown_limit=-0.05)
        # UPGRADE 2: Configured AlpacaOEMS replacing mock generator. Shadow currently locked explicitly for verification sequences mapping to real WebSockets
        self.oems = AlpacaOEMS(shadow_mode=True) 
        self.scaler = OnlineStateScaler(15, exclude_dims=[13])
        self.agent = ContinuousExecutionAgent(state_dim=15)
        
        self.state_mgr = StateManager(scaler=self.scaler, risk_manager=self.health_monitor)
        
        # 2. Portfolio PnL States
        self.entry_price = 0.0
        self.peak_pnl = 0.0
        self.position_sz = 1.0 
        self.total_portfolio_usd = 1000000.0
        self.asset_quantity = 0.0
        
        # UPGRADE 3: Reconciliation Engine Hooking Reference Memory Array
        self.reconciler = PositionReconciler(self)

    async def run(self):
        logger.info("Initializing Live Async Execution Engine...")
        self.state_mgr.load_checkpoint() 
        
        # UPGRADE 1 & 4: Live Data Socket Consumer Drop
        target_asset = "BTC/USD"
        stream = AlpacaDataStream(symbol=target_asset)
        
        ticks_processed = 0
        logger.info("Binding Core Daemon Task Handlers...")
        
        # Spin 60s Ledger Reconciliation Target
        asyncio.create_task(self.reconciler.sync_ledger(target_asset.replace("/", "")))
        
        # Continuous Async Live Event Loop
        async for tick in stream.stream_ticks():
            # Step 1: Update O(1) buffer instantly maintaining pure high-frequency execution sequences
            self.tick_buffer.update(tick)
            ticks_processed += 1
            current_price = tick['price']
            
            if ticks_processed == 1:
                self.entry_price = current_price
                self.asset_quantity = (self.position_sz * self.total_portfolio_usd) / current_price
                logger.info(f"Origin Baseline Extracted @ ${current_price:.2f}")
                
            if ticks_processed < 50:
                continue # Warmup buffer mapping true constraints properly
                
            # Step 2: Extract real-time streaming constraints
            vol = self.tick_buffer.current_volatility
            ma = self.tick_buffer.moving_average
            mom = self.tick_buffer.momentum
            
            # Reconstruct PnL internally ensuring logic traces live limits directly decoupled from broker
            open_pnl = (current_price / self.entry_price) - 1.0 if self.entry_price > 0 else 0.0
            if open_pnl > self.peak_pnl: self.peak_pnl = open_pnl
            drawdown = min(0.0, open_pnl - self.peak_pnl)
            
            # Simulated Matrix Outputs mirroring the Neural Architecture for Event loops
            meta_prob = np.random.uniform(0.4, 0.6)
            confidence = np.random.uniform(0.7, 0.95)
            kl_div = np.random.uniform(0.01, 0.05)
            
            condition = self.health_monitor.evaluate_health(vol, kl_div, drawdown)
            
            raw_state = np.zeros(15)
            raw_state[0] = ma / current_price 
            raw_state[1] = mom               
            raw_state[4] = meta_prob
            raw_state[5] = confidence
            raw_state[6] = kl_div
            raw_state[7] = vol 
            raw_state[10] = self.position_sz
            raw_state[11] = open_pnl
            raw_state[12] = drawdown
            raw_state[13] = 1 # Generic baseline mapping
            raw_state[14] = self.tick_buffer.returns[-1] if self.tick_buffer.returns else 0.0
            
            s_scaled = self.scaler.scale(raw_state, update=True)
            action_dict = self.agent.select_action(s_scaled, raw_state, evaluate=True)
            
            # Graceful Degradation Filters Overriding Deep Matrices strictly on physical outputs natively
            action_dict['action'] = self.health_monitor.apply_graceful_degradation(action_dict['action'], condition)
            action_fraction = action_dict['action']
            
            # Step 5: OEMS Async Routing (Awaits physical POST network IO cleanly avoiding loop stalls)
            order_intent = self.oems.resolve_action(action_fraction, self.asset_quantity, current_price, self.total_portfolio_usd)
            
            if order_intent['quantity'] > 0:
                await self.oems.route_order(order_intent['side'], order_intent['quantity'], target_asset, current_price)
                self.asset_quantity -= order_intent['quantity']
                
            # Step 6: Non-Blocking State Persistence Checkpoint Serialization Trap
            if ticks_processed % 1000 == 0:
                self.state_mgr.save_checkpoint()
                logger.info(f"Asynchronous Live State Checkpoint Dumped -> Tick {ticks_processed}")
                
if __name__ == "__main__":
    engine = AsyncExecutionEngine()
    
    # Optional graceful fault mappings
    try:
        asyncio.run(engine.run())
    except KeyboardInterrupt:
        logger.warning("Operator Override Caught. Engine Down.")
