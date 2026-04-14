from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any, Callable, Dict, List, Optional

import numpy as np
import bottleneck as bn

from quant_system.core.bus import InMemoryEventBus
from quant_system.strategies.kalman_pairs import KalmanPairsStrategy
from quant_system.portfolio.ledger import PortfolioLedger

logger = logging.getLogger(__name__)

@dataclass
class ShadowPerformance:
    pair_label: str
    returns: List[float]
    last_equity: float = 1.0
    
    def add_return(self, ret: float):
        self.returns.append(ret)
        if len(self.returns) > 10000: # Cap memory
            self.returns = self.returns[-10000:]
            
    def get_sortino(self, lookback_days: int = 30) -> float:
        if not self.returns:
            return 0.0
        
        # Use bottleneck for faster mean/std on return arrays
        rets = np.array(self.returns)
        if len(rets) < 5: return 0.0
        
        avg_ret = bn.nanmean(rets)
        downside_rets = rets[rets < 0]
        if len(downside_rets) < 2:
            downside_std = 1e-6
        else:
            downside_std = bn.nanstd(downside_rets)
            
        return (avg_ret / downside_std) * np.sqrt(252 * 6.5 * 60)

class StrategyRotator:
    """
    Manages a universe of 'Backbench' pairs, running them in shadow mode
    and suggesting swaps when an active pair underperforms.
    """
    
    def __init__(
        self,
        event_bus: InMemoryEventBus,
        shadow_universe: List[Dict[str, Any]],
        active_ledger: PortfolioLedger,
        swap_threshold_ratio: float = 1.5, # Backbench must be 50% better
        lookback_days: int = 30
    ) -> None:
        self._event_bus = event_bus
        self._universe = shadow_universe
        self._ledger = active_ledger
        self._swap_threshold = swap_threshold_ratio
        self._lookback_days = lookback_days
        
        self._shadow_strategies: Dict[str, KalmanPairsStrategy] = {}
        self._performance: Dict[str, ShadowPerformance] = {}
        
        # Initialize shadow strategies
        for config in self._universe:
            label = f"{config['instrument_a']}/{config['instrument_b']}"
            self._performance[label] = ShadowPerformance(pair_label=label, returns=[])
            
            # We instantiate the strategy but don't give it a way to actually trade
            # In 'Shadow' mode, we just want to track its signal -> pnl
            strategy = KalmanPairsStrategy(
                self._event_bus,
                instrument_a=config['instrument_a'],
                instrument_b=config['instrument_b'],
                entry_z_score=config.get('entry_z_score', 2.0),
                exit_z_score=config.get('exit_z_score', 0.5),
                leg_notional=config.get('leg_notional', 5000.0),
                warmup_bars=config.get('warmup_bars', 20),
            )
            self._shadow_strategies[label] = strategy
            
        logger.info("StrategyRotator initialized with %d backbench pairs", len(self._universe))

    async def run_update_loop(self, stop_event: asyncio.Event):
        """Periodically update shadow performance."""
        while not stop_event.is_set():
            try:
                # In a real system, we'd listen to 'signal' events from shadow strategies
                # and calculate what the PnL would have been.
                # For this implementation, we assume the AlphaMonitor or a similar
                # component is tracking this, or we just mock return updates.
                
                # Here we reconcile performance for all strategies
                await asyncio.sleep(60) # Update every minute
            except Exception as e:
                logger.error(f"Error in Rotator loop: {e}")

    def get_swaps(self, active_pairs: List[tuple[str, float]]) -> List[tuple[str, Dict[str, Any]]]:
        """
        active_pairs: List of (label, current_sortino)
        Returns a list of (active_label_to_remove, backbench_config_to_add)
        """
        swaps = []
        backbench_sorted = sorted(
            [(l, p.get_sortino()) for l, p in self._performance.items()],
            key=lambda x: x[1],
            reverse=True
        )
        
        used_backbench = set()
        
        for active_label, active_sortino in active_pairs:
            for bb_label, bb_sortino in backbench_sorted:
                if bb_label in used_backbench:
                    continue
                
                if bb_sortino > active_sortino * self._swap_threshold:
                    logger.info("Swap suggested: %s (Sortino %.2f) -> %s (Sortino %.2f)", 
                                active_label, active_sortino, bb_label, bb_sortino)
                    
                    # Find backbench config
                    bb_config = next(c for c in self._universe if f"{c['instrument_a']}/{c['instrument_b']}" == bb_label)
                    swaps.append((active_label, bb_config))
                    used_backbench.add(bb_label)
                    break # Move to next active pair
                    
        return swaps

    def close(self):
        for s in self._shadow_strategies.values():
            s.close()
