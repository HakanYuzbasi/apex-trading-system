import sys
import os
import time

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from quant_system.core.bus import InMemoryEventBus
from quant_system.core.bus import InMemoryEventBus
from quant_system.events import SignalEvent
from quant_system.execution.fast_math import cython_kalman_update
from quant_system.strategies.breakout_pod import BreakoutPodStrategy
import numpy as np

class MockReconciler:
    def __init__(self, event_bus):
        self.tracked_signals = []
        event_bus.subscribe("signal", self.on_signal)

    def on_signal(self, event):
        self.tracked_signals.append(event)

def test():
    print("Testing Cython integration...")
    try:
        # Mock calculation
        start = time.perf_counter()
        theta = np.array([1.0, 0.0], dtype=np.float64)
        cov = np.eye(2, dtype=np.float64)
        cython_kalman_update(150.0, 140.0, theta, cov, 1e-4, 1e-3)
        end = time.perf_counter()
        elapsed = (end - start) * 1e6
        print(f"Cython calculation time: {elapsed:.2f} µs")
        assert elapsed < 50, "Latency threshold breached (>50 µs)!"
        print("Cython integration verified.\n")
    except Exception as e:
        print(f"Cython Integration failed: {e}")

    print("Testing Shadow Accounting / Event Bus...")
    bus = InMemoryEventBus()
    reconciler = MockReconciler(bus)

    # Breakout strategy emission
    import datetime
    now_ts = datetime.datetime.now(datetime.timezone.utc)
    
    breakout = BreakoutPodStrategy(bus, instrument_id="NVDA")
    print("Simulating breakout signal...")
    bus.publish(SignalEvent(
        instrument_id="NVDA", 
        side="buy", 
        target_value=1.0, 
        target_type="notional",
        strategy_id="breakout", 
        metadata={"reason": "high_vol"},
        confidence=0.8,
        stop_model="atr",
        exchange_ts=now_ts, received_ts=now_ts, processed_ts=now_ts, sequence_id=0, source="test"
    ))
    
    # Kalman strategy emission
    print("Simulating kalman signal...")
    bus.publish(SignalEvent(
        instrument_id="MSFT-AAPL", 
        side="short", 
        target_value=0.5, 
        target_type="notional",
        strategy_id="mean_reversion", 
        metadata={"reason": "z_cross"},
        confidence=0.8,
        stop_model="fixed",
        exchange_ts=now_ts, received_ts=now_ts, processed_ts=now_ts, sequence_id=1, source="test"
    ))

    time.sleep(0.1)  # allow handlers to process
    
    print(f"Shadow Reconciler tracked {len(reconciler.tracked_signals)} signals.")
    for sig in reconciler.tracked_signals:
        print(f"  -> {sig.instrument_id} | {sig.side} | strat: {sig.strategy_id}")

    assert len(reconciler.tracked_signals) == 2, "Shadow Reconciler failed to track both strategies!"
    print("Shadow Accounting tracking verified.")

if __name__ == "__main__":
    test()
