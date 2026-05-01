from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from quant_system.core.bus import InMemoryEventBus
from quant_system.strategies.base import BaseStrategy

class MockStrategy(BaseStrategy):
    def on_bar(self, event): pass
    def on_tick(self, event): pass

def test_emit():
    bus = InMemoryEventBus()
    strat = MockStrategy(bus)
    try:
        strat.emit_signal(
            instrument_id="AAPL",
            target_type="notional",
            target_value=1000.0,
            confidence=0.9,
            stop_model="fixed"
        )
        print("Success!")
    except TypeError as e:
        print(f"Caught TypeError: {e}")
    except Exception as e:
        print(f"Caught Exception: {type(e).__name__}: {e}")

if __name__ == "__main__":
    test_emit()
