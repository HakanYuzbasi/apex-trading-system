from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from datetime import datetime, timezone
from quant_system.events.signal import SignalEvent

def test_signal():
    now = datetime.now(timezone.utc)
    try:
        sig = SignalEvent(
            instrument_id="AAPL",
            exchange_ts=now,
            received_ts=now,
            processed_ts=now,
            sequence_id=0,
            source="test",
            strategy_id="test_strat",
            side="buy",
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
    test_signal()
