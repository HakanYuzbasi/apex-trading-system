import logging
import asyncio
from unittest.mock import MagicMock
from quant_system.execution.neural_sniper import NeuralSniper
from quant_system.events.order import OrderEvent

# Configure logging to see the shadow mode output
logging.basicConfig(level=logging.INFO)

async def test_shadow_mode():
    event_bus = MagicMock()
    # Mock PPO model
    mock_model = MagicMock()
    # Mock prediction: 0 (Wait)
    mock_model.predict.return_value = (0, None)
    
    sniper = NeuralSniper(event_bus, model_path="run_state/models/ppo_execution_v1.zip")
    sniper.model = mock_model
    sniper.model_loaded = True
    
    order = OrderEvent(
        order_id="test_1",
        instrument_id="BTC/USD",
        side="buy",
        quantity=1.0,
        order_type="limit"
    )
    
    print("\n--- Testing Shadow Mode Override ---")
    price = sniper._target_limit_price(order, bid=50000.0, ask=50100.0)
    
    print(f"PPO predicted Action 0 (Wait), but forced Action 3 (Market Sweep)")
    print(f"Resulting price for Buy: {price} (should be 50100.0)")
    
    if price == 50100.0:
        print("✅ SUCCESS: Shadow mode override confirmed.")
    else:
        print("❌ FAILURE: Shadow mode override failed.")

if __name__ == "__main__":
    asyncio.run(test_shadow_mode())
