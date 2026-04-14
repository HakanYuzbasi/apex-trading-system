import os
import re

def inject_sor_into_execution_loop():
    # Look for the execution loop file in standard directories
    paths = ["core/execution_loop.py", "execution/execution_loop.py", "execution_loop.py"]
    filepath = None
    for p in paths:
        if os.path.exists(p):
            filepath = p
            break
    
    if not filepath:
        print("⚠️ Could not find execution_loop.py. Please verify the folder structure.")
        return
        
    with open(filepath, "r", encoding="utf-8") as f:
        content = f.read()

    # 1. Inject the Required Imports safely
    if "SmartOrderRouter" not in content:
        import_statement = "\nfrom execution.smart_order_router import SmartOrderRouter\nfrom config import ApexConfig\n"
        
        # Find a safe place to put the imports (before the first class or function definition)
        insert_idx = content.find("class ")
        if insert_idx == -1: insert_idx = content.find("def ")
        
        if insert_idx != -1:
            content = content[:insert_idx] + import_statement + content[insert_idx:]
        else:
            content = import_statement + content

    # 2. Inject the SOR Wrapper Method
    if "async def execute_with_sor" not in content:
        sor_method = """
    async def execute_with_sor(self, symbol: str, qty: float, side: str, current_price: float):
        \"\"\"
        Intercepts standard execution and routes it through the Adaptive Pegging SOR.
        \"\"\"
        if not getattr(ApexConfig, 'SOR_ENABLED', False):
            self.logger.info(f"SOR disabled. Routing standard market order for {symbol}")
            return await self.broker.submit_order(symbol, qty, side, type="MARKET")

        self.logger.info(f"Initializing Smart Order Router for {symbol}")
        sor = SmartOrderRouter(
            max_urgency_steps=getattr(ApexConfig, 'SOR_MAX_URGENCY_STEPS', 3),
            step_delay_seconds=getattr(ApexConfig, 'SOR_STEP_DELAY_SECONDS', 10)
        )

        # Estimate bid/ask if top-of-book isn't passed directly (assumes a 0.05% spread)
        # For maximum alpha, replace this by fetching live L1 quotes: await self.broker.get_quote(symbol)
        estimated_spread = current_price * 0.0005
        bid_price = current_price - (estimated_spread / 2)
        ask_price = current_price + (estimated_spread / 2)

        # Wire up the SOR callbacks to your existing broker client
        async def place_fn(price, order_type):
            limit_val = round(price, 2) if order_type == "LIMIT" else None
            return await self.broker.submit_order(symbol, qty, side, type=order_type, limit_price=limit_val)
            
        async def status_fn(order_id):
            order = await self.broker.get_order(order_id)
            return order.status if order else "UNKNOWN"
            
        async def cancel_fn(order_id):
            return await self.broker.cancel_order(order_id)

        # Fire the Smart Order Router
        return await sor.execute_adaptive_order(
            symbol=symbol,
            qty=qty,
            side=side,
            bid=bid_price,
            ask=ask_price,
            place_order_fn=place_fn,
            check_status_fn=status_fn,
            cancel_order_fn=cancel_fn
        )
"""
        # Append to the end of the file (assuming it's part of the main execution class)
        content += sor_method
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(content)
        print(f"✅ Successfully injected `execute_with_sor()` into {filepath}")
    else:
        print(f"⚡ `execute_with_sor()` is already present in {filepath}")

if __name__ == "__main__":
    print("🔌 Wiring the Smart Order Router into the Execution Loop...")
    inject_sor_into_execution_loop()
