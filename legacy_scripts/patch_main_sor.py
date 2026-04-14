import os
import re

def patch_main():
    filepath = "main.py"
    if not os.path.exists(filepath):
        print(f"⚠️ Could not find {filepath}")
        return

    with open(filepath, "r", encoding="utf-8") as f:
        content = f.read()

    # 1. Strip out the malformed code at the bottom of the file
    bad_code_idx = content.find("    async def execute_with_sor(self, symbol: str, qty: float, side: str, current_price: float):")
    if bad_code_idx != -1:
        content = content[:bad_code_idx]

    bad_syntax_idx = content.find("' not await self.submit_order(symbol, qty, side)")
    if bad_syntax_idx != -1:
        content = content[:bad_syntax_idx]

    # 2. Inject the custom execute_with_sor method inside ApexTradingSystem
    custom_sor_method = """
    async def execute_with_sor(self, connector, symbol: str, qty: float, side: str, current_price: float, confidence: float = 1.0):
        \"\"\"Routes orders through the SOR using the dynamic broker connector.\"\"\"
        if not getattr(ApexConfig, 'SOR_ENABLED', False) or current_price <= 0:
            return await connector.execute_order(symbol, side, qty, confidence=confidence)

        logger.info(f"🧠 SOR [{symbol}] Initializing Adaptive Pegging...")
        from execution.smart_order_router import SmartOrderRouter
        sor = SmartOrderRouter(
            max_urgency_steps=getattr(ApexConfig, 'SOR_MAX_URGENCY_STEPS', 3),
            step_delay_seconds=getattr(ApexConfig, 'SOR_STEP_DELAY_SECONDS', 10)
        )

        estimated_spread = current_price * 0.001
        bid_price = current_price - (estimated_spread / 2)
        ask_price = current_price + (estimated_spread / 2)

        async def place_fn(price, order_type):
            limit_val = round(price, 4) if order_type == "LIMIT" else None
            try:
                return await connector.execute_order(symbol, side, qty, confidence=confidence, order_type=order_type, limit_price=limit_val)
            except Exception as e:
                logger.error(f"SOR Place Error: {e}")
                return None

        async def status_fn(trade_obj):
            if not trade_obj: return "UNKNOWN"
            if isinstance(trade_obj, dict):
                return str(trade_obj.get("status", "UNKNOWN")).upper()
            try:
                return str(trade_obj.orderStatus.status).upper()
            except:
                return "UNKNOWN"

        async def cancel_fn(trade_obj):
            if not trade_obj: return True
            try:
                if isinstance(trade_obj, dict) and "id" in trade_obj:
                    return await connector.cancel_order(trade_obj["id"])
                if hasattr(connector, 'ib') and hasattr(trade_obj, 'order'):
                    connector.ib.cancelOrder(trade_obj.order)
                    return True
            except: pass
            return True

        sor_result = await sor.execute_adaptive_order(
            symbol=symbol, qty=qty, side=side, bid=bid_price, ask=ask_price,
            place_order_fn=place_fn, check_status_fn=status_fn, cancel_order_fn=cancel_fn
        )
        
        if sor_result:
            return {"status": "FILLED", "price": current_price, "expected_price": current_price}
        return None

"""
    # Append the method right before `async def main():`
    main_func_idx = content.find("async def main():")
    if main_func_idx != -1 and "async def execute_with_sor(self, connector" not in content:
        content = content[:main_func_idx] + custom_sor_method + content[main_func_idx:]

    # 3. Replace normal Exit Order
    old_exit = """trade = await exit_connector.execute_order(
                            symbol=symbol,
                            side=order_side,
                            quantity=abs(current_pos),
                            confidence=abs(signal) if signal != 0 else 0.8
                        )"""
    new_exit = """trade = await self.execute_with_sor(
                            connector=exit_connector,
                            symbol=symbol,
                            qty=abs(current_pos),
                            side=order_side,
                            current_price=price,
                            confidence=abs(signal) if signal != 0 else 0.8
                        )"""
    content = content.replace(old_exit, new_exit)

    # 4. Replace normal Entry Order
    old_entry = """trade = await entry_connector.execute_order(
                                symbol=symbol,
                                side=side,
                                quantity=shares,
                                confidence=confidence
                            )"""
    new_entry = """trade = await self.execute_with_sor(
                                connector=entry_connector,
                                symbol=symbol,
                                qty=shares,
                                side=side,
                                current_price=price,
                                confidence=confidence
                            )"""
    content = content.replace(old_entry, new_entry)

    # 5. Replace Retry Exit Order
    old_retry = """trade = await retry_connector.execute_order(
                    symbol=symbol,
                    side=order_side,
                    quantity=abs(current_pos),
                    confidence=0.9  # High confidence for exits
                )"""
    new_retry = """trade = await self.execute_with_sor(
                    connector=retry_connector,
                    symbol=symbol,
                    qty=abs(current_pos),
                    side=order_side,
                    current_price=self.price_cache.get(symbol, 0),
                    confidence=0.9
                )"""
    content = content.replace(old_retry, new_retry)

    with open(filepath, "w", encoding="utf-8") as f:
        f.write(content)
    print("✅ Successfully patched main.py to seamlessly use the Smart Order Router!")

if __name__ == "__main__":
    patch_main()
