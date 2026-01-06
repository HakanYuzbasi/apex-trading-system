"""
execution/ibkr_connector.py - Interactive Brokers Connector
Handles live/paper trading with IBKR TWS/Gateway
FIXED VERSION: Delayed market data + Position sync + LONG/SHORT enabled
"""

import asyncio
import logging
from typing import Dict, Optional
from ib_insync import IB, Stock, MarketOrder, LimitOrder, util
import pandas as pd


logger = logging.getLogger(__name__)


class IBKRConnector:
    """Interactive Brokers connector with delayed market data support."""
    
    def __init__(self, host: str = '127.0.0.1', port: int = 7497, client_id: int = 1):
        """
        Initialize IBKR connector.
        
        Args:
            host: TWS/Gateway host (default: localhost)
            port: 7497 for paper, 7496 for live
            client_id: Unique client ID
        """
        self.host = host
        self.port = port
        self.client_id = client_id
        self.ib = IB()
        self.account = None
        
        # Cache for contracts
        self.contracts = {}
        
        # Market data request IDs
        self.req_id_counter = 0
    
    async def connect(self):
        """Connect to Interactive Brokers with delayed market data."""
        logger.info(f"🔌 Connecting to IBKR at {self.host}:{self.port}...")
        
        try:
            await self.ib.connectAsync(self.host, self.port, clientId=self.client_id)
            
            # ✅ FIX: Enable delayed/frozen market data (FREE for paper trading)
            # Type 1 = Live (requires paid subscription)
            # Type 2 = Frozen (last available)
            # Type 3 = Delayed (15-20 min delayed, FREE)
            # Type 4 = Delayed-Frozen
            self.ib.reqMarketDataType(3)
            logger.info("📊 Delayed market data enabled (free)")
            
            # Get account info
            await asyncio.sleep(1)  # Wait for account sync
            accounts = self.ib.wrapper.accounts
            if accounts:
                self.account = accounts[0]
            
            logger.info("✅ Connected to Interactive Brokers!")
            logger.info(f"📋 Account: {self.account}")
            
        except Exception as e:
            logger.error(f"❌ Failed to connect to IBKR: {e}")
            raise
    
    def disconnect(self):
        """Disconnect from Interactive Brokers."""
        if self.ib.isConnected():
            self.ib.disconnect()
            logger.info("🔌 Disconnected from IBKR")
    
    async def get_contract(self, symbol: str) -> Optional[Stock]:
        """
        Get or create stock contract.
        
        Args:
            symbol: Stock ticker (e.g., 'AAPL')
            
        Returns:
            Stock contract or None if not found
        """
        if symbol in self.contracts:
            return self.contracts[symbol]
        
        try:
            # Create stock contract
            contract = Stock(symbol, 'SMART', 'USD')
            
            # Qualify contract (get full details from IBKR)
            qualified = await self.ib.qualifyContractsAsync(contract)
            
            if qualified:
                self.contracts[symbol] = qualified[0]
                return qualified[0]
            else:
                logger.warning(f"⚠️  Could not qualify contract for {symbol}")
                return None
                
        except Exception as e:
            logger.error(f"❌ Error getting contract for {symbol}: {e}")
            return None
    
    async def get_market_price(self, symbol: str) -> float:
        """
        Get current market price for a symbol.
        
        Args:
            symbol: Stock ticker
            
        Returns:
            Current price or 0 if unavailable
        """
        try:
            contract = await self.get_contract(symbol)
            if not contract:
                return 0.0
            
            # Request market data snapshot
            ticker = self.ib.reqMktData(contract, '', False, False)
            
            # Wait for price update (max 2 seconds)
            for _ in range(20):
                await asyncio.sleep(0.1)
                
                # Try different price fields
                if ticker.last and ticker.last > 0:
                    price = ticker.last
                    self.ib.cancelMktData(contract)
                    return float(price)
                elif ticker.close and ticker.close > 0:
                    price = ticker.close
                    self.ib.cancelMktData(contract)
                    return float(price)
                elif ticker.bid and ticker.ask and ticker.bid > 0 and ticker.ask > 0:
                    price = (ticker.bid + ticker.ask) / 2
                    self.ib.cancelMktData(contract)
                    return float(price)
            
            # Fallback: try to get last close from historical data
            self.ib.cancelMktData(contract)
            bars = await self.ib.reqHistoricalDataAsync(
                contract,
                endDateTime='',
                durationStr='1 D',
                barSizeSetting='1 day',
                whatToShow='TRADES',
                useRTH=True
            )
            
            if bars:
                return float(bars[-1].close)
            
            logger.warning(f"⚠️  No price available for {symbol}")
            return 0.0
            
        except Exception as e:
            logger.debug(f"Error getting price for {symbol}: {e}")
            return 0.0
    
    async def get_portfolio_value(self) -> float:
        """
        Get total portfolio value.
        
        Returns:
            Total account value in USD
        """
        try:
            # Request account summary
            account_values = self.ib.accountValues()
            
            # Find NetLiquidation (total portfolio value)
            for av in account_values:
                if av.tag == 'NetLiquidation' and av.currency == 'USD':
                    logger.debug(f"💼 Portfolio Value: ${float(av.value):,.2f}")
                    return float(av.value)
            
            # Fallback: calculate from positions
            positions = await self.get_all_positions()
            total = 0.0
            for symbol, qty in positions.items():
                price = await self.get_market_price(symbol)
                total += qty * price
            
            return total
            
        except Exception as e:
            logger.error(f"❌ Error getting portfolio value: {e}")
            return 0.0
    
    async def get_position(self, symbol: str) -> int:
        """
        Get current position quantity for a symbol.
        
        Args:
            symbol: Stock ticker
            
        Returns:
            Position quantity (positive=long, negative=short, 0=no position)
        """
        try:
            positions = self.ib.positions()
            
            for pos in positions:
                if pos.contract.symbol == symbol:
                    return int(pos.position)
            
            return 0
            
        except Exception as e:
            logger.error(f"❌ Error getting position for {symbol}: {e}")
            return 0
    
    async def get_all_positions(self) -> Dict[str, int]:
        """
        ✅ FIXED: Get all current positions (long AND short).
        
        Returns:
            Dictionary of {symbol: quantity} where qty can be positive (long) or negative (short)
        """
        try:
            positions = self.ib.positions()
            
            result = {}
            for pos in positions:
                symbol = pos.contract.symbol
                qty = int(pos.position)
                
                if qty != 0:
                    result[symbol] = qty
                    position_type = "LONG" if qty > 0 else "SHORT"
                    logger.debug(f"   {symbol}: {abs(qty)} shares ({position_type})")
            
            if result:
                logger.info(f"📊 Loaded {len(result)} existing positions")
            
            return result
            
        except Exception as e:
            logger.error(f"❌ Error getting positions: {e}")
            return {}
    
    async def ensure_delayed_data_mode(self):
        """Ensure delayed data mode is active."""
        try:
            if self.ib.isConnected():
                self.ib.reqMarketDataType(3)  # 3 = Delayed (15-20 min)
                logger.debug("✅ Delayed data mode re-confirmed")
        except Exception as e:
            logger.debug(f"Error setting delayed data mode: {e}")

    async def execute_order(self, symbol: str, side: str, quantity: int, 
                        confidence: float = 0.5, force_market: bool = False) -> Optional[dict]:
        """
        ✅ FIXED: Execute order - LONG AND SHORT enabled with safety checks.
        
        Args:
            symbol: Stock ticker
            side: 'BUY' or 'SELL'
            quantity: Number of shares (always positive)
            confidence: Signal confidence (0-1) - affects limit buffer
            force_market: If True, always use market order
            
        Returns:
            Trade details or None if failed
        """
        from datetime import datetime
        
        try:
            # ✅ Safety check: Validate side
            if side not in ['BUY', 'SELL']:
                logger.error(f"❌ Invalid order side '{side}'. Only BUY/SELL allowed")
                return None
            
            # ✅ Safety check: Validate quantity
            if quantity <= 0:
                logger.error(f"❌ Invalid quantity {quantity}. Must be positive")
                return None
            
            # ✅ Safety check: Get current position
            current_pos = await self.get_position(symbol)
            
            # ✅ Warn if creating/increasing short position
            if side == 'SELL':
                if current_pos == 0:
                    logger.warning(f"⚠️  {symbol}: Opening SHORT position ({quantity} shares)")
                elif current_pos > 0:
                    if quantity > current_pos:
                        short_qty = quantity - current_pos
                        logger.warning(f"⚠️  {symbol}: Flipping to SHORT by {short_qty} shares")
                    else:
                        logger.info(f"✅ {symbol}: Reducing LONG position")
                elif current_pos < 0:
                    logger.warning(f"⚠️  {symbol}: Increasing SHORT position (now {abs(current_pos + quantity)} shares short)")
            
            # Get current price
            price = await self.get_market_price(symbol)
            if price == 0:
                logger.error(f"❌ Cannot execute: no price for {symbol}")
                return None
            
            # Determine if market is open
            now = datetime.now()
            
            # Market hours: 15:30-22:00 CET (9:30 AM - 4:00 PM EST)
            market_open_hour = 15
            market_open_minute = 30
            market_close_hour = 22
            
            market_open_time = now.replace(hour=market_open_hour, minute=market_open_minute, second=0)
            market_close_time = now.replace(hour=market_close_hour, minute=0, second=0)
            
            is_market_hours = market_open_time <= now < market_close_time
            
            # Decision logic
            if force_market or is_market_hours:
                # Use market order during trading hours
                logger.info(f"   📈 Market order (hours={is_market_hours})")
                return await self._execute_market_order(symbol, side, quantity)
            else:
                # Use limit order pre-market with confidence-based buffer
                return await self._execute_limit_order(symbol, side, quantity, price, confidence)
                
        except Exception as e:
            logger.error(f"❌ Error executing order {side} {quantity} {symbol}: {e}")
            return None

    async def _execute_market_order(self, symbol: str, side: str, quantity: int) -> Optional[dict]:
        """Execute market order."""
        try:
            contract = await self.get_contract(symbol)
            if not contract:
                logger.error(f"❌ Invalid contract for {symbol}")
                return None
            
            # Create market order
            action = 'BUY' if side.upper() == 'BUY' else 'SELL'
            order = MarketOrder(action, quantity)
            
            # Place order
            trade = self.ib.placeOrder(contract, order)
            
            # Wait for order to fill (max 10 seconds)
            for _ in range(100):
                await asyncio.sleep(0.1)
                
                if trade.orderStatus.status == 'Filled':
                    fill_price = trade.orderStatus.avgFillPrice
                    logger.info(f"✅ {action} {quantity} {symbol} @ ${fill_price:.2f}")
                    
                    return {
                        'symbol': symbol,
                        'side': side,
                        'quantity': quantity,
                        'price': fill_price,
                        'status': 'FILLED'
                    }
                
                elif trade.orderStatus.status in ['Cancelled', 'ApiCancelled', 'Inactive']:
                    logger.error(f"❌ Order {action} {quantity} {symbol} cancelled")
                    return None
            
            # Timeout
            if trade.orderStatus.status != 'Filled':
                logger.warning(f"⚠️  Order timeout for {symbol}")
                self.ib.cancelOrder(order)
                return None
            
            return {
                'symbol': symbol,
                'side': side,
                'quantity': quantity,
                'price': trade.orderStatus.avgFillPrice,
                'status': 'FILLED'
            }
            
        except Exception as e:
            logger.error(f"❌ Market order error: {e}")
            return None

    async def _execute_limit_order(self, symbol: str, side: str, quantity: int, 
                                current_price: float, confidence: float) -> Optional[dict]:
        """
        Execute limit order with confidence-based buffer.
        
        Args:
            symbol: Stock ticker
            side: 'BUY' or 'SELL'
            quantity: Number of shares
            current_price: Current market price
            confidence: Signal confidence (0-1)
        """
        try:
            contract = await self.get_contract(symbol)
            if not contract:
                logger.error(f"❌ Invalid contract for {symbol}")
                return None
            
            # Calculate limit price based on confidence
            if confidence > 0.70:
                buffer = 0.03  # 3% - aggressive (high conviction)
            elif confidence > 0.50:
                buffer = 0.02  # 2% - moderate
            else:
                buffer = 0.01  # 1% - conservative (low conviction)
            
            action = 'BUY' if side.upper() == 'BUY' else 'SELL'
            
            if action == 'BUY':
                limit_price = current_price * (1 + buffer)  # Buy up to X% above
            else:
                limit_price = current_price * (1 - buffer)  # Sell X% below
            
            # Round to 2 decimals
            limit_price = round(limit_price, 2)
            
            logger.info(f"   💰 Limit @ ${limit_price:.2f} (buffer: {buffer*100:.1f}%, conf: {confidence:.2f})")
            
            # Create limit order
            order = LimitOrder(action, quantity, limit_price)
            
            # Place order
            trade = self.ib.placeOrder(contract, order)
            
            # For pre-market orders, they'll be queued until market opens
            # Return immediately with PreSubmitted status
            await asyncio.sleep(0.5)  # Brief wait for order acknowledgment
            
            status = trade.orderStatus.status
            
            if status in ['PreSubmitted', 'Submitted', 'PendingSubmit']:
                logger.info(f"✅ {action} {quantity} {symbol} limit order placed (status: {status})")
                
                return {
                    'symbol': symbol,
                    'side': side,
                    'quantity': quantity,
                    'price': limit_price,  # Expected price
                    'status': 'PENDING',
                    'order_id': trade.order.orderId
                }
            
            elif status == 'Filled':
                fill_price = trade.orderStatus.avgFillPrice
                logger.info(f"✅ {action} {quantity} {symbol} @ ${fill_price:.2f} (limit filled)")
                
                return {
                    'symbol': symbol,
                    'side': side,
                    'quantity': quantity,
                    'price': fill_price,
                    'status': 'FILLED'
                }
            
            else:
                logger.error(f"❌ Limit order rejected for {symbol}, status: {status}")
                return None
            
        except Exception as e:
            logger.error(f"❌ Limit order error: {e}")
            return None
    
    async def get_account_cash(self) -> float:
        """
        Get available cash in account.
        
        Returns:
            Available cash in USD
        """
        try:
            account_values = self.ib.accountValues()
            
            for av in account_values:
                if av.tag == 'AvailableFunds' and av.currency == 'USD':
                    return float(av.value)
            
            return 0.0
            
        except Exception as e:
            logger.error(f"❌ Error getting account cash: {e}")
            return 0.0
    
    async def get_buying_power(self) -> float:
        """
        Get buying power (includes margin).
        
        Returns:
            Buying power in USD
        """
        try:
            account_values = self.ib.accountValues()
            
            for av in account_values:
                if av.tag == 'BuyingPower' and av.currency == 'USD':
                    return float(av.value)
            
            return 0.0
            
        except Exception as e:
            logger.error(f"❌ Error getting buying power: {e}")
            return 0.0
    
    async def get_historical_bars(self, symbol: str, days: int = 100) -> pd.DataFrame:
        """
        Get historical price data.
        
        Args:
            symbol: Stock ticker
            days: Number of days of history
            
        Returns:
            DataFrame with OHLCV data
        """
        try:
            contract = await self.get_contract(symbol)
            if not contract:
                return pd.DataFrame()
            
            # Request historical data
            bars = await self.ib.reqHistoricalDataAsync(
                contract,
                endDateTime='',
                durationStr=f'{days} D',
                barSizeSetting='1 day',
                whatToShow='TRADES',
                useRTH=True,
                formatDate=1
            )
            
            if not bars:
                return pd.DataFrame()
            
            # Convert to DataFrame
            df = util.df(bars)
            df.set_index('date', inplace=True)
            
            return df[['open', 'high', 'low', 'close', 'volume']]
            
        except Exception as e:
            logger.error(f"❌ Error getting historical data for {symbol}: {e}")
            return pd.DataFrame()
    
    def get_open_orders(self) -> list:
        """
        Get all open orders.
        
        Returns:
            List of open orders
        """
        try:
            return self.ib.openOrders()
        except Exception as e:
            logger.error(f"❌ Error getting open orders: {e}")
            return []
    
    async def cancel_all_orders(self):
        """Cancel all open orders."""
        try:
            open_orders = self.get_open_orders()
            for order in open_orders:
                self.ib.cancelOrder(order)
            logger.info(f"🚫 Cancelled {len(open_orders)} open orders")
        except Exception as e:
            logger.error(f"❌ Error cancelling orders: {e}")
    
    def has_pending_order(self, symbol: str) -> bool:
        """Check if symbol has pending or active orders."""
        try:
            # Get all open orders
            open_orders = self.ib.openOrders()
            
            # Check if any order exists for this symbol
            for trade in open_orders:
                if trade.contract.symbol == symbol:
                    status = trade.orderStatus.status
                    # Consider PreSubmitted, Submitted, Filled statuses
                    if status in ['PreSubmitted', 'Submitted', 'PendingSubmit', 'ApiPending']:
                        logger.debug(f"⏳ Skipping {symbol}: pending order exists (status={status})")
                        return True
            
            return False
        except Exception as e:
            logger.error(f"Error checking pending orders for {symbol}: {e}")
            return False
    
    def __del__(self):
        """Cleanup on deletion."""
        self.disconnect()
