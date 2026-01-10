"""
execution/ibkr_connector.py - Interactive Brokers Connector
Handles live/paper trading with IBKR TWS/Gateway

Features:
- Delayed market data support (free for paper trading)
- Position sync with IBKR
- Long/Short position support
- Exponential backoff retry logic for network resilience
- Slippage tracking for performance analysis
"""

import asyncio
import logging
import random
from datetime import datetime
from typing import Dict, Optional, Callable, TypeVar, Any
from functools import wraps
from ib_insync import IB, Stock, MarketOrder, LimitOrder, util
import pandas as pd

from config import ApexConfig

logger = logging.getLogger(__name__)

T = TypeVar('T')


def with_retry(
    max_retries: int = None,
    base_delay: float = None,
    max_delay: float = None,
    retryable_exceptions: tuple = (Exception,)
):
    """
    Decorator for exponential backoff retry logic.

    Args:
        max_retries: Maximum number of retry attempts
        base_delay: Initial delay between retries (seconds)
        max_delay: Maximum delay between retries (seconds)
        retryable_exceptions: Tuple of exception types to retry on
    """
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @wraps(func)
        async def async_wrapper(*args, **kwargs) -> T:
            retries = max_retries or ApexConfig.IBKR_MAX_RETRIES
            delay = base_delay or ApexConfig.IBKR_RETRY_BASE_DELAY
            max_d = max_delay or ApexConfig.IBKR_RETRY_MAX_DELAY

            last_exception = None

            for attempt in range(retries + 1):
                try:
                    return await func(*args, **kwargs)
                except retryable_exceptions as e:
                    last_exception = e
                    if attempt < retries:
                        # Calculate delay with exponential backoff and jitter
                        current_delay = min(delay * (2 ** attempt), max_d)
                        jitter = random.uniform(0, current_delay * 0.1)
                        sleep_time = current_delay + jitter

                        logger.warning(
                            f"⚠️  {func.__name__} failed (attempt {attempt + 1}/{retries + 1}): {e}"
                        )
                        logger.info(f"   Retrying in {sleep_time:.1f}s...")
                        await asyncio.sleep(sleep_time)
                    else:
                        logger.error(
                            f"❌ {func.__name__} failed after {retries + 1} attempts: {e}"
                        )

            raise last_exception

        @wraps(func)
        def sync_wrapper(*args, **kwargs) -> T:
            retries = max_retries or ApexConfig.IBKR_MAX_RETRIES
            delay = base_delay or ApexConfig.IBKR_RETRY_BASE_DELAY
            max_d = max_delay or ApexConfig.IBKR_RETRY_MAX_DELAY

            last_exception = None

            for attempt in range(retries + 1):
                try:
                    return func(*args, **kwargs)
                except retryable_exceptions as e:
                    last_exception = e
                    if attempt < retries:
                        import time
                        current_delay = min(delay * (2 ** attempt), max_d)
                        jitter = random.uniform(0, current_delay * 0.1)
                        sleep_time = current_delay + jitter
                        logger.warning(f"⚠️  {func.__name__} retry {attempt + 1}/{retries}")
                        time.sleep(sleep_time)

            raise last_exception

        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        return sync_wrapper

    return decorator


class IBKRConnector:
    """
    Interactive Brokers connector with enhanced features.

    Features:
    - Delayed market data support (free for paper trading)
    - Exponential backoff retry logic
    - Slippage and commission tracking
    - Long/Short position support
    """

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

        # Execution metrics tracking
        self.execution_metrics = {
            'total_trades': 0,
            'total_slippage': 0.0,
            'total_commission': 0.0,
            'slippage_history': [],  # List of (symbol, expected_price, fill_price, slippage_bps)
        }

    def record_execution_metrics(
        self,
        symbol: str,
        expected_price: float,
        fill_price: float,
        commission: float = 0.0
    ):
        """
        Record execution metrics for performance analysis.

        Args:
            symbol: Stock ticker
            expected_price: Price at signal generation
            fill_price: Actual fill price
            commission: Commission paid
        """
        if expected_price > 0:
            slippage_bps = ((fill_price - expected_price) / expected_price) * 10000
        else:
            slippage_bps = 0.0

        self.execution_metrics['total_trades'] += 1
        self.execution_metrics['total_slippage'] += abs(slippage_bps)
        self.execution_metrics['total_commission'] += commission

        self.execution_metrics['slippage_history'].append({
            'timestamp': datetime.now().isoformat(),
            'symbol': symbol,
            'expected_price': expected_price,
            'fill_price': fill_price,
            'slippage_bps': slippage_bps,
            'commission': commission
        })

        # Log significant slippage
        if abs(slippage_bps) > 10:  # > 10 bps
            logger.warning(
                f"⚠️  High slippage on {symbol}: {slippage_bps:+.1f} bps "
                f"(expected ${expected_price:.2f}, filled ${fill_price:.2f})"
            )
        else:
            logger.debug(
                f"📊 {symbol} slippage: {slippage_bps:+.1f} bps, commission: ${commission:.2f}"
            )

    def get_execution_summary(self) -> Dict:
        """Get summary of execution metrics."""
        metrics = self.execution_metrics
        n_trades = metrics['total_trades']

        if n_trades == 0:
            return {
                'total_trades': 0,
                'avg_slippage_bps': 0.0,
                'total_commission': 0.0,
                'avg_commission': 0.0
            }

        return {
            'total_trades': n_trades,
            'avg_slippage_bps': metrics['total_slippage'] / n_trades,
            'total_commission': metrics['total_commission'],
            'avg_commission': metrics['total_commission'] / n_trades,
            'recent_slippage': metrics['slippage_history'][-10:]  # Last 10 trades
        }
    
    @with_retry(retryable_exceptions=(ConnectionError, TimeoutError, OSError))
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
    
    @with_retry(max_retries=3, retryable_exceptions=(ConnectionError, TimeoutError))
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

    def _is_market_open(self) -> bool:
        """
        Check if US stock market is currently open.

        Uses UTC time to determine EST market hours (9:30 AM - 4:00 PM EST).
        Handles basic DST approximation.

        Returns:
            True if market is likely open, False otherwise
        """
        try:
            # Try using pytz for accurate timezone handling
            try:
                import pytz
                eastern = pytz.timezone('America/New_York')
                now_est = datetime.now(pytz.UTC).astimezone(eastern)
                hour = now_est.hour
                minute = now_est.minute
                weekday = now_est.weekday()
            except ImportError:
                # Fallback: Use UTC with EST offset (UTC-5, ignoring DST for simplicity)
                from datetime import timezone, timedelta
                utc_now = datetime.now(timezone.utc)
                est_offset = timedelta(hours=-5)
                now_est = utc_now + est_offset
                hour = now_est.hour
                minute = now_est.minute
                weekday = now_est.weekday()

            # Market closed on weekends
            if weekday >= 5:  # Saturday = 5, Sunday = 6
                return False

            # Market hours: 9:30 AM - 4:00 PM EST
            market_open = (hour > 9) or (hour == 9 and minute >= 30)
            market_close = hour < 16

            return market_open and market_close

        except Exception as e:
            logger.debug(f"Error checking market hours: {e}")
            # Default to True to allow trading in case of errors
            return True

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
            
            # Determine if market is open (US market hours in EST)
            is_market_hours = self._is_market_open()
            
            # Decision logic
            if force_market or is_market_hours:
                # Use market order during trading hours
                logger.info(f"   📈 Market order (hours={is_market_hours})")
                return await self._execute_market_order(symbol, side, quantity, expected_price=price)
            else:
                # Use limit order pre-market with confidence-based buffer
                return await self._execute_limit_order(symbol, side, quantity, price, confidence)
                
        except Exception as e:
            logger.error(f"❌ Error executing order {side} {quantity} {symbol}: {e}")
            return None

    async def _execute_market_order(
        self,
        symbol: str,
        side: str,
        quantity: int,
        expected_price: float = 0.0
    ) -> Optional[dict]:
        """
        Execute market order with slippage tracking.

        Args:
            symbol: Stock ticker
            side: 'BUY' or 'SELL'
            quantity: Number of shares
            expected_price: Price at signal generation (for slippage calculation)
        """
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
                    commission = ApexConfig.COMMISSION_PER_TRADE

                    # Record execution metrics (slippage + commission)
                    self.record_execution_metrics(
                        symbol=symbol,
                        expected_price=expected_price,
                        fill_price=fill_price,
                        commission=commission
                    )

                    logger.info(f"✅ {action} {quantity} {symbol} @ ${fill_price:.2f}")

                    return {
                        'symbol': symbol,
                        'side': side,
                        'quantity': quantity,
                        'price': fill_price,
                        'expected_price': expected_price,
                        'slippage_bps': ((fill_price - expected_price) / expected_price * 10000) if expected_price > 0 else 0,
                        'commission': commission,
                        'status': 'FILLED'
                    }

                elif trade.orderStatus.status in ['Cancelled', 'ApiCancelled', 'Inactive']:
                    logger.error(f"❌ Order {action} {quantity} {symbol} cancelled")
                    return None

            # Timeout - cancel the order using the trade object
            if trade.orderStatus.status != 'Filled':
                logger.warning(f"⚠️  Order timeout for {symbol}")
                try:
                    self.ib.cancelOrder(trade.order)
                except Exception as cancel_err:
                    logger.debug(f"Error cancelling order: {cancel_err}")
                return None

            fill_price = trade.orderStatus.avgFillPrice
            commission = ApexConfig.COMMISSION_PER_TRADE

            self.record_execution_metrics(symbol, expected_price, fill_price, commission)

            return {
                'symbol': symbol,
                'side': side,
                'quantity': quantity,
                'price': fill_price,
                'expected_price': expected_price,
                'slippage_bps': ((fill_price - expected_price) / expected_price * 10000) if expected_price > 0 else 0,
                'commission': commission,
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
        """
        Check if symbol has pending or active orders.

        Args:
            symbol: Stock ticker to check

        Returns:
            True if there's a pending order for this symbol
        """
        try:
            # Get all open trades (Trade objects have both order and status)
            open_trades = self.ib.openTrades()

            # Check if any trade exists for this symbol
            for trade in open_trades:
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
