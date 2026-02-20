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
import math
import random
import time
from datetime import datetime
from typing import Dict, Optional, Callable, TypeVar, Any
from functools import wraps
from ib_insync import IB, Stock, Option, MarketOrder, LimitOrder, util, Contract
try:
    from ib_insync import Forex, Crypto
except Exception:
    Forex = None
    Crypto = None
import pandas as pd

from config import ApexConfig
from pathlib import Path
from execution.metrics_store import ExecutionMetricsStore
from core.symbols import AssetClass, parse_symbol, normalize_symbol, is_market_open

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
        self.offline_mode = False

        # Cache for contracts
        self.contracts = {}

        # Market data request IDs
        self.req_id_counter = 0

        # Pacing protection (IBKR API limits)
        self._pace_lock = asyncio.Lock()
        self._last_req_ts = 0.0
        max_rps = max(1.0, float(getattr(ApexConfig, "IBKR_MAX_REQ_PER_SEC", 6)))
        self._min_req_interval = 1.0 / max_rps

        # Execution metrics tracking (Persistent)
        self.metrics_store = ExecutionMetricsStore(ApexConfig.DATA_DIR / "execution_metrics.json")
        
        # Event-driven streaming state
        self.tickers: Dict[str, Any] = {}
        self.active_streams: int = 0
        self.MAX_STREAMS = getattr(ApexConfig, "IBKR_MAX_STREAMS", 200)
        
        # Callback for data updates (used by Data Watchdog)
        self.data_callback: Optional[Callable[[str], None]] = None

        # Normalized broker pair mapping for paper trading
        self.ibkr_pair_map = self._build_ibkr_pair_map()

    def _normalize_symbol(self, symbol: str) -> str:
        return normalize_symbol(symbol)

    def _symbol_from_contract(self, contract: Contract) -> str:
        sec_type = getattr(contract, "secType", "")
        symbol = getattr(contract, "symbol", "")
        currency = getattr(contract, "currency", "")

        if sec_type == "CASH":
            return f"FX:{symbol}/{currency}"
        if sec_type == "CRYPTO":
            return f"CRYPTO:{symbol}/{currency}"
        return symbol

    def _unit_label(self, asset_class: AssetClass) -> str:
        return "shares" if asset_class == AssetClass.EQUITY else "units"

    @staticmethod
    def _finite_float(value: Any, default: float = 0.0) -> float:
        """Convert arbitrary value to finite float; fallback to default when invalid/NaN/Inf."""
        try:
            converted = float(value)
        except Exception:
            return float(default)
        if not math.isfinite(converted):
            return float(default)
        return float(converted)

    def _build_ibkr_pair_map(self) -> Dict[str, str]:
        mapping = {}
        raw_map = getattr(ApexConfig, "IBKR_PAIR_MAP", {}) or {}
        for key, value in raw_map.items():
            try:
                k = parse_symbol(key).normalized
                v = parse_symbol(value).normalized
            except ValueError:
                continue
            mapping[k] = v
        return mapping

    async def _throttle_ibkr(self, reason: str):
        """Throttle IBKR requests to avoid pacing violations."""
        async with self._pace_lock:
            now = time.time()
            elapsed = now - self._last_req_ts
            if elapsed < self._min_req_interval:
                await asyncio.sleep(self._min_req_interval - elapsed)
            self._last_req_ts = time.time()
            logger.debug("event=ibkr_throttle reason=%s interval=%.3fs", reason, self._min_req_interval)

    def _build_contract(self, parsed):
        if parsed.asset_class == AssetClass.EQUITY:
            return Stock(parsed.base, 'SMART', parsed.quote)
        if parsed.asset_class == AssetClass.FOREX:
            fx_exchange = getattr(ApexConfig, "IBKR_FX_EXCHANGE", "IDEALPRO")
            if Forex is not None:
                contract = Forex(f"{parsed.base}{parsed.quote}")
                contract.exchange = fx_exchange
                return contract
            return Contract(
                secType="CASH",
                symbol=parsed.base,
                currency=parsed.quote,
                exchange=fx_exchange,
            )
        crypto_exchange = getattr(ApexConfig, "IBKR_CRYPTO_EXCHANGE", "PAXOS")
        if Crypto is not None:
            # ib_insync Crypto signature: Crypto(symbol, exchange, currency)
            return Crypto(parsed.base, crypto_exchange, parsed.quote)
        return Contract(
            secType="CRYPTO",
            symbol=parsed.base,
            currency=parsed.quote,
            exchange=crypto_exchange,
        )

    def _fx_fallback_contracts(self, parsed) -> list:
        """Return explicit CASH contract fallbacks for FX qualification."""
        exchanges = []
        preferred = getattr(ApexConfig, "IBKR_FX_EXCHANGE", "IDEALPRO")
        for exch in (preferred, "IDEALPRO", "SMART"):
            if exch and exch not in exchanges:
                exchanges.append(exch)
        return [
            Contract(
                secType="CASH",
                symbol=parsed.base,
                currency=parsed.quote,
                exchange=exchange,
            )
            for exchange in exchanges
        ]

    def _fallback_price(self, normalized: str, reason: str) -> float:
        if not getattr(ApexConfig, "USE_DATA_FALLBACK_FOR_PRICES", False):
            return 0.0
        try:
            from data.market_data import MarketDataFetcher
            price = MarketDataFetcher().get_current_price(normalized)
            if price and price > 0:
                logger.info("event=price_fallback symbol=%s price=%.4f reason=%s", normalized, price, reason)
                return float(price)
        except Exception:
            pass
        return 0.0

    def _map_for_broker(self, parsed):
        if not getattr(ApexConfig, "IBKR_USE_PAIR_MAP", True):
            return parsed
        # Apply mapping only in paper mode (default: port 7497)
        if getattr(ApexConfig, "IBKR_PORT", 7497) != 7497:
            return parsed
        mapped = self.ibkr_pair_map.get(parsed.normalized)
        if not mapped:
            return parsed
        try:
            mapped_parsed = parse_symbol(mapped)
        except ValueError:
            return parsed
        logger.info(
            "event=symbol_mapping input=%s normalized=%s broker=%s",
            parsed.raw,
            parsed.normalized,
            mapped_parsed.normalized,
        )
        return mapped_parsed

    def set_data_callback(self, callback: Callable[[str], None]):
        """Set callback to be called on every data update (for heartbeat monitoring)."""
        self.data_callback = callback
        
    def _on_data_update(self, tickers):
        """Internal handler for IBKR pendingTickers event."""
        if self.data_callback:
            for ticker in tickers:
                if ticker.contract and ticker.contract.symbol:
                    self.data_callback(self._symbol_from_contract(ticker.contract))

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

        self.metrics_store.record_metrics(
            slippage_bps=slippage_bps,
            commission=commission,
            trade_details={
                'symbol': symbol,
                'expected_price': expected_price,
                'fill_price': fill_price,
                'slippage_bps': slippage_bps,
                'commission': commission
            }
        )

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
        metrics = self.metrics_store.get_metrics()
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
    
    def is_connected(self) -> bool:
        """Check if connected to IBKR."""
        return self.ib.isConnected() and not self.offline_mode

    @with_retry(retryable_exceptions=(ConnectionError, TimeoutError, OSError))
    async def connect(self):
        """Connect to Interactive Brokers with delayed market data."""
        logger.info(f"🔌 Connecting to IBKR at {self.host}:{self.port}...")
        
        try:
            from execution.ibkr_lease_manager import lease_manager
            
            # Allocate a guaranteed unique client ID across cluster
            max_id_retries = 3
            current_client_id = self.client_id
            
            for id_attempt in range(max_id_retries):
                try:
                    current_client_id = await lease_manager.allocate(self.client_id if id_attempt == 0 else None)
                    self._active_client_id = current_client_id
                    
                    await self.ib.connectAsync(self.host, self.port, clientId=current_client_id)
                    
                    # Start heartbeat task to keep lease alive
                    if getattr(self, "_heartbeat_task", None) is None:
                        async def _heartbeat():
                            while True:
                                try:
                                    await asyncio.sleep(60) # Renew every 60s
                                    if getattr(self, "_active_client_id", None):
                                        await lease_manager.heartbeat(self._active_client_id)
                                except asyncio.CancelledError:
                                    break
                                except Exception as e:
                                    logger.error(f"IBKR heartbeat failed: {e}")
                        
                        self._heartbeat_task = asyncio.create_task(_heartbeat())
                    break

                except (Exception, asyncio.TimeoutError) as conn_err:
                    # Release the failed lease immediately
                    if getattr(self, "_active_client_id", None):
                        await lease_manager.release(self._active_client_id)
                        self._active_client_id = None
                        
                    is_collision = "clientId" in str(conn_err) and "already in use" in str(conn_err)
                    is_timeout = isinstance(conn_err, asyncio.TimeoutError) or "TimeoutError" in str(conn_err)
                    
                    if (is_collision or is_timeout) and id_attempt < max_id_retries - 1:
                        logger.warning(f"⚠️  Connection failed with clientId {current_client_id} (collision/timeout), retrying allocation...")
                        await asyncio.sleep(1)
                        continue
                    raise
            
            # ✅ Phase 2.2: Configurable market data type
            # Type 1 = Live (requires paid subscription)
            # Type 2 = Frozen (last available)
            # Type 3 = Delayed (15-20 min delayed, FREE, but NO after-hours data)
            # Type 4 = Delayed-Frozen (delayed + frozen, works after-hours)
            if ApexConfig.USE_LIVE_MARKET_DATA:
                self.ib.reqMarketDataType(1)
                logger.info("📊 LIVE market data enabled (requires subscription)")
            else:
                # Use Type 4 instead of Type 3 to get frozen prices after-hours
                self.ib.reqMarketDataType(4)
                logger.info("📊 Delayed-Frozen market data enabled (free, works after-hours)")
            
            # Get account info
            await asyncio.sleep(1)  # Wait for account sync
            accounts = self.ib.wrapper.accounts
            if accounts:
                self.account = accounts[0]
            
            logger.info(f"✅ Connected to Interactive Brokers (clientId={current_client_id})!")
            logger.info(f"📋 Account: {self.account}")
            
            # Hook up data event listener
            self.ib.pendingTickersEvent += self._on_data_update
            
        except Exception as e:
            logger.error(f"❌ Failed to connect to IBKR: {e}")
            if getattr(ApexConfig, "IBKR_ALLOW_OFFLINE", False):
                self.offline_mode = True
                logger.warning("event=ibkr_offline mode=enabled reason=connect_failed")
                return
            raise
    
    def disconnect(self):
        """Disconnect from Interactive Brokers."""
        if self.ib.isConnected():
            self.ib.disconnect()
            logger.info("🔌 Disconnected from IBKR")
            
        # Stop lease heartbeat
        if getattr(self, "_heartbeat_task", None):
            self._heartbeat_task.cancel()
            self._heartbeat_task = None
            
        # Release the client_id lease
        if getattr(self, "_active_client_id", None):
            try:
                from execution.ibkr_lease_manager import lease_manager
                # Create a background task for the release since disconnect might be called sync
                try:
                    loop = asyncio.get_running_loop()
                    loop.create_task(lease_manager.release(self._active_client_id))
                except RuntimeError:
                    # No running loop, just execute it directly in a new one if possible or ignore
                    asyncio.run(lease_manager.release(self._active_client_id))
            except Exception as e:
                logger.debug(f"Failed to release IBKR lease gracefully: {e}")
            finally:
                self._active_client_id = None
    
    async def get_contract(self, symbol: str) -> Optional[Contract]:
        """
        Get or create contract (equity, forex, or crypto).
        
        Args:
            symbol: Stock ticker (e.g., 'AAPL')
            
        Returns:
            Stock contract or None if not found
        """
        try:
            parsed = parse_symbol(symbol)
        except ValueError as e:
            logger.error(f"❌ Invalid symbol format {symbol}: {e}")
            return None

        broker_parsed = self._map_for_broker(parsed)
        if broker_parsed.normalized != parsed.normalized:
            logger.info(
                "event=symbol_normalization input=%s normalized=%s broker=%s",
                parsed.raw,
                parsed.normalized,
                broker_parsed.normalized,
            )
        else:
            logger.info(
                "event=symbol_normalization input=%s normalized=%s broker=%s",
                parsed.raw,
                parsed.normalized,
                parsed.normalized,
            )

        key = parsed.normalized
        if key in self.contracts:
            return self.contracts[key]

        try:
            contract = self._build_contract(broker_parsed)

            # If offline or not connected, return unqualified contract for logging/tests
            if self.offline_mode or not self.ib.isConnected():
                self.contracts[key] = contract
                logger.info("event=ibkr_offline_contract symbol=%s broker=%s", parsed.normalized, broker_parsed.normalized)
                return contract

            # Qualify contract (get full details from IBKR)
            await self._throttle_ibkr("qualify_contract")
            qualified = await self.ib.qualifyContractsAsync(contract)
            
            if qualified:
                self.contracts[key] = qualified[0]
                return qualified[0]
            else:
                if parsed.asset_class == AssetClass.FOREX:
                    for fallback in self._fx_fallback_contracts(broker_parsed):
                        await self._throttle_ibkr("qualify_contract_fx_fallback")
                        fallback_qualified = await self.ib.qualifyContractsAsync(fallback)
                        if fallback_qualified:
                            self.contracts[key] = fallback_qualified[0]
                            logger.warning(
                                "⚠️  FX fallback contract qualified for %s via exchange=%s",
                                parsed.normalized,
                                getattr(fallback_qualified[0], "exchange", "unknown"),
                            )
                            return fallback_qualified[0]
                logger.warning(f"⚠️  Could not qualify contract for {parsed.normalized}")
                return None
                
        except Exception as e:
            logger.error(f"❌ Error getting contract for {parsed.normalized}: {e}")
            return None
    
    async def stream_quotes(self, symbols: list):
        """
        Start streaming quotes for a list of symbols (Free-Tier Safe).
        
        Args:
            symbols: List of symbols to stream
        """
        try:
            if self.offline_mode or not self.ib.isConnected():
                # Log normalization for non-equities to validate mapping in offline mode
                for symbol in symbols:
                    try:
                        parsed = parse_symbol(symbol)
                    except ValueError:
                        continue
                    if parsed.asset_class != AssetClass.EQUITY:
                        await self.get_contract(symbol)
                logger.warning("event=stream_quotes_skipped reason=ibkr_offline symbols=%d", len(symbols))
                self.tickers.clear()
                self.active_streams = 0
                return
            # Smart management: prioritize positions and top symbols
            # Stop existing streams if over limit or refreshing
            for ticker in list(self.tickers.values()):
                self.ib.cancelMktData(ticker.contract)
            
            self.tickers.clear()
            self.active_streams = 0
            
            # Take top N symbols to avoid pacing violations
            target_symbols = symbols[:self.MAX_STREAMS]
            skipped = len(symbols) - len(target_symbols)
            
            if skipped > 0:
                logger.warning(f"⚠️  Capping streams at {self.MAX_STREAMS} (skipping {skipped} symbols)")
            
            count = 0
            for symbol in target_symbols:
                try:
                    normalized = self._normalize_symbol(symbol)
                except ValueError:
                    logger.warning(f"⚠️  Skipping invalid symbol for streaming: {symbol}")
                    continue

                contract = await self.get_contract(symbol)
                if contract:
                    # snapshot=False enables streaming
                    await self._throttle_ibkr("stream_mkt_data")
                    ticker = self.ib.reqMktData(contract, '', False, False)
                    self.tickers[normalized] = ticker
                    count += 1
            
            self.active_streams = count
            logger.info(f"✅ Started streaming {count} symbols (Delayed-Frozen Type 4)")
            
        except Exception as e:
            logger.error(f"Error starting streams: {e}")

    @with_retry(max_retries=3, retryable_exceptions=(ConnectionError, TimeoutError))
    async def get_market_price(self, symbol: str) -> float:
        """
        Get current market price using STREAMING (not snapshots).
        This works better with IBKR Paper accounts.
        """
        try:
            try:
                normalized = self._normalize_symbol(symbol)
            except ValueError:
                logger.error(f"❌ Invalid symbol format for price: {symbol}")
                return 0.0

            # If market closed, prefer data fallback (weekends/holidays)
            if getattr(ApexConfig, "PRICE_FALLBACK_WHEN_MARKET_CLOSED", False):
                try:
                    if not is_market_open(normalized, datetime.utcnow()):
                        price = self._fallback_price(normalized, reason="market_closed")
                        if price > 0:
                            return price
                except Exception:
                    pass

            # If IBKR is offline, rely on data provider fallback
            if self.offline_mode or not self.ib.isConnected():
                price = self._fallback_price(normalized, reason="ibkr_offline")
                return float(price) if price > 0 else 0.0

            # 1. Check if we already have an active stream for this symbol
            if normalized in self.tickers:
                ticker = self.tickers[normalized]
                price = ticker.marketPrice()
                
                # If marketPrice is invalid (nan/0), try specific fields
                if pd.isna(price) or price <= 0:
                    if ticker.last and ticker.last > 0: 
                        price = ticker.last
                    elif ticker.close and ticker.close > 0: 
                        price = ticker.close
                    elif ticker.bid and ticker.ask and ticker.bid > 0 and ticker.ask > 0:
                        price = (ticker.bid + ticker.ask) / 2.0
                    
                if price > 0:
                    return float(price)
            
            # 2. No active stream exists - create a temporary STREAMING request (not snapshot)
            contract = await self.get_contract(symbol)
            if not contract:
                price = self._fallback_price(normalized, reason="no_contract")
                return float(price) if price > 0 else 0.0
            
            # Request STREAMING data (snapshot=False) instead of snapshot
            await self._throttle_ibkr("stream_temp_price")
            ticker = self.ib.reqMktData(contract, '', False, False)  # False = streaming!
            
            # Store the stream for future use
            self.tickers[normalized] = ticker
            
            # Wait for price data to arrive (max 3 seconds)
            for _ in range(30):
                await asyncio.sleep(0.1)
                price = ticker.marketPrice()
                
                # Try all available price fields
                if pd.isna(price) or price <= 0:
                    if ticker.last and ticker.last > 0:
                        price = ticker.last
                    elif ticker.close and ticker.close > 0:
                        price = ticker.close
                    elif ticker.bid and ticker.ask and ticker.bid > 0 and ticker.ask > 0:
                        price = (ticker.bid + ticker.ask) / 2.0
                
                if price > 0:
                    logger.debug(f"✅ Got streaming price for {normalized}: ${price:.2f}")
                    return float(price)
            
            # Still no price after 3 seconds - fall back to data provider
            logger.warning(f"⚠️  No streaming price for {normalized} after 3s")
            price = self._fallback_price(normalized, reason="ibkr_no_stream_price")
            if price > 0:
                return float(price)
            return 0.0
            
        except Exception as e:
            logger.debug(f"Error getting price for {symbol}: {e}")
            return 0.0

    async def get_quote(self, symbol: str) -> Dict[str, float]:
        """Get latest bid/ask/mid quote for spread controls."""
        try:
            try:
                normalized = self._normalize_symbol(symbol)
            except ValueError:
                return {}

            if self.offline_mode or not self.ib.isConnected():
                return {}

            if normalized in self.tickers:
                ticker = self.tickers[normalized]
                bid = float(ticker.bid or 0.0)
                ask = float(ticker.ask or 0.0)
                last = float(ticker.last or 0.0)
                if bid > 0 and ask > 0:
                    return {
                        "symbol": normalized,
                        "bid": bid,
                        "ask": ask,
                        "mid": (bid + ask) / 2.0,
                        "last": last,
                    }

            contract = await self.get_contract(symbol)
            if not contract:
                return {}

            await self._throttle_ibkr("snapshot_quote")
            ticker = self.ib.reqMktData(contract, '', False, True)
            bid = 0.0
            ask = 0.0
            last = 0.0
            for _ in range(15):
                await asyncio.sleep(0.1)
                bid = float(ticker.bid or 0.0)
                ask = float(ticker.ask or 0.0)
                last = float(ticker.last or 0.0)
                if bid > 0 and ask > 0:
                    break
            try:
                self.ib.cancelMktData(contract)
            except Exception:
                pass

            if bid > 0 and ask > 0:
                return {
                    "symbol": normalized,
                    "bid": bid,
                    "ask": ask,
                    "mid": (bid + ask) / 2.0,
                    "last": last,
                }
            return {}
        except Exception:
            return {}
    
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
    
    async def get_position(self, symbol: str) -> float:
        """
        Get current position quantity for a symbol.
        
        Args:
            symbol: Trading symbol
            
        Returns:
            Position quantity (positive=long, negative=short, 0=no position)
        """
        try:
            try:
                parsed = parse_symbol(symbol)
                normalized = parsed.normalized
                broker_parsed = self._map_for_broker(parsed)
            except ValueError:
                logger.error(f"❌ Invalid symbol format for position lookup: {symbol}")
                return 0.0

            positions = self.ib.positions()
            
            for pos in positions:
                # Ignore options (handled separately)
                if pos.contract.secType == 'OPT':
                    continue

                pos_symbol = self._symbol_from_contract(pos.contract)
                if pos_symbol == broker_parsed.normalized or pos_symbol == normalized:
                    return float(pos.position)
            
            return 0.0
            
        except Exception as e:
            logger.error(f"❌ Error getting position for {symbol}: {e}")
            return 0.0
    
    async def get_all_positions(self) -> Dict[str, float]:
        """
        Get all current positions (long AND short).
        
        Returns:
            Dictionary of {symbol: quantity}
        """
        try:
            positions = self.ib.positions()
            result = {}
            for pos in positions:
                # Ignore options (handled separately in get_option_positions)
                sec_type = str(pos.contract.secType).strip().upper()
                if 'OPT' in sec_type:
                    continue

                symbol = self._symbol_from_contract(pos.contract)
                qty = float(pos.position)
                if qty != 0:
                    result[symbol] = qty
            
            if result:
                logger.info(f"📊 Loaded {len(result)} existing positions")
            return result
        except Exception as e:
            logger.error(f"❌ Error getting positions: {e}")
            return {}

    async def get_detailed_positions(self) -> Dict[str, Dict]:
        """
        Get all current positions with metadata (avgCost, etc).
        
        Returns:
            Dictionary of {symbol: {'qty': float, 'avg_cost': float}}
        """
        try:
            positions = self.ib.positions()
            
            result = {}
            for pos in positions:
                symbol = self._symbol_from_contract(pos.contract)
                qty = float(pos.position)
                avg_cost = float(pos.avgCost)
                
                if qty != 0:
                    result[symbol] = {
                        'qty': qty,
                        'avg_cost': avg_cost
                    }
            
            return result
            
        except Exception as e:
            logger.error(f"❌ Error getting detailed positions: {e}")
            return {}
    
    async def ensure_delayed_data_mode(self):
        """Ensure delayed data mode is active."""
        try:
            if self.ib.isConnected():
                self.ib.reqMarketDataType(3)  # 3 = Delayed (15-20 min)
                logger.debug("✅ Delayed data mode re-confirmed")
        except Exception as e:
            logger.debug(f"Error setting delayed data mode: {e}")

    def _is_market_open(self, asset_class: AssetClass = AssetClass.EQUITY) -> bool:
        """
        Check if market is currently open for the given asset class.

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

            # Crypto trades 24/7
            if asset_class == AssetClass.CRYPTO:
                return True

            # Forex trades 24/5 (Sunday 5pm ET to Friday 5pm ET)
            if asset_class == AssetClass.FOREX:
                if weekday == 5:  # Saturday
                    return False
                if weekday == 6:  # Sunday
                    return hour >= 17
                if weekday == 4:  # Friday
                    return hour < 17
                return True

            # Equities: Market closed on weekends
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

    async def execute_order(self, symbol: str, side: str, quantity: float, 
                        confidence: float = 0.5, force_market: bool = False) -> Optional[dict]:
        """
        ✅ FIXED: Execute order - LONG AND SHORT enabled with safety checks.
        
        Args:
            symbol: Trading symbol
            side: 'BUY' or 'SELL'
            quantity: Quantity (always positive)
            confidence: Signal confidence (0-1) - affects limit buffer
            force_market: If True, always use market order
            
        Returns:
            Trade details or None if failed
        """
        from datetime import datetime
        
        try:
            try:
                parsed = parse_symbol(symbol)
                symbol = parsed.normalized
            except ValueError as e:
                logger.error(f"❌ Invalid symbol format {symbol}: {e}")
                return None

            unit_label = self._unit_label(parsed.asset_class)

            # ✅ Safety check: Validate side
            if side not in ['BUY', 'SELL']:
                logger.error(f"❌ Invalid order side '{side}'. Only BUY/SELL allowed")
                logger.warning("event=order_rejected symbol=%s reason=invalid_side side=%s", symbol, side)
                return None
            
            # ✅ Safety check: Validate quantity
            try:
                qty = float(quantity)
            except Exception:
                logger.error(f"❌ Invalid quantity {quantity}. Must be numeric")
                logger.warning("event=order_rejected symbol=%s reason=invalid_quantity", symbol)
                return None

            if qty <= 0:
                logger.error(f"❌ Invalid quantity {quantity}. Must be positive")
                logger.warning("event=order_rejected symbol=%s reason=non_positive_quantity quantity=%s", symbol, quantity)
                return None

            if parsed.asset_class == AssetClass.EQUITY and not qty.is_integer():
                logger.error(f"❌ Fractional shares not supported for equity orders: {qty}")
                logger.warning("event=order_rejected symbol=%s reason=fractional_equity quantity=%s", symbol, qty)
                return None

            quantity = int(qty) if parsed.asset_class == AssetClass.EQUITY else qty
            
            # ✅ Safety check: Get current position
            current_pos = await self.get_position(symbol)

            # ✅ CRITICAL: Block adding to existing positions (prevents duplicates)
            if side == 'BUY' and current_pos > 0:
                logger.error(f"🚫 {symbol}: BLOCKED - Already have LONG position ({current_pos}). Cannot add more.")
                logger.warning("event=order_rejected symbol=%s reason=existing_long position=%s", symbol, current_pos)
                return None
            if side == 'SELL' and current_pos < 0:
                logger.error(f"🚫 {symbol}: BLOCKED - Already have SHORT position ({current_pos}). Cannot add more.")
                logger.warning("event=order_rejected symbol=%s reason=existing_short position=%s", symbol, current_pos)
                return None

            # ✅ Warn if creating/increasing short position
            if side == 'SELL':
                if current_pos == 0:
                    logger.warning(f"⚠️  {symbol}: Opening SHORT position ({quantity} {unit_label})")
                elif current_pos > 0:
                    if quantity > current_pos:
                        short_qty = quantity - current_pos
                        logger.warning(f"⚠️  {symbol}: Flipping to SHORT by {short_qty} {unit_label}")
                    else:
                        logger.info(f"✅ {symbol}: Reducing LONG position")
                elif current_pos < 0:
                    logger.warning(f"⚠️  {symbol}: Increasing SHORT position (now {abs(current_pos + quantity)} {unit_label} short)")
            
            # Get current price
            price = await self.get_market_price(symbol)
            if price == 0:
                logger.error(f"❌ Cannot execute: no price for {symbol}")
                logger.warning("event=order_rejected symbol=%s reason=no_price", symbol)
                return None
            
            # Determine if market is open (US market hours in EST)
            is_market_hours = self._is_market_open(parsed.asset_class)
            
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
        quantity: float,
        expected_price: float = 0.0
    ) -> Optional[dict]:
        """
        Execute market order with slippage tracking.

        Args:
            symbol: Trading symbol
            side: 'BUY' or 'SELL'
            quantity: Quantity
            expected_price: Price at signal generation (for slippage calculation)
        """
        try:
            try:
                parsed = parse_symbol(symbol)
                symbol = parsed.normalized
            except ValueError as e:
                logger.error(f"❌ Invalid symbol format {symbol}: {e}")
                return None

            unit_label = self._unit_label(parsed.asset_class)

            contract = await self.get_contract(symbol)
            if not contract:
                logger.error(f"❌ Invalid contract for {symbol}")
                return None

            # Create market order
            action = 'BUY' if side.upper() == 'BUY' else 'SELL'
            order = MarketOrder(action, quantity)

            # Place order
            trade = self.ib.placeOrder(contract, order)

            # Wait for order to fill (max 30 seconds for less liquid stocks)
            for _ in range(300):
                await asyncio.sleep(0.1)

                if trade.orderStatus.status == 'Filled':
                    fill_price = trade.orderStatus.avgFillPrice
                    commission = ApexConfig.COMMISSION_PER_TRADE
                    logger.info(
                        "event=fee_model asset=%s symbol=%s commission=%.4f",
                        parsed.asset_class.value,
                        symbol,
                        commission,
                    )

                    # Record execution metrics (slippage + commission)
                    self.record_execution_metrics(
                        symbol=symbol,
                        expected_price=expected_price,
                        fill_price=fill_price,
                        commission=commission
                    )

                    logger.info(f"✅ {action} {quantity} {unit_label} {symbol} @ ${fill_price:.2f}")

                    # Slippage: positive = adverse (BUY filled higher, SELL filled lower)
                    sign = 1 if action == 'BUY' else -1
                    slippage = (sign * (fill_price - expected_price) / expected_price * 10000) if expected_price > 0 else 0

                    return {
                        'symbol': symbol,
                        'side': side,
                        'quantity': quantity,
                        'price': fill_price,
                        'expected_price': expected_price,
                        'slippage_bps': slippage,
                        'commission': commission,
                        'status': 'FILLED'
                    }

                elif trade.orderStatus.status in ['Cancelled', 'ApiCancelled', 'Inactive']:
                    error_msg = ""
                    if trade.log:
                        error_msg = f" - {trade.log[-1].message}"
                    logger.error(f"❌ Order {action} {quantity} {symbol} cancelled{error_msg}")
                    return None

            # Timeout - cancel the order
            logger.warning(f"⚠️  Order timeout for {symbol}")
            try:
                self.ib.cancelOrder(trade.order)
            except Exception as cancel_err:
                logger.debug(f"Error cancelling order: {cancel_err}")
            return None
            
        except Exception as e:
            logger.error(f"❌ Market order error: {e}")
            return None

    async def _execute_limit_order(self, symbol: str, side: str, quantity: float, 
                                current_price: float, confidence: float) -> Optional[dict]:
        """
        Execute limit order with confidence-based buffer.
        
        Args:
            symbol: Trading symbol
            side: 'BUY' or 'SELL'
            quantity: Quantity
            current_price: Current market price
            confidence: Signal confidence (0-1)
        """
        try:
            try:
                parsed = parse_symbol(symbol)
                symbol = parsed.normalized
            except ValueError as e:
                logger.error(f"❌ Invalid symbol format {symbol}: {e}")
                return None

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
                logger.info(
                    "event=fee_model asset=%s symbol=%s commission=%.4f",
                    parsed.asset_class.value,
                    symbol,
                    ApexConfig.COMMISSION_PER_TRADE,
                )
                
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
                logger.info(
                    "event=fee_model asset=%s symbol=%s commission=%.4f",
                    parsed.asset_class.value,
                    symbol,
                    ApexConfig.COMMISSION_PER_TRADE,
                )
                
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
        Get TotalCashValue + AccruedCash (Gross Cash) in account.
        Used for Equity Reconciliation (Cash + Positions = Equity).
        """
        try:
            account_values = self.ib.accountValues()
            
            total_cash = 0.0
            accrued = 0.0
            
            for av in account_values:
                # Prefer TotalCashValue (Gross Cash)
                if av.tag == 'TotalCashValue' and av.currency == 'USD':
                    total_cash = float(av.value)
                elif av.tag == 'AccruedCash' and av.currency == 'USD':
                    accrued = float(av.value)
            
            # Fallback to CashBalance if TotalCashValue not found
            if total_cash == 0.0:
                for av in account_values:
                    if av.tag == 'CashBalance' and av.currency == 'USD':
                        total_cash = float(av.value)

            return total_cash + accrued
            
        except Exception as e:
            logger.error(f"❌ Error getting account cash: {e}")
            return 0.0

    def get_portfolio_market_values(self) -> Dict[str, float]:
        """
        Get market value of all positions from IBKR portfolio updates.
        Returns dict {key: marketValue} for OPTIONS.
        Key format matches get_option_positions: Symbol_Expiry_Strike_Right
        """
        try:
            # portfolio() returns list of PortfolioItem
            items = self.ib.portfolio()
            result = {}
            for item in items:
                contract = item.contract
                if contract.secType == 'OPT':
                    key = f"{contract.symbol}_{contract.lastTradeDateOrContractMonth}_{contract.strike}_{contract.right}"
                    result[key] = item.marketValue
            
            return result
        except Exception as e:
            logger.error(f"❌ Error getting portfolio values: {e}")
            return {}
    
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

# PSEUDO-TESTS (paper trading mapping)
# ApexConfig.IBKR_PAIR_MAP = {"BTC/USDT": "BTC/USD"}
# parse_symbol("BTC/USDT").normalized -> "CRYPTO:BTC/USDT"
# _map_for_broker(parse_symbol("BTC/USDT")).normalized -> "CRYPTO:BTC/USD"
    
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
    
    # ========== OPTIONS TRADING METHODS ==========

    async def get_option_contract(
        self,
        symbol: str,
        expiry: str,  # Format: YYYYMMDD
        strike: float,
        right: str,  # 'C' for call, 'P' for put
        trading_class: str = "",
        multiplier: int = 100
    ) -> Optional[Option]:
        """
        Get or create an options contract.

        Args:
            symbol: Underlying symbol (e.g., 'AAPL')
            expiry: Expiration date in YYYYMMDD format
            strike: Strike price
            right: 'C' for call, 'P' for put

        Returns:
            Option contract or None if not found
        """
        cache_key = f"{symbol}_{expiry}_{strike}_{right}_{trading_class}"

        if cache_key in self.contracts:
            return self.contracts[cache_key]

        try:
            # Try qualification with several fallback strategies
            attempts = [
                # 1. Original request with tradingClass + SMART
                dict(symbol=symbol, lastTradeDateOrContractMonth=expiry,
                     strike=strike, right=right, multiplier=str(multiplier),
                     tradingClass=trading_class, exchange='SMART', currency='USD'),
            ]
            # 2. Without tradingClass (let IBKR resolve)
            if trading_class:
                attempts.append(
                    dict(symbol=symbol, lastTradeDateOrContractMonth=expiry,
                         strike=strike, right=right, multiplier=str(multiplier),
                         tradingClass='', exchange='SMART', currency='USD'))
            # 3. Without explicit exchange (let IBKR pick)
            attempts.append(
                dict(symbol=symbol, lastTradeDateOrContractMonth=expiry,
                     strike=strike, right=right, multiplier=str(multiplier),
                     tradingClass='', exchange='', currency='USD'))

            for i, params in enumerate(attempts):
                contract = Option(**params)
                await self._throttle_ibkr("qualify_option_contract")
                qualified = await self.ib.qualifyContractsAsync(contract)
                if qualified:
                    self.contracts[cache_key] = qualified[0]
                    if i > 0:
                        logger.debug(f"✅ Qualified option on fallback attempt {i+1}: {symbol} {expiry} {strike} {right}")
                    else:
                        logger.debug(f"✅ Qualified option: {symbol} {expiry} {strike} {right}")
                    return qualified[0]

            logger.warning(f"⚠️ Could not qualify option contract: {symbol} Exp:{expiry} Strike:{strike} Right:{right}")
            return None

        except Exception as e:
            logger.error(f"❌ Error getting option contract: {e}")
            return None

    async def get_option_chain(
        self,
        symbol: str
    ) -> Optional[list]:
        """
        Get available option chain parameters for a symbol.

        Args:
            symbol: Underlying symbol

        Returns:
            List of chain definitions or None
        """
        try:
            contract = await self.get_contract(symbol)
            if not contract:
                return None

            # Request option chain parameters
            chains = self.ib.reqSecDefOptParams(
                contract.symbol,
                '',
                contract.secType,
                contract.conId
            )

            await asyncio.sleep(1)  # Wait for response

            if chains:
                logger.info(f"📊 Retrieved option chain for {symbol}: {len(chains)} exchanges")
                return chains
            else:
                logger.warning(f"⚠️ No option chains for {symbol}")
                return None

        except Exception as e:
            logger.error(f"❌ Error getting option chain for {symbol}: {e}")
            return None

    async def get_option_price(
        self,
        symbol: str,
        expiry: str,
        strike: float,
        right: str
    ) -> dict:
        """
        Get current option price and Greeks.

        Args:
            symbol: Underlying symbol
            expiry: Expiration date (YYYYMMDD)
            strike: Strike price
            right: 'C' or 'P'

        Returns:
            Dict with bid, ask, last, delta, gamma, theta, vega, iv
        """
        try:
            contract = await self.get_option_contract(symbol, expiry, strike, right)
            if not contract:
                return {}

            # Request market data
            await self._throttle_ibkr("snapshot_option_data")
            ticker = self.ib.reqMktData(contract, '', False, False)

            # Wait for data (max 3 seconds)
            for _ in range(30):
                await asyncio.sleep(0.1)
                if ticker.bid and ticker.ask:
                    break

            self.ib.cancelMktData(contract)

            bid = self._finite_float(getattr(ticker, "bid", 0.0), 0.0)
            ask = self._finite_float(getattr(ticker, "ask", 0.0), 0.0)
            last = self._finite_float(getattr(ticker, "last", 0.0), 0.0)
            mid = (bid + ask) / 2.0 if bid > 0 and ask > 0 else 0.0
            if mid <= 0:
                mid = last if last > 0 else (bid if bid > 0 else ask)

            # Get model Greeks if available
            delta = self._finite_float(getattr(ticker.modelGreeks, 'delta', 0), 0.0) if ticker.modelGreeks else 0.0
            gamma = self._finite_float(getattr(ticker.modelGreeks, 'gamma', 0), 0.0) if ticker.modelGreeks else 0.0
            theta = self._finite_float(getattr(ticker.modelGreeks, 'theta', 0), 0.0) if ticker.modelGreeks else 0.0
            vega = self._finite_float(getattr(ticker.modelGreeks, 'vega', 0), 0.0) if ticker.modelGreeks else 0.0
            iv = self._finite_float(getattr(ticker.modelGreeks, 'impliedVol', 0), 0.0) if ticker.modelGreeks else 0.0

            return {
                'bid': bid,
                'ask': ask,
                'last': last,
                'mid': mid,
                'delta': delta,
                'gamma': gamma,
                'theta': theta,
                'vega': vega,
                'implied_vol': iv
            }

        except Exception as e:
            logger.error(f"❌ Error getting option price: {e}")
            return {}

    async def execute_option_order(
        self,
        symbol: str,
        expiry: str,
        strike: float,
        right: str,
        side: str,
        quantity: int,
        order_type: str = 'MKT',
        limit_price: float = None,
        trading_class: str = "",
        multiplier: int = 100
    ) -> Optional[dict]:
        """
        Execute an options order.

        Args:
            symbol: Underlying symbol
            expiry: Expiration date (YYYYMMDD)
            strike: Strike price
            right: 'C' for call, 'P' for put
            side: 'BUY' or 'SELL'
            quantity: Number of contracts
            order_type: 'MKT' for market, 'LMT' for limit
            limit_price: Required for limit orders

        Returns:
            Trade result or None
        """
        try:
            # Validate inputs
            if side not in ['BUY', 'SELL']:
                logger.error(f"❌ Invalid side: {side}")
                return None

            if quantity <= 0:
                logger.error(f"❌ Invalid quantity: {quantity}")
                return None

            if order_type == 'LMT' and limit_price is None:
                logger.error("❌ Limit price required for limit orders")
                return None

            # Get option contract
            contract = await self.get_option_contract(
                symbol, expiry, strike, right, 
                trading_class=trading_class, 
                multiplier=multiplier
            )
            if not contract:
                return None

            # Get current price for reference
            price_data = await self.get_option_price(symbol, expiry, strike, right)
            expected_price = self._finite_float(price_data.get('mid', 0.0), 0.0)
            if expected_price <= 0:
                expected_price = self._finite_float(price_data.get('last', 0.0), 0.0)

            # Create order
            action = 'BUY' if side.upper() == 'BUY' else 'SELL'

            # Use LMT by default for paper trading safety with delayed data
            if order_type == 'MKT' and not ApexConfig.USE_LIVE_MARKET_DATA:
                order_type = 'LMT'
                if limit_price is None:
                    limit_price = expected_price
                limit_price = self._finite_float(limit_price, 0.0)
                if limit_price <= 0:
                    limit_price = 0.01
                logger.info(f"   🔄 Switching to LMT order for paper trading safety (${limit_price:.2f})")

            if order_type == 'MKT':
                order = MarketOrder(action, quantity)
                logger.info(f"📈 Option Market Order: {action} {quantity} {symbol} {expiry} {strike}{right}")
            else:
                # Ensure we have a valid limit price
                limit_price = self._finite_float(limit_price, 0.0)
                if limit_price <= 0:
                    limit_price = expected_price if expected_price > 0 else 0.01  # Minimum tick

                # Round to 2 decimals for USD options
                limit_price = max(0.01, round(limit_price, 2))
                
                order = LimitOrder(action, quantity, limit_price)
                logger.info(f"💰 Option Limit Order: {action} {quantity} {symbol} {expiry} {strike}{right} @ ${limit_price:.2f}")

            # Place order
            trade = self.ib.placeOrder(contract, order)

            # Wait for fill (max 15 seconds for options)
            for _ in range(150):
                await asyncio.sleep(0.1)

                if trade.orderStatus.status == 'Filled':
                    fill_price = trade.orderStatus.avgFillPrice
                    commission = ApexConfig.COMMISSION_PER_TRADE * 0.65  # Options typically cheaper

                    logger.info(f"✅ Option {action} {quantity} {symbol} {expiry} {strike}{right} @ ${fill_price:.2f}")

                    return {
                        'symbol': symbol,
                        'expiry': expiry,
                        'strike': strike,
                        'right': right,
                        'side': side,
                        'quantity': quantity,
                        'price': fill_price,
                        'expected_price': expected_price,
                        'commission': commission,
                        'status': 'FILLED',
                        'order_id': trade.order.orderId
                    }

                elif trade.orderStatus.status in ['Cancelled', 'ApiCancelled', 'Inactive']:
                    error_msg = ""
                    if trade.log:
                        # Find the first entry with an error message
                        for entry in reversed(trade.log):
                            if entry.message:
                                error_msg = f" - {entry.message}"
                                break
                    logger.error(f"❌ Option order cancelled: {symbol} {expiry} {strike}{right}{error_msg}")
                    return None

            # Timeout
            logger.warning(f"⚠️ Option order timeout: {symbol} {expiry} {strike}{right}")
            try:
                self.ib.cancelOrder(trade.order)
            except:
                pass
            return None

        except Exception as e:
            logger.error(f"❌ Option order error: {e}")
            return None

    async def get_option_positions(self) -> Dict[str, dict]:
        """
        Get all current option positions.

        Returns:
            Dict of {contract_key: position_info}
        """
        try:
            positions = self.ib.positions()

            result = {}
            for pos in positions:
                if pos.contract.secType == 'OPT':
                    contract = pos.contract
                    key = f"{contract.symbol}_{contract.lastTradeDateOrContractMonth}_{contract.strike}_{contract.right}"

                    result[key] = {
                        'symbol': contract.symbol,
                        'expiry': contract.lastTradeDateOrContractMonth,
                        'strike': contract.strike,
                        'right': contract.right,
                        'quantity': int(pos.position),
                        'avg_cost': pos.avgCost
                    }

                    pos_type = "LONG" if pos.position > 0 else "SHORT"
                    opt_type = "CALL" if contract.right == 'C' else "PUT"
                    logger.debug(f"   {contract.symbol} {contract.lastTradeDateOrContractMonth} ${contract.strike} {opt_type}: {abs(int(pos.position))} ({pos_type})")

            if result:
                logger.info(f"📊 Loaded {len(result)} option positions")

            return result

        except Exception as e:
            logger.error(f"❌ Error getting option positions: {e}")
            return {}

    async def close_option_position(
        self,
        symbol: str,
        expiry: str,
        strike: float,
        right: str,
        quantity: int
    ) -> Optional[dict]:
        """
        Close an existing option position.

        Args:
            symbol: Underlying symbol
            expiry: Expiration date
            strike: Strike price
            right: 'C' or 'P'
            quantity: Number of contracts to close (always positive)

        Returns:
            Trade result or None
        """
        # Get current position
        positions = await self.get_option_positions()
        key = f"{symbol}_{expiry}_{strike}_{right}"

        if key not in positions:
            logger.warning(f"⚠️  Cannot close {key}: Position not found")
            return None
            
        current_qty = positions[key]['quantity']

        # Determine closing action
        if current_qty > 0:
            side = 'SELL'  # Close long
        else:
            side = 'BUY'  # Close short

        close_qty = min(abs(quantity), abs(current_qty))

        logger.info(f"🚪 Closing option position: {side} {close_qty} {symbol} {expiry} {strike}{right}")

        return await self.execute_option_order(
            symbol=symbol,
            expiry=expiry,
            strike=strike,
            right=right,
            side=side,
            quantity=close_qty
        )

    def get_portfolio_market_values(self) -> Dict[str, float]:
        """
        Get market value of all positions from IBKR portfolio updates.
        Returns dict {key: marketValue} for OPTIONS.
        Key format matches get_option_positions: Symbol_Expiry_Strike_Right
        """
        try:
            # portfolio() returns list of PortfolioItem
            # PortfolioItem(contract, position, marketPrice, marketValue, averageCost, unrealizedPNL, realizedPNL, account)
            items = self.ib.portfolio()
            result = {}
            for item in items:
                contract = item.contract
                if contract.secType == 'OPT':
                    key = f"{contract.symbol}_{contract.lastTradeDateOrContractMonth}_{contract.strike}_{contract.right}"
                    result[key] = item.marketValue
            
            return result
        except Exception as e:
            logger.error(f"❌ Error getting portfolio values: {e}")
            return {}
    def __del__(self):
        """Cleanup on deletion."""
        self.disconnect()
