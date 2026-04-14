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
from ib_insync import IB, Stock, Option, MarketOrder, LimitOrder, util, Contract, TagValue
try:
    from ib_insync import Forex, Crypto
except Exception as _import_err:
    logger_tmp = logging.getLogger(__name__)
    logger_tmp.warning("ib_insync Forex/Crypto import failed (%s: %s) — FX/Crypto contracts disabled", type(_import_err).__name__, _import_err)
    Forex = None
    Crypto = None
import pandas as pd

from config import ApexConfig
from execution.metrics_store import ExecutionMetricsStore
from core.symbols import AssetClass, parse_symbol, normalize_symbol, is_market_open
from monitoring.prometheus_metrics import PrometheusMetrics

logger = logging.getLogger(__name__)

# Process-wide semaphore: at most 1 IBKRConnector can be executing connectAsync()
# at the same time.  In dual-session mode two IBKRConnector instances exist; without
# this guard they both race to fill TWS's ~32 connection limit simultaneously.
_global_ibkr_connect_semaphore = asyncio.Semaphore(1)

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

        # Fatal error flag for unrecoverable API connection states (e.g., unaccepted disclaimers)
        self._fatal_error: Optional[str] = None
        self.ib.errorEvent += self._on_error

        # Cache for contracts (in-memory; reloaded from disk on startup)
        self.contracts = {}

        # Disk-backed contract cache: avoids qualifyContractsAsync timeouts on restart
        self._contract_cache_path = ApexConfig.DATA_DIR / "ibkr_contract_cache.json"
        self._load_contract_disk_cache()
        
        # Concurrency limit for qualifyContractsAsync storms on startup
        self._qualify_semaphore = asyncio.Semaphore(5)
        self._cache_lock = asyncio.Lock()

        # Market data request IDs
        self.req_id_counter = 0

        # Pacing protection (IBKR API limits)
        self._pace_lock = asyncio.Lock()
        self._last_req_ts = 0.0
        max_rps = max(1.0, float(getattr(ApexConfig, "IBKR_MAX_REQ_PER_SEC", 6)))
        self._min_req_interval = 1.0 / max_rps

        # Execution metrics tracking (Persistent)
        self.metrics_store = ExecutionMetricsStore(ApexConfig.DATA_DIR / "execution_metrics.json")
        
        # Event-driven streaming state (Centralized Stream Manager)
        self.tickers: Dict[str, Any] = {}
        self.stream_registry: Dict[int, Any] = {}  # conId -> ticker (Tracks all active reqMktData)
        self.active_streams: int = 0        
        # IBKR sets a hard limit on concurrent market data lines (typically 100).
        # We cap at 85 to leave room for TWS UI and snapshot requests.
        self.MAX_STREAMS = getattr(ApexConfig, "IBKR_MAX_STREAMS", 85)
        self._stream_lock = asyncio.Lock()
        
        # Callback for data updates (used by Data Watchdog)
        self.data_callback: Optional[Callable[[str], None]] = None

        # Real-time fill callback: called instantly on every IBKR execution detail.
        # Signature: on_fill_callback(symbol: str, side: str, filled_qty: float, avg_price: float)
        # Set by execution_loop.py to update self.positions without waiting for the poll loop.
        self.on_fill_callback: Optional[Callable[[str, str, float, float], None]] = None

        # Normalized broker pair mapping for paper trading
        self.ibkr_pair_map = self._build_ibkr_pair_map()
        
        # Track live ticks for pre-market staleness guard
        self.live_ticks_received_today: Dict[str, Any] = {}

    def has_live_tick_today(self, symbol: str) -> bool:
        from datetime import datetime, timezone
        try:
            normalized = self._normalize_symbol(symbol)
            today = datetime.now(timezone.utc).date()
            return self.live_ticks_received_today.get(normalized) == today
        except Exception:
            return False

    def _normalize_symbol(self, symbol: str) -> str:
        return normalize_symbol(symbol)

    def _symbol_from_contract(self, contract: Contract) -> str:
        sec_type = getattr(contract, "secType", "")
        symbol = getattr(contract, "symbol", "")
        currency = getattr(contract, "currency", "")

        if sec_type == "CASH":
            return f"FX:{symbol}/{currency}"
            
        if sec_type == "CRYPTO":
            # Normalise IBKR crypto (e.g., XRPUSD) to system format (CRYPTO:XRP/USD)
            if symbol.endswith(currency) and len(symbol) > len(currency):
                base = symbol[:-len(currency)]
                return f"CRYPTO:{base}/{currency}"
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
                is_pre_open = False
                if getattr(ApexConfig, "PRE_MARKET_STALENESS_GUARD", True):
                    try:
                        from datetime import datetime
                        import pytz
                        now_et = datetime.now(pytz.timezone('US/Eastern'))
                        if now_et.hour == 9 and 0 <= now_et.minute < 30:
                            is_pre_open = True
                    except Exception:
                        pass
                
                if is_pre_open:
                    logger.warning(f"⚠️ Stale price fallback: {normalized} @ {price:.4f} — {reason}. Live tick not yet received.")
                else:
                    logger.info("event=price_fallback symbol=%s price=%.4f reason=%s", normalized, price, reason)
                return float(price)
        except Exception:
            pass
        return 0.0

    async def _request_snapshot_price(self, contract, normalized: str) -> float:
        """
        Request a one-shot IBKR market-data snapshot (reqMktData snapshot=True).

        Used as a fallback when the streaming subscription returns no price after
        the initial wait.  IBKR throttling / pacing violations are handled with a
        5-second timeout; on timeout we gracefully return 0.0 so the calling cycle
        simply skips this symbol rather than crashing the execution loop.

        Returns:
            Price float > 0 on success, 0.0 on failure.
        """
        try:
            await self._throttle_ibkr("snapshot_price_fallback")
            ticker = self.ib.reqMktData(contract, '', False, True)  # True = snapshot
            price = 0.0
            # Poll up to 5 seconds (50 × 100ms) — longer than 3s to tolerate light
            # IBKR gateway lag and pacing delays without unnecessary failures.
            for _ in range(50):
                await asyncio.sleep(0.1)
                mp = ticker.marketPrice()
                if not pd.isna(mp) and mp > 0:
                    price = float(mp)
                    break
                last = getattr(ticker, 'last', 0)
                close = getattr(ticker, 'close', 0)
                bid = getattr(ticker, 'bid', 0)
                ask = getattr(ticker, 'ask', 0)
                if not pd.isna(last) and last > 0:
                    price = float(last)
                    break
                if not pd.isna(close) and close > 0:
                    price = float(close)
                    break
                if not pd.isna(bid) and not pd.isna(ask) and bid > 0 and ask > 0:
                    price = (float(bid) + float(ask)) / 2.0
                    break
            # Always cancel the one-shot subscription to avoid leaking subscriptions
            try:
                self.ib.cancelMktData(contract)
            except Exception:
                pass
            if price > 0:
                logger.info(
                    "event=snapshot_price_success symbol=%s price=%.4f",
                    normalized, price
                )
            else:
                logger.debug(
                    "event=snapshot_price_empty symbol=%s — skipping cycle for this symbol",
                    normalized
                )
            return price
        except asyncio.TimeoutError:
            logger.debug(
                "event=snapshot_price_timeout symbol=%s — skipping cycle gracefully",
                normalized
            )
            return 0.0
        except Exception as exc:
            logger.debug("event=snapshot_price_error symbol=%s error=%s", normalized, exc)
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

    def _on_error(self, reqId: int, errorCode: int, errorString: str, contract: Any):
        """Handle real-time errors from IBKR API to catch fatal connection conditions."""
        if errorCode == 10141:
            self._fatal_error = f"Error {errorCode}: {errorString}"
            logger.critical(f"❌ IBKR UNRECOVERABLE ERROR: {self._fatal_error}")
        elif errorCode in (201, 202):
            # 201: Order rejected by risk management
            # 202: Order cancelled by broker
            sym = getattr(contract, "symbol", "?") if contract is not None else "?"
            logger.error(
                "❌ IBKR order rejected (reqId=%d, code=%d, sym=%s): %s",
                reqId, errorCode, sym, errorString,
            )
        elif errorCode in (103, 110, 399, 10147, 10148):
            # 103: Invalid contract, 110: Invalid price, 399: cannot place
            # 10147/10148: account risk / position size limit
            sym = getattr(contract, "symbol", "?") if contract is not None else "?"
            logger.warning(
                "⚠️ IBKR order warning (reqId=%d, code=%d, sym=%s): %s",
                reqId, errorCode, sym, errorString,
            )
        elif errorCode < 1000 and errorCode not in (0, 2104, 2106, 2158):
            # Log other non-info error codes at debug level to avoid noise
            logger.debug("IBKR api message (reqId=%d, code=%d): %s", reqId, errorCode, errorString)

    def _on_disconnected(self):
        """Handle disconnection from IBKR."""
        if getattr(self, "_reconnect_task", None) and not self._reconnect_task.done():
            logger.debug("IBKR disconnected – reconnect already in progress, skipping duplicate")
            return
        
        if self._fatal_error:
            logger.error(f"IBKR disconnected due to fatal error. Reconnection ABORTED.")
            return

        logger.warning("IBKR disconnected. Attempting to reconnect...")
        self._reconnect_task = asyncio.create_task(self._reconnect_with_backoff())

    async def _reconnect_with_backoff(self):
        """Reconnect with retries; sets _persistently_down after max failures (Upgrade A)."""
        from config import ApexConfig
        max_retries = int(getattr(ApexConfig, "IBKR_FAILOVER_MAX_RETRIES", 3))
        retry_delay = float(getattr(ApexConfig, "IBKR_FAILOVER_RETRY_SECONDS", 30.0))

        await asyncio.sleep(3)  # brief pause for ib_insync cleanup
        for attempt in range(1, max_retries + 1):
            # Ensure the previous socket is fully closed before each reconnect attempt
            try:
                self.ib.disconnect()
            except Exception:
                pass
            await asyncio.sleep(1.0)  # Give TWS a moment to release the slot
            try:
                await self.connect()
                self._reconnect_failure_count = 0
                self._persistently_down = False
                logger.info("✅ IBKR reconnected successfully (attempt %d)", attempt)
                return
            except Exception as e:
                self._reconnect_failure_count = getattr(self, "_reconnect_failure_count", 0) + 1
                logger.warning("IBKR reconnect attempt %d/%d failed: %s", attempt, max_retries, e)
                if attempt < max_retries:
                    await asyncio.sleep(retry_delay)

        # All retries exhausted — flag as persistently down for failover
        self._persistently_down = True
        logger.error(
            "❌ IBKR persistently down after %d reconnect attempts – "
            "execution loop will degrade to Alpaca-only; background recovery loop started",
            max_retries,
        )
        # Spawn a persistent background recovery task that keeps retrying every 5 minutes
        # so that a TWS restart or network recovery is picked up without needing an engine restart.
        asyncio.ensure_future(self._persistent_recovery_loop())

    async def _persistent_recovery_loop(self):
        """
        Background loop that keeps retrying IBKR connection every IBKR_RECOVERY_INTERVAL_SECONDS
        (default: 300s) once _persistently_down has been set.  Exits when connection succeeds or
        when _fatal_error is set.
        """
        from config import ApexConfig
        interval = float(getattr(ApexConfig, "IBKR_RECOVERY_INTERVAL_SECONDS", 300.0))
        attempt = 0
        while self._persistently_down and not self._fatal_error:
            await asyncio.sleep(interval)
            if not self._persistently_down or self._fatal_error:
                return
            attempt += 1
            logger.info("🔄 IBKR recovery attempt #%d (background loop)...", attempt)
            try:
                self.ib.disconnect()
            except Exception:
                pass
            await asyncio.sleep(1.0)
            try:
                await self.connect()
                self._reconnect_failure_count = 0
                self._persistently_down = False
                logger.info("✅ IBKR recovered after %d background attempt(s)", attempt)
                return
            except Exception as e:
                logger.warning("🔁 IBKR recovery attempt #%d failed: %s — retrying in %.0fs", attempt, e, interval)

    def set_data_callback(self, callback: Callable[[str], None]):
        """Set callback to be called on every data update (for heartbeat monitoring)."""
        self.data_callback = callback
        
    def _on_data_update(self, tickers):
        """Internal handler for IBKR pendingTickers event."""
        from datetime import datetime, timezone
        today = datetime.now(timezone.utc).date()
        
        for ticker in tickers:
            if ticker.contract and ticker.contract.symbol:
                symbol = self._symbol_from_contract(ticker.contract)
                self.live_ticks_received_today[symbol] = today
                if self.data_callback:
                    self.data_callback(symbol)

    def _on_exec_details(self, trade, fill) -> None:
        """Real-time fill handler — fired instantly by ib_insync on every (partial) fill.

        ``trade`` is an ib_insync Trade object.  ``fill`` is an Execution object
        with fields: execution.side ('BOT'/'SLD'), execution.shares, execution.avgPrice.

        We parse the underlying contract into an APEX-normalized symbol and
        forward to ``on_fill_callback`` so execution_loop.py can update
        self.positions immediately — without waiting for the 2-5 minute
        reconciliation polling cycle.
        """
        try:
            contract = getattr(trade, "contract", None) or getattr(fill, "contract", None)
            if contract is None:
                return

            symbol = self._symbol_from_contract(contract)
            if not symbol:
                return

            exec_obj = getattr(fill, "execution", fill)     # fill IS the Execution in some versions
            side_raw = str(getattr(exec_obj, "side", "")).upper()   # 'BOT' or 'SLD'
            filled_qty = float(getattr(exec_obj, "shares", 0.0) or 0.0)
            avg_price = float(getattr(exec_obj, "avgPrice", 0.0) or 0.0)

            if filled_qty <= 0:
                return

            # Normalise IBKR's 'BOT'/'SLD' to 'BUY'/'SELL'
            side = "BUY" if side_raw in ("BOT", "BUY") else "SELL"

            logger.debug(
                "event=ibkr_fill_realtime symbol=%s side=%s qty=%.4f price=%.4f",
                symbol, side, filled_qty, avg_price,
            )

            if callable(self.on_fill_callback):
                self.on_fill_callback(symbol, side, filled_qty, avg_price)

        except Exception as exc:
            logger.warning("_on_exec_details error: %s", exc)

    def set_fill_callback(self, callback: Callable[[str, str, float, float], None]) -> None:
        """Register a fill callback to be called on every real-time IBKR execution detail.

        Signature: callback(symbol, side, filled_qty, avg_price)
        """
        self.on_fill_callback = callback

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

        # Record to Prometheus
        try:
            from monitoring.prometheus_metrics import PrometheusMetrics
            metrics = PrometheusMetrics()
            metrics.record_execution_slippage(abs(slippage_bps))
        except Exception:
            pass

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
        self._fatal_error = None
        # Ensure any prior socket is closed before opening a new connection.
        # Without this, each retry in the @with_retry loop leaves a zombie TCP
        # socket open at TWS, accumulating until the ~32-connection pool is full.
        try:
            self.ib.disconnect()
        except Exception:
            pass
        logger.info(f"🔌 Connecting to IBKR at {self.host}:{self.port}...")
        
        try:
            from execution.ibkr_lease_manager import lease_manager
            
            # Allocate a guaranteed unique client ID across cluster
            max_id_retries = 5
            current_client_id = self.client_id

            for id_attempt in range(max_id_retries):
                try:
                    # Preferred must be within the lease manager's valid range (min_id..max_id).
                    # Using self.client_id (default=1) + id_attempt stays below min_id=100,
                    # so the allocator always falls through to 100 and re-leases the same blocked ID.
                    preferred = lease_manager.min_id + id_attempt
                    current_client_id = await lease_manager.allocate(preferred)
                    self._active_client_id = current_client_id

                    logger.info(f"🔑 Lease manager returned client ID {current_client_id} (preferred={preferred})")
                    # Serialise connectAsync calls process-wide: only 1 IBKRConnector
                    # connects at a time.  In dual-session mode both core and crypto
                    # sessions previously raced here, exhausting TWS's ~32 slot limit.
                    async with _global_ibkr_connect_semaphore:
                        await asyncio.wait_for(
                            # timeout=15: Since TWS paper trading accounts have gigabytes of
                            # historical execution data, we cap the initial sync at 15s.
                            # ib_insync will cleanly catch these 15s inner timeouts, log
                            # "proceeding without full execution history", and return the connection!
                            self.ib.connectAsync(self.host, self.port, clientId=current_client_id, timeout=15),
                            timeout=240,
                        )

                    # Give TWS a brief moment to settle session state
                    await asyncio.sleep(1.0)
                    logger.info(f"✅ Connected to IBKR with clientId {current_client_id}")

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

                except (Exception, asyncio.TimeoutError, asyncio.CancelledError) as conn_err:
                    # Release the failed lease immediately
                    if getattr(self, "_active_client_id", None):
                        await lease_manager.release(self._active_client_id)
                        self._active_client_id = None

                    if self._fatal_error:
                        logger.error(f"❌ Aborting IBKR connection sequence due to fatal error: {self._fatal_error}")
                        raise RuntimeError(f"IBKR connect aborted: {self._fatal_error}")

                    err_str = str(conn_err)
                    is_collision = ("clientId" in err_str and "already in use" in err_str) or "rejected by peer" in err_str
                    is_timeout = (
                        isinstance(conn_err, (asyncio.TimeoutError, asyncio.CancelledError))
                        or "TimeoutError" in err_str
                        or "CancelledError" in err_str
                    )

                    if (is_collision or is_timeout) and id_attempt < max_id_retries - 1:
                        logger.warning(
                            "⚠️  Connection failed with clientId %d (%s), retrying allocation in 5s...",
                            current_client_id,
                            type(conn_err).__name__,
                        )
                        # Ensure ib_insync is fully disconnected before next attempt
                        try:
                            self.ib.disconnect()
                        except Exception:
                            pass
                        await asyncio.sleep(5)  # give TWS more time (was 2s)
                        continue
                    # Last attempt or non-retriable error — re-raise as plain Exception
                    # so the outer handler can catch it cleanly without cancelling tasks
                    raise RuntimeError(
                        f"IBKR connect failed (attempt {id_attempt + 1}/{max_id_retries}): {type(conn_err).__name__}: {conn_err}"
                    ) from conn_err
            
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
                logger.info("📋 Available Accounts: %s", ", ".join(accounts))
                logger.info("📋 Active Account: %s", self.account)
            
            logger.info(f"✅ Connected to Interactive Brokers (clientId={current_client_id})!")
            
            # Hook up data event listener
            self.ib.pendingTickersEvent += self._on_data_update
            self.ib.disconnectedEvent += self._on_disconnected
            # ─── Real-time fill listener ──────────────────────────────────
            # ib_insync fires execDetailsEvent immediately on every partial or
            # full fill. We use this to update local position state without
            # waiting for the periodic reconciliation loop.
            self.ib.execDetailsEvent += self._on_exec_details
            logger.info("✅ IBKR real-time fill listener registered (execDetailsEvent)")
            
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
    
    async def get_contract(self, symbol: str, require_qualified: bool = False) -> Optional[Contract]:
        """
        Get or create contract (equity, forex, or crypto).

        When ``require_qualified`` is False (default) the method returns an
        unqualified contract immediately if TWS doesn't respond to
        qualifyContractsAsync within the timeout — so the caller is never
        blocked for more than ~5 seconds on a broken TWS session.
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
            logger.debug(
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

            # If offline or not connected, return unqualified contract immediately
            if self.offline_mode or not self.ib.isConnected():
                self.contracts[key] = contract
                logger.debug("event=ibkr_offline_contract symbol=%s broker=%s", parsed.normalized, broker_parsed.normalized)
                return contract
                
            # IBKR does not reliably qualify standard CRYPTO streams via qualifyContractsAsync
            if parsed.asset_class == AssetClass.CRYPTO:
                self.contracts[key] = contract
                return contract

            # Qualify contract (get full details from IBKR)
            # Throttle only when we're going to the network.
            
            async with self._qualify_semaphore:
                await self._throttle_ibkr("qualify_contract")
                try:
                    qualified = await asyncio.wait_for(self.ib.qualifyContractsAsync(contract), timeout=15.0)
                except asyncio.TimeoutError:
                    logger.warning(
                        "⚠️  qualifyContractsAsync timed out for %s — using unqualified contract",
                        parsed.normalized,
                    )
                    # Cache the unqualified contract so we don't retry on every call
                    self.contracts[key] = contract
                    return contract if not require_qualified else None

                if qualified:
                    self.contracts[key] = qualified[0]
                    await self._save_contract_to_disk_cache(key, qualified[0])  # persist for next restart
                    return qualified[0]
                else:
                    if parsed.asset_class == AssetClass.FOREX:
                        for fallback in self._fx_fallback_contracts(broker_parsed):
                            await self._throttle_ibkr("qualify_contract_fx_fallback")
                            try:
                                fallback_qualified = await asyncio.wait_for(self.ib.qualifyContractsAsync(fallback), timeout=15.0)
                            except asyncio.TimeoutError:
                                fallback_qualified = []
                            if fallback_qualified:
                                self.contracts[key] = fallback_qualified[0]
                                await self._save_contract_to_disk_cache(key, fallback_qualified[0])  # persist
                                logger.warning(
                                    "⚠️  FX fallback contract qualified for %s via exchange=%s",
                                    parsed.normalized,
                                    getattr(fallback_qualified[0], "exchange", "unknown"),
                                )
                                return fallback_qualified[0]
                    # Fall back to unqualified
                    logger.warning(
                        "⚠️  Could not qualify contract for %s — using unqualified",
                        parsed.normalized,
                    )
                    self.contracts[key] = contract
                    return contract if not require_qualified else None

        except Exception as e:
            logger.error(f"❌ Error getting contract for {parsed.normalized}: {e}")
            return None
    
    # ─── Contract Disk Cache ────────────────────────────────────────────

    def _load_contract_disk_cache(self) -> None:
        """Load previously qualified contracts from disk into self.contracts."""
        import json
        from datetime import datetime, timedelta
        try:
            if not self._contract_cache_path.exists():
                return
            data = json.loads(self._contract_cache_path.read_text())
            now = datetime.utcnow()
            max_age = timedelta(days=7)
            loaded = 0
            for sym, entry in data.items():
                try:
                    cached_at = datetime.fromisoformat(entry["cached_at"])
                    if now - cached_at > max_age:
                        continue  # stale — re-qualify next time
                    contract = Contract(
                        conId=entry["conId"],
                        symbol=entry["symbol"],
                        secType=entry["secType"],
                        exchange=entry["exchange"],
                        currency=entry["currency"],
                    )
                    self.contracts[sym] = contract
                    loaded += 1
                except Exception:
                    continue
            if loaded:
                logger.info("💾 Loaded %d cached IBKR contracts from disk (skipping qualifyContractsAsync)", loaded)
        except Exception as exc:
            logger.debug("Contract disk cache load failed (non-fatal): %s", exc)

    async def _save_contract_to_disk_cache(self, normalized_symbol: str, contract) -> None:
        """Persist a newly qualified contract to the disk cache safely."""
        import json
        import fcntl
        from datetime import datetime
        async with self._cache_lock:
            try:
                if not self._contract_cache_path.exists():
                    self._contract_cache_path.write_text("{}")
                    
                with open(self._contract_cache_path, "r+") as f:
                    fcntl.flock(f, fcntl.LOCK_EX)
                    try:
                        try:
                            data = json.load(f)
                        except json.JSONDecodeError:
                            data = {}  # Overwrite corrupted file
                            
                        data[normalized_symbol] = {
                            "conId": getattr(contract, "conId", 0),
                            "symbol": getattr(contract, "symbol", ""),
                            "secType": getattr(contract, "secType", ""),
                            "exchange": getattr(contract, "exchange", ""),
                            "currency": getattr(contract, "currency", ""),
                            "cached_at": datetime.utcnow().isoformat(),
                        }
                        
                        f.seek(0)
                        f.truncate()
                        json.dump(data, f, indent=2)
                    finally:
                        fcntl.flock(f, fcntl.LOCK_UN)
            except Exception as exc:
                logger.warning("Contract disk cache save failed: %s", exc)

    # ─── Contract Lookup ────────────────────────────────────────────────

    async def _qualify_contract_semaphored(self, symbol: str, semaphore: asyncio.Semaphore) -> None:
        """Try to qualify a single contract, limiting concurrency via semaphore."""
        async with semaphore:
            await self.get_contract(symbol)  # result cached in self.contracts

    async def prewarm_contracts(self, symbols: list, concurrency: int = 5) -> None:
        """
        Pre-qualify a list of contracts concurrently.
        Uses a semaphore to cap in-flight TWS requests and avoids hammering TWS.
        """
        if self.offline_mode or not self.ib.isConnected():
            return
        sem = asyncio.Semaphore(concurrency)
        tasks = [
            self._qualify_contract_semaphored(s, sem)
            for s in symbols
            if parse_symbol(s).normalized not in self.contracts  # skip already cached
            if True  # parse may raise; get_contract handles it safely
        ]
        if tasks:
            logger.info("⚡ Pre-warming %d contracts (concurrency=%d)…", len(tasks), concurrency)
            await asyncio.gather(*tasks, return_exceptions=True)
            logger.info("✅ Contract pre-warm complete")

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
                return

            async with self._stream_lock:
                # 1. Normalize and deduplicate incoming symbols
                normalized_incoming = set()
                incoming_map = {}
                for s in symbols:
                    try:
                        norm = self._normalize_symbol(s)
                        if norm not in normalized_incoming:
                            normalized_incoming.add(norm)
                            incoming_map[norm] = s
                    except ValueError:
                        continue

                # 2. Hard limit check to protect against IBKR pacing violations
                if len(normalized_incoming) > self.MAX_STREAMS:
                    # Sort deterministically so we drop predictably if passed too many
                    sorted_incoming = list(normalized_incoming)[:self.MAX_STREAMS]
                    normalized_incoming = set(sorted_incoming)
                    logger.warning(
                        "⚠️ Capping incoming IBKR streams at %d (skipped %d)",
                        self.MAX_STREAMS, len(incoming_map) - self.MAX_STREAMS
                    )

                # 3. Identify differences
                current_active = set(self.tickers.keys())
                to_cancel = current_active - normalized_incoming
                to_add = normalized_incoming - current_active

                # 4. Cancel stale streams directly
                for sym in to_cancel:
                    ticker = self.tickers.pop(sym, None)
                    if ticker:
                        self.ib.cancelMktData(ticker.contract)
                        self.stream_registry.pop(ticker.contract.conId, None)

                # 5. Subscribe to new streams
                add_count = 0
                for sym in to_add:
                    original_symbol = incoming_map[sym]
                    contract = await self.get_contract(original_symbol)
                    if contract:
                        await self._throttle_ibkr("stream_mkt_data")
                        ticker = self.ib.reqMktData(contract, '', False, False)
                        self.tickers[sym] = ticker
                        self.stream_registry[contract.conId] = ticker
                        add_count += 1
                
                self.active_streams = len(self.tickers)
                
                if to_cancel or to_add:
                    logger.info(
                        "✅ IBKR Stream Sync: Cancelled %d, Added %d (Total Active: %d/%d)",
                        len(to_cancel), add_count, self.active_streams, self.MAX_STREAMS
                    )
                else:
                    logger.debug("⚡ IBKR Stream Sync: No changes required (Active: %d)", self.active_streams)
                    
        except Exception as e:
            logger.error(f"Error syncing streams: {e}")

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

            # Immediate check for Delayed-Frozen Type 4 availability
            if not ApexConfig.USE_LIVE_MARKET_DATA:
                price = ticker.marketPrice()
                if not pd.isna(price) and price > 0:
                    return float(price)
                
                # Check last/close safely with pd.isna
                last = getattr(ticker, 'last', 0)
                close = getattr(ticker, 'close', 0)
                if not pd.isna(last) and last > 0: return float(last)
                if not pd.isna(close) and close > 0: return float(close)
            
            # Wait for price data to arrive (max 5 seconds - increased from 2s to reduce noise)
            for _ in range(50):
                await asyncio.sleep(0.1)
                if not pd.isna(price) and price > 0:
                    logger.debug(f"✅ Got streaming price for {normalized}: ${price:.2f}")
                    return float(price)
                
                # Try all available price fields safely
                last = getattr(ticker, 'last', 0)
                close = getattr(ticker, 'close', 0)
                if not pd.isna(last) and last > 0:
                    price = float(last)
                elif not pd.isna(close) and close > 0:
                    price = float(close)
                else:
                    bid = getattr(ticker, 'bid', 0)
                    ask = getattr(ticker, 'ask', 0)
                    if not pd.isna(bid) and not pd.isna(ask) and bid > 0 and ask > 0:
                        price = (float(bid) + float(ask)) / 2.0
                
                if not pd.isna(price) and price > 0:
                    logger.debug(f"✅ Got fallback streaming price for {normalized}: ${price:.2f}")
                    return float(price)
            
            # Still no price after 5 seconds.
            # Step 1: Try a one-shot IBKR snapshot request (reqMktData snapshot=True).
            # This handles cases where the streaming subscription is stale but IBKR
            # can still serve a delayed/frozen quote via a snapshot pull (5s window).
            # If the snapshot also times out, we skip this symbol gracefully rather
            # than crashing the execution loop (per user feedback on IBKR pacing).
            logger.debug(f"⚠️  No streaming price for {normalized} after 5s — attempting snapshot fallback")
            if contract:
                snapshot_price = await self._request_snapshot_price(contract, normalized)
                if snapshot_price > 0:
                    return snapshot_price
            # Step 2: Data-provider fallback (only active when USE_DATA_FALLBACK_FOR_PRICES=True)
            logger.warning(f"⚠️  No streaming price for {normalized} after 5s — snapshot also empty")
            price = self._fallback_price(normalized, reason="ibkr_no_stream_price")
            if price > 0:
                return float(price)
            return 0.0
        except Exception as e:
            logger.debug(f"Error getting price for {symbol}: {e}")
            return 0.0

    def get_quote_age(self, symbol: str) -> float:
        """
        Returns the age of the cached quote in seconds. 
        Returns 999999.0 if no quote exists.
        """
        try:
            normalized = self._normalize_symbol(symbol)
            if normalized in self.tickers:
                ticker = self.tickers[normalized]
                if ticker.time:
                    # ticker.time is timezone-aware UTC datetime in ib_insync
                    from datetime import datetime, timezone
                    now = datetime.now(timezone.utc) if ticker.time.tzinfo else datetime.utcnow()
                    return (now - ticker.time).total_seconds()
        except:
            pass
        return 999999.0

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
            if bid > 0 and ask > 0:
                return {
                    "symbol": normalized,
                    "bid": bid,
                    "ask": ask,
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
                sec_type = str(getattr(pos.contract, "secType", "")).strip().upper()
                # Options are tracked separately via get_option_positions()
                # and must never pollute equity/spot position aggregation.
                if "OPT" in sec_type:
                    continue
                symbol = self._symbol_from_contract(pos.contract)
                qty = float(pos.position)
                avg_cost = float(pos.avgCost)
                
                if qty != 0:
                    result[symbol] = {
                        'qty': qty,
                        'avg_cost': avg_cost,
                        'security_type': sec_type or "EQUITY",
                    }
            
            return result
            
        except Exception as e:
            logger.error(f"❌ Error getting detailed positions: {e}")
            return {}

    async def get_net_liquidation(self) -> float:
        """Fetch the account's NetLiquidation value (NAV) from IBKR."""
        try:
            if not self.ib.isConnected():
                return 0.0
            
            summary = self.ib.accountSummary()
            for item in summary:
                if item.tag == 'NetLiquidation':
                    return float(item.value)
            
            # Fallback to get_account_values if summary is empty
            acc_values = self.ib.accountValues()
            for v in acc_values:
                if v.tag == 'NetLiquidation' and v.currency == 'USD':
                    return float(v.value)
                    
            return 0.0
        except Exception as e:
            logger.error(f"❌ Error getting IBKR NetLiquidation: {e}")
            return 0.0

    async def get_account_equity(self) -> float:
        """Alias for get_net_liquidation for consistent multi-venue cross-validation."""
        return await self.get_net_liquidation()
    
    async def ensure_delayed_data_mode(self):
        """Ensure delayed data mode is active."""
        try:
            if self.ib.isConnected():
                self.ib.reqMarketDataType(3)  # 3 = Delayed (15-20 min)
                logger.debug("✅ Delayed data mode re-confirmed")
        except Exception as e:
            logger.debug(f"Error setting delayed data mode: {e}")

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
            
            # Determine if market is open (delegates to core/market_hours.py)
            is_market_hours = is_market_open(symbol, datetime.now())

            # Decision logic — three tiers:
            #  1. force_market → immediate market sweep
            #  2. market hours + passive limit enabled → SOR passive pegging (Pillar 1B)
            #  3. pre/post-market → confidence-based limit (existing behaviour)
            passive_enabled = (
                not force_market
                and is_market_hours
                and getattr(ApexConfig, "SOR_ENABLED", True)
                and getattr(ApexConfig, "PASSIVE_LIMIT_ENABLED", True)
            )

            if force_market:
                logger.info("   📈 Market order (force_market=True)")
                return await self._execute_market_order(symbol, side, quantity, expected_price=price)
            elif passive_enabled:
                logger.info("   🎯 Passive limit order (SOR mid-peg, Pillar 1B)")
                return await self._execute_passive_limit_order(symbol, side, quantity, price)
            else:
                # Pre/post-market: confidence-based limit
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
            order = MarketOrder(action, quantity, outsideRth=True)
            if parsed.asset_class == AssetClass.FOREX:
                order.tif = 'GTC'

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
                        last_entry = trade.log[-1]
                        code_tag = f"[{last_entry.errorCode}] " if getattr(last_entry, "errorCode", 0) else ""
                        error_msg = f" - {code_tag}{last_entry.message}" if last_entry.message else ""
                    logger.error(f"❌ Order {action} {quantity} {symbol} cannot be placed{error_msg}")
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
            order = LimitOrder(action, quantity, limit_price, outsideRth=True)
            order.algoStrategy = 'Adaptive'
            order.algoParams = [TagValue('adaptivePriority', 'Normal')]
            if parsed.asset_class == AssetClass.FOREX:
                order.tif = 'GTC'
            
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
                error_msg = ""
                if trade.log:
                    last_entry = trade.log[-1]
                    code_tag = f"[{last_entry.errorCode}] " if getattr(last_entry, "errorCode", 0) else ""
                    error_msg = f" - {code_tag}{last_entry.message}" if last_entry.message else ""
                logger.error(f"❌ Limit order rejected for {symbol}, status: {status}{error_msg}")
                return None

        except Exception as e:
            logger.error(f"❌ Limit order error: {e}")
            return None
    
    async def _execute_passive_limit_order(
        self,
        symbol: str,
        side: str,
        quantity: float,
        expected_price: float,
    ) -> Optional[dict]:
        """
        Passive Limit Execution — Pillar 1B.

        Posts at the estimated mid-price and steps toward the touch in
        PASSIVE_LIMIT_MAX_STEPS increments before falling back to a market sweep.
        Captures approximately 50-80% of the bid-ask spread on average, reducing
        per-trade slippage by ~5-12bps vs outright market orders.

        Urgency schedule (default 3 steps × 10 s):
          Step 0  — mid-price                 (pure passive, full spread capture)
          Step 1  — mid ± 0.25 × spread       (quarter-touch)
          Step 2  — mid ± 0.50 × spread       (half-touch)
          Fallback — market sweep
        """
        try:
            try:
                parsed = parse_symbol(symbol)
                symbol = parsed.normalized
            except ValueError as e:
                logger.error("❌ Invalid symbol format %s: %s", symbol, e)
                return None

            contract = await self.get_contract(symbol)
            if not contract:
                logger.error("❌ Invalid contract for %s", symbol)
                return None

            action = "BUY" if side.upper() == "BUY" else "SELL"
            unit_label = self._unit_label(parsed.asset_class)

            # Estimate spread from config; avoids needing live L2 data.
            spread_bps = max(
                getattr(ApexConfig, "PASSIVE_LIMIT_MIN_SPREAD_BPS", 2),
                1,
            )
            estimated_spread = expected_price * spread_bps / 10_000.0
            max_steps = getattr(ApexConfig, "PASSIVE_LIMIT_MAX_STEPS", 3)
            step_secs = getattr(ApexConfig, "PASSIVE_LIMIT_STEP_SECONDS", 10)

            active_trade = None

            for step in range(max_steps):
                # Urgency 0 → 0.5 across the steps (never fully crosses to touch)
                urgency = (step / max(max_steps - 1, 1)) * 0.5
                if action == "BUY":
                    limit_price = round(expected_price + estimated_spread * urgency, 4)
                else:
                    limit_price = round(expected_price - estimated_spread * urgency, 4)

                phase = "passive mid-peg" if step == 0 else f"urgency={int(urgency*100)}%"
                logger.info(
                    "SOR [%s] %s %s — step %d/%d %s @ %.4f",
                    symbol, action, unit_label, step + 1, max_steps, phase, limit_price,
                )

                # Cancel previous tier if it exists
                if active_trade is not None:
                    try:
                        self.ib.cancelOrder(active_trade.order)
                        await asyncio.sleep(0.2)
                    except Exception:
                        pass

                order = LimitOrder(action, quantity, limit_price, outsideRth=True)
                order.algoStrategy = 'Adaptive'
                order.algoParams = [TagValue('adaptivePriority', 'Normal')]
                if parsed.asset_class == AssetClass.FOREX:
                    order.tif = 'GTC'
                active_trade = self.ib.placeOrder(contract, order)

                # Poll for fill during the patience window
                for _ in range(step_secs * 10):  # 100ms granularity
                    await asyncio.sleep(0.1)
                    status = active_trade.orderStatus.status
                    if status == "Filled":
                        fill_price = active_trade.orderStatus.avgFillPrice
                        self.record_execution_metrics(
                            symbol=symbol,
                            expected_price=expected_price,
                            fill_price=fill_price,
                            commission=ApexConfig.COMMISSION_PER_TRADE,
                        )
                        sign = 1 if action == "BUY" else -1
                        slippage_bps = (
                            sign * (fill_price - expected_price) / expected_price * 10_000
                            if expected_price > 0 else 0
                        )
                        logger.info(
                            "✅ SOR [%s] filled elegantly @ %.4f (slippage %.2f bps)",
                            symbol, fill_price, slippage_bps,
                        )
                        return {
                            "symbol": symbol,
                            "side": side,
                            "quantity": quantity,
                            "price": fill_price,
                            "expected_price": expected_price,
                            "slippage_bps": slippage_bps,
                            "commission": ApexConfig.COMMISSION_PER_TRADE,
                            "status": "FILLED",
                            "execution_algo": "PASSIVE_LIMIT",
                        }
                    if status in ["Cancelled", "ApiCancelled", "Inactive"]:
                        _sor_err = ""
                        if active_trade and active_trade.log:
                            _last = active_trade.log[-1]
                            _code_tag = f"[{_last.errorCode}] " if getattr(_last, "errorCode", 0) else ""
                            _sor_err = f" — {_code_tag}{_last.message}" if _last.message else ""
                        logger.warning("⚠️ SOR [%s] step %d rejected (%s)%s — falling back", symbol, step, status, _sor_err)
                        break  # Move to next step

            # Fallback — sweep the book with a market order
            logger.warning(
                "⏳ SOR [%s] patience exhausted after %d steps — market sweep",
                symbol, max_steps,
            )
            if active_trade is not None:
                try:
                    self.ib.cancelOrder(active_trade.order)
                    await asyncio.sleep(0.2)
                except Exception:
                    pass

            return await self._execute_market_order(symbol, side, quantity, expected_price)

        except Exception as e:
            logger.error("❌ Passive limit error for %s: %s", symbol, e)
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

            # IBKR requires year-based duration for requests beyond 365 days.
            if days > 365:
                years = max(1, int(math.ceil(days / 365)))
                duration_str = f"{years} Y"
            else:
                duration_str = f"{days} D"

            sec_type = getattr(contract, "secType", "").upper()
            if sec_type == "CASH":
                what_to_show = "MIDPOINT"
                use_rth = False
            elif sec_type == "CRYPTO":
                what_to_show = "AGGTRADES"
                use_rth = False
            else:
                what_to_show = "TRADES"
                use_rth = True
            
            # Request historical data
            bars = await self.ib.reqHistoricalDataAsync(
                contract,
                endDateTime='',
                durationStr=duration_str,
                barSizeSetting='1 day',
                whatToShow=what_to_show,
                useRTH=use_rth,
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
        expiry = str(expiry).replace("-", "").replace("/", "") # Format: YYYYMMDD
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
                try:
                    qualified = await asyncio.wait_for(
                        self.ib.qualifyContractsAsync(contract),
                        timeout=10
                    )
                except asyncio.TimeoutError:
                    logger.warning(f"🕒 Timeout qualifying option contract {symbol} {expiry} - Step {i+1}")
                    continue
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
        expiry = str(expiry).replace("-", "").replace("/", "")
        try:
            contract = await self.get_option_contract(symbol, expiry, strike, right)
            if not contract:
                return {}

            # Request market data via StreamManager to avoid Error 101
            await self._throttle_ibkr("snapshot_option_data")
            
            # Ensure we have room for this stream
            if len(self.stream_registry) >= self.MAX_STREAMS:
                # Evict the oldest non-position stream from self.tickers if possible
                evicted = False
                for sym, ticker in list(self.tickers.items()):
                    if ticker.contract.conId in self.stream_registry:
                        logger.debug(f"📐 Evicting stream {sym} to make room for option snapshot")
                        self.ib.cancelMktData(ticker.contract)
                        self.stream_registry.pop(ticker.contract.conId, None)
                        self.tickers.pop(sym, None)
                        evicted = True
                        break
                if not evicted and self.stream_registry:
                    # Just pick one and cancel it
                    cid, t = next(iter(self.stream_registry.items()))
                    self.ib.cancelMktData(t.contract)
                    self.stream_registry.pop(cid)

            ticker = self.ib.reqMktData(contract, '', False, False)
            self.stream_registry[contract.conId] = ticker

            # Wait for data (max 3 seconds)
            for _ in range(30):
                await asyncio.sleep(0.1)
                if ticker.bid and ticker.ask:
                    break

            self.ib.cancelMktData(contract)
            self.stream_registry.pop(contract.conId, None)

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
                if limit_price is None or limit_price <= 0:
                    limit_price = expected_price
                limit_price = self._finite_float(limit_price, 0.0)
                if limit_price <= 0:
                    limit_price = 0.01
                # Increase by a safety margin to ensure fill on paper (since we wanted MKT anyway)
                if side.upper() == 'BUY':
                    limit_price = limit_price * 1.05  # Pay 5% more
                else:
                    limit_price = limit_price * 0.95  # Accept 5% less
                logger.info(f"   🔄 Switching to LMT order for paper trading safety (${limit_price:.2f})")

            if order_type == 'MKT':
                order = MarketOrder(action, quantity, tif='DAY')
                logger.info(f"📈 Option Market Order: {action} {quantity} {symbol} {expiry} {strike}{right}")
            else:
                # Ensure we have a valid limit price
                limit_price = self._finite_float(limit_price, 0.0)
                if limit_price <= 0:
                    limit_price = expected_price if expected_price > 0 else 0.01  # Minimum tick

                # Round to 2 decimals for USD options
                limit_price = max(0.01, round(limit_price, 2))

                order = LimitOrder(action, quantity, limit_price, tif='DAY')
                order.algoStrategy = 'Adaptive'
                order.algoParams = [TagValue('adaptivePriority', 'Normal')]
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
            except Exception as cancel_err:
                logger.debug("cancelOrder failed during timeout cleanup: %s", cancel_err)
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
                if getattr(pos.contract, 'secType', '') == 'OPT':
                    contract = pos.contract
                    expiry_val = getattr(contract, 'lastTradeDateOrContractMonth', None)
                    if not expiry_val:
                        continue
                    import re
                    expiry = re.sub(r'[^0-9]', '', str(expiry_val).strip()) # Normalize strictly to YYYYMMDD
                    key = f"{contract.symbol}_{expiry}_{contract.strike}_{contract.right}"

                    result[key] = {
                        'symbol': contract.symbol,
                        'expiry': expiry,
                        'strike': contract.strike,
                        'right': contract.right,
                        'quantity': int(getattr(pos, 'position', 0)),
                        'avg_cost': getattr(pos, 'avgCost', 0.0)
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
        norm_expiry = str(expiry).replace("-", "").replace("/", "")
        key = f"{symbol}_{norm_expiry}_{strike}_{right}"

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

    def __del__(self):
        """Cleanup on deletion."""
        self.disconnect()
