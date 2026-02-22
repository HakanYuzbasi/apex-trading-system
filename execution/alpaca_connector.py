"""
execution/alpaca_connector.py - Alpaca Crypto Paper Trading Connector

Handles crypto paper trading via Alpaca Markets REST API.

Features:
- Crypto paper trading (BTC, ETH, SOL, etc.)
- Async HTTP via httpx
- Symbol mapping: APEX format (CRYPTO:BTC/USDT) -> Alpaca format (BTC/USD)
- Execution metrics tracking (matching IBKRConnector patterns)
- Rate limiting with exponential backoff
- Price polling for live quotes
"""

import asyncio
import logging
import time
from datetime import datetime
from typing import Dict, Optional, Callable, Any, List

import httpx

from config import ApexConfig
from core.symbols import AssetClass, parse_symbol, normalize_symbol

logger = logging.getLogger(__name__)


class AlpacaConnector:
    """
    Alpaca Markets connector for crypto paper trading.

    Implements the same public interface as IBKRConnector for
    compatibility with the APEX trading system.
    """

    PAPER_BASE_URL = "https://paper-api.alpaca.markets"
    DATA_BASE_URL = "https://data.alpaca.markets"

    def __init__(
        self,
        api_key: str = "",
        secret_key: str = "",
        base_url: str = "",
    ):
        self.api_key = api_key or getattr(ApexConfig, "ALPACA_API_KEY", "")
        self.secret_key = secret_key or getattr(ApexConfig, "ALPACA_SECRET_KEY", "")
        self.base_url = base_url or getattr(
            ApexConfig, "ALPACA_BASE_URL", self.PAPER_BASE_URL
        )

        self._client: Optional[httpx.AsyncClient] = None
        self._connected = False
        self.offline_mode = False

        # Rate limiting (Alpaca: 200 req/min)
        self._pace_lock = asyncio.Lock()
        self._last_req_ts = 0.0
        self._min_req_interval = 60.0 / 200.0  # ~0.3s

        # Price cache
        self._price_cache: Dict[str, float] = {}
        self._quote_task: Optional[asyncio.Task] = None

        # Pending orders tracking
        self._pending_orders: Dict[str, str] = {}  # symbol -> order_id

        # Execution metrics (same structure as IBKRConnector)
        self.execution_metrics: Dict[str, Any] = {
            "total_trades": 0,
            "total_slippage": 0.0,
            "total_commission": 0.0,
            "slippage_history": [],
        }

        # Data callback for watchdog heartbeat
        self.data_callback: Optional[Callable[[str], None]] = None

        # Symbol mapping (APEX -> Alpaca)
        self._pair_map = self._build_pair_map()

    # ------------------------------------------------------------------
    # Symbol mapping
    # ------------------------------------------------------------------

    def _build_pair_map(self) -> Dict[str, str]:
        """Build mapping from APEX normalized symbols to Alpaca symbols."""
        mapping: Dict[str, str] = {}
        raw_map = getattr(ApexConfig, "DATA_PAIR_MAP", {}) or {}
        for key, value in raw_map.items():
            try:
                k = parse_symbol(key).normalized
            except ValueError:
                continue
            mapping[k] = value
        return mapping

    def _to_alpaca_symbol(self, symbol: str) -> str:
        """Convert APEX symbol to Alpaca crypto symbol (e.g. BTC/USD)."""
        try:
            parsed = parse_symbol(symbol)
        except ValueError:
            return symbol

        mapped = self._pair_map.get(parsed.normalized)
        if mapped:
            return mapped

        if parsed.asset_class == AssetClass.CRYPTO:
            return f"{parsed.base}/USD"

        return parsed.base

    # Known crypto base assets for reverse mapping
    _CRYPTO_BASES = {
        "BTC", "ETH", "SOL", "ADA", "XRP", "DOT", "LTC", "BCH",
        "DOGE", "AVAX", "LINK", "MATIC", "XLM", "XMR", "ETC",
        "AAVE", "UNI", "SHIB", "ATOM",
    }

    def _from_alpaca_symbol(self, alpaca_symbol: str) -> str:
        """Convert Alpaca symbol back to APEX format (e.g. CRYPTO:BTC/USD)."""
        if "/" in alpaca_symbol:
            return f"CRYPTO:{alpaca_symbol}"
        # Handle condensed format like "BTCUSD" -> "CRYPTO:BTC/USD"
        for base in self._CRYPTO_BASES:
            if alpaca_symbol.startswith(base):
                quote = alpaca_symbol[len(base):]
                if quote in ("USD", "USDT", "USDC"):
                    return f"CRYPTO:{base}/{quote}"
        return alpaca_symbol

    # ------------------------------------------------------------------
    # HTTP layer
    # ------------------------------------------------------------------

    @property
    def _headers(self) -> dict:
        return {
            "APCA-API-KEY-ID": self.api_key,
            "APCA-API-SECRET-KEY": self.secret_key,
        }

    async def _throttle(self):
        """Rate-limit requests."""
        async with self._pace_lock:
            now = time.time()
            elapsed = now - self._last_req_ts
            if elapsed < self._min_req_interval:
                await asyncio.sleep(self._min_req_interval - elapsed)
            self._last_req_ts = time.time()

    async def _request(
        self,
        method: str,
        path: str,
        base_url: str = "",
        json_body: Optional[dict] = None,
        params: Optional[dict] = None,
    ) -> Any:
        """Make authenticated API request with retry."""
        if not self._client or not self._connected:
            raise ConnectionError("Not connected. Call connect() first.")

        url = f"{base_url or self.base_url}{path}"
        await self._throttle()

        max_retries = 3
        for attempt in range(max_retries + 1):
            try:
                resp = await self._client.request(
                    method,
                    url,
                    headers=self._headers,
                    json=json_body,
                    params=params,
                    timeout=15.0,
                )

                if resp.status_code in (200, 201, 204):
                    if resp.status_code == 204:
                        return {}
                    return resp.json()
                elif resp.status_code == 429:
                    wait = 2 ** attempt
                    logger.warning(f"Rate limited by Alpaca, waiting {wait}s")
                    await asyncio.sleep(wait)
                    continue
                elif resp.status_code == 404:
                    return {}
                else:
                    text = resp.text
                    logger.error(f"Alpaca API error {resp.status_code}: {text}")
                    if resp.status_code >= 500 and attempt < max_retries:
                        await asyncio.sleep(2 ** attempt)
                        continue
                    return {}
            except (httpx.RequestError, asyncio.TimeoutError) as e:
                if "client has been closed" in str(e).lower():
                    self._connected = False
                    self._client = None
                    logger.warning("Alpaca HTTP client was closed; marking connector disconnected")
                    return {}
                if attempt < max_retries:
                    logger.warning(
                        f"Alpaca request failed (attempt {attempt + 1}): {e}"
                    )
                    await asyncio.sleep(2 ** attempt)
                else:
                    logger.error(
                        f"Alpaca request failed after {max_retries + 1} attempts: {e}"
                    )
                    return {}
        return {}

    # ------------------------------------------------------------------
    # Connection management
    # ------------------------------------------------------------------

    async def connect(self) -> None:
        """Connect to Alpaca Paper Trading API."""
        if not self.api_key or not self.secret_key:
            raise ConnectionError(
                "Alpaca API credentials not set. "
                "Set APEX_ALPACA_API_KEY and APEX_ALPACA_SECRET_KEY env vars."
            )

        logger.info("Connecting to Alpaca Paper Trading API...")

        try:
            self._client = httpx.AsyncClient()
            account = await self._request("GET", "/v2/account")

            if not account or "id" not in account:
                raise ConnectionError(
                    "Failed to authenticate with Alpaca. Check API credentials."
                )

            self._connected = True
            status = account.get("status", "N/A")
            equity = float(account.get("equity", 0))
            buying_power = float(account.get("buying_power", 0))

            logger.info("Connected to Alpaca Paper Trading")
            logger.info(f"  Account: {account.get('account_number', 'N/A')}")
            logger.info(f"  Status:  {status}")
            logger.info(f"  Equity:  ${equity:,.2f}")
            logger.info(f"  Buying Power: ${buying_power:,.2f}")

            if account.get("crypto_status") == "ACTIVE":
                logger.info("  Crypto:  ACTIVE")
            else:
                logger.warning(
                    f"  Crypto status: {account.get('crypto_status', 'UNKNOWN')} "
                    "— crypto trading may not be enabled"
                )

        except ConnectionError:
            raise
        except Exception as e:
            logger.error(f"Failed to connect to Alpaca: {e}")
            if getattr(ApexConfig, "ALPACA_ALLOW_OFFLINE", False):
                self.offline_mode = True
                self._connected = True
                logger.warning("Alpaca offline mode enabled")
                return
            raise

    def disconnect(self) -> None:
        """Disconnect from Alpaca."""
        if self._quote_task and not self._quote_task.done():
            self._quote_task.cancel()
        client = self._client
        self._client = None
        if client:
            # Schedule async close
            try:
                loop = asyncio.get_running_loop()
                loop.create_task(client.aclose())
            except RuntimeError:
                pass
        self._connected = False
        logger.info("Disconnected from Alpaca")

    def is_connected(self) -> bool:
        """Check if connected to Alpaca."""
        return self._connected and not self.offline_mode

    def set_data_callback(self, callback: Callable[[str], None]) -> None:
        """Set callback for data update notifications."""
        self.data_callback = callback

    # ------------------------------------------------------------------
    # Market data
    # ------------------------------------------------------------------

    async def get_market_price(self, symbol: str) -> float:
        """Get current crypto price from Alpaca."""
        try:
            normalized = normalize_symbol(symbol)

            # Check cache first
            cached = self._price_cache.get(normalized, 0.0)
            if cached > 0:
                return cached

            if self.offline_mode:
                return self._fallback_price(normalized)
            if not self._connected or self._client is None:
                return self._fallback_price(normalized)

            alpaca_sym = self._to_alpaca_symbol(symbol)

            data = await self._request(
                "GET",
                "/v1beta3/crypto/us/latest/quotes",
                base_url=self.DATA_BASE_URL,
                params={"symbols": alpaca_sym},
            )

            quotes = data.get("quotes", {})
            quote = quotes.get(alpaca_sym, {})

            bp = float(quote.get("bp", 0))  # bid
            ap = float(quote.get("ap", 0))  # ask

            if bp > 0 and ap > 0:
                price = (bp + ap) / 2.0
                self._price_cache[normalized] = price
                if self.data_callback:
                    self.data_callback(normalized)
                return price

            logger.warning(f"No price for {symbol} from Alpaca")
            return self._fallback_price(normalized)

        except Exception as e:
            logger.error(f"Error getting price for {symbol}: {e}")
            return 0.0

    async def get_quote(self, symbol: str) -> Dict[str, float]:
        """Get latest bid/ask/mid quote for spread controls."""
        try:
            normalized = normalize_symbol(symbol)
            if self.offline_mode:
                return {}
            if not self._connected or self._client is None:
                return {}

            alpaca_sym = self._to_alpaca_symbol(symbol)
            data = await self._request(
                "GET",
                "/v1beta3/crypto/us/latest/quotes",
                base_url=self.DATA_BASE_URL,
                params={"symbols": alpaca_sym},
            )
            quotes = data.get("quotes", {})
            quote = quotes.get(alpaca_sym, {})
            bid = float(quote.get("bp", 0.0) or 0.0)
            ask = float(quote.get("ap", 0.0) or 0.0)
            if bid <= 0 or ask <= 0:
                return {}
            return {
                "symbol": normalized,
                "bid": bid,
                "ask": ask,
                "mid": (bid + ask) / 2.0,
                "last": float(quote.get("ap", 0.0) or 0.0),
            }
        except Exception:
            return {}

    def _fallback_price(self, normalized: str) -> float:
        """Fallback to MarketDataFetcher if Alpaca unavailable."""
        if not getattr(ApexConfig, "USE_DATA_FALLBACK_FOR_PRICES", False):
            return 0.0
        try:
            from data.market_data import MarketDataFetcher

            price = MarketDataFetcher().get_current_price(normalized)
            if price and price > 0:
                return float(price)
        except Exception:
            pass
        return 0.0

    async def stream_quotes(self, symbols: list) -> None:
        """Start polling crypto quotes (replaces IBKR streaming)."""
        crypto_symbols = []
        for sym in symbols:
            try:
                parsed = parse_symbol(sym)
                if parsed.asset_class == AssetClass.CRYPTO:
                    crypto_symbols.append(sym)
            except ValueError:
                continue

        if not crypto_symbols:
            logger.info("No crypto symbols to stream via Alpaca")
            return

        logger.info(
            f"Starting Alpaca crypto quote polling for {len(crypto_symbols)} symbols"
        )

        if self._quote_task and not self._quote_task.done():
            self._quote_task.cancel()

        self._quote_task = asyncio.create_task(
            self._poll_quotes_loop(crypto_symbols)
        )

    async def _poll_quotes_loop(self, symbols: list) -> None:
        """Background polling loop for crypto quotes."""
        while True:
            try:
                for symbol in symbols:
                    # Clear cache to force fresh fetch
                    normalized = normalize_symbol(symbol)
                    self._price_cache.pop(normalized, None)
                    await self.get_market_price(symbol)
                await asyncio.sleep(10)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Quote polling error: {e}")
                await asyncio.sleep(30)

    # ------------------------------------------------------------------
    # Order execution
    # ------------------------------------------------------------------

    async def execute_order(
        self,
        symbol: str,
        side: str,
        quantity: float,
        confidence: float = 0.5,
        force_market: bool = False,
    ) -> Optional[dict]:
        """Execute a crypto order on Alpaca Paper."""
        try:
            parsed = parse_symbol(symbol)
            symbol = parsed.normalized

            if side not in ("BUY", "SELL"):
                logger.error(f"Invalid order side: {side}")
                return None
            if quantity <= 0:
                logger.error(f"Invalid quantity: {quantity}")
                return None

            expected_price = await self.get_market_price(symbol)
            if expected_price <= 0:
                logger.error(f"Cannot execute: no price for {symbol}")
                return None

            alpaca_sym = self._to_alpaca_symbol(symbol)

            order_body = {
                "symbol": alpaca_sym,
                "qty": str(quantity),
                "side": side.lower(),
                "type": "market",
                "time_in_force": "gtc",
            }

            logger.info(
                f"Placing {side} {quantity} {alpaca_sym} @ ~${expected_price:.2f} (Alpaca)"
            )

            data = await self._request("POST", "/v2/orders", json_body=order_body)

            if not data or "id" not in data:
                logger.error(f"Order rejected by Alpaca: {data}")
                return None

            order_id = data["id"]
            status = data.get("status", "")

            # For market orders, poll until filled
            if status in ("new", "accepted", "pending_new"):
                fill_data = await self._wait_for_fill(order_id, timeout=30)
                if fill_data:
                    data = fill_data

            filled_qty = float(data.get("filled_qty", 0))
            filled_price = float(data.get("filled_avg_price", 0))

            if filled_qty > 0 and filled_price > 0:
                sign = 1 if side == "BUY" else -1
                slippage = (
                    (sign * (filled_price - expected_price) / expected_price * 10000)
                    if expected_price > 0
                    else 0
                )

                commission = getattr(ApexConfig, "COMMISSION_PER_TRADE", 0.0)

                self.record_execution_metrics(
                    symbol=symbol,
                    expected_price=expected_price,
                    fill_price=filled_price,
                    commission=commission,
                )

                logger.info(
                    f"{side} {filled_qty} {symbol} @ ${filled_price:.2f} (Alpaca, "
                    f"slippage: {slippage:+.1f} bps)"
                )

                return {
                    "symbol": symbol,
                    "side": side,
                    "quantity": filled_qty,
                    "price": filled_price,
                    "expected_price": expected_price,
                    "slippage_bps": slippage,
                    "commission": commission,
                    "status": "FILLED",
                }

            # Order pending
            self._pending_orders[symbol] = order_id
            logger.info(f"{side} {quantity} {symbol} order pending (id: {order_id})")

            return {
                "symbol": symbol,
                "side": side,
                "quantity": quantity,
                "price": expected_price,
                "expected_price": expected_price,
                "slippage_bps": 0,
                "commission": 0,
                "status": "PENDING",
                "order_id": order_id,
            }

        except Exception as e:
            logger.error(f"Error executing {side} {quantity} {symbol}: {e}")
            return None

    async def _wait_for_fill(
        self, order_id: str, timeout: int = 30
    ) -> Optional[dict]:
        """Poll for order fill status."""
        start = time.time()
        while time.time() - start < timeout:
            await asyncio.sleep(0.5)
            data = await self._request("GET", f"/v2/orders/{order_id}")
            status = data.get("status", "")
            if status == "filled":
                return data
            if status in ("cancelled", "expired", "rejected", "canceled"):
                logger.error(f"Order {order_id} failed: {status}")
                return None
        logger.warning(f"Order {order_id} fill timeout after {timeout}s")
        return None

    # ------------------------------------------------------------------
    # Position management
    # ------------------------------------------------------------------

    async def get_position(self, symbol: str) -> float:
        """Get position quantity for a symbol."""
        try:
            alpaca_sym = self._to_alpaca_symbol(symbol)
            encoded = alpaca_sym.replace("/", "%2F")
            data = await self._request("GET", f"/v2/positions/{encoded}")
            if data and "qty" in data:
                return float(data["qty"])
            return 0.0
        except Exception:
            return 0.0

    async def get_all_positions(self) -> Dict[str, float]:
        """Get all positions as {symbol: quantity}."""
        try:
            data = await self._request("GET", "/v2/positions")
            if not isinstance(data, list):
                return {}
            result: Dict[str, float] = {}
            for pos in data:
                alpaca_sym = pos.get("symbol", "")
                qty = float(pos.get("qty", 0))
                if qty != 0:
                    apex_sym = self._from_alpaca_symbol(alpaca_sym)
                    result[apex_sym] = qty
            return result
        except Exception as e:
            logger.error(f"Error getting positions: {e}")
            return {}

    async def get_detailed_positions(self) -> Dict[str, Dict]:
        """Get positions with metadata."""
        try:
            data = await self._request("GET", "/v2/positions")
            if not isinstance(data, list):
                return {}
            result: Dict[str, Dict] = {}
            for pos in data:
                alpaca_sym = pos.get("symbol", "")
                qty = float(pos.get("qty", 0))
                avg_cost = float(pos.get("avg_entry_price", 0))
                current_price = float(pos.get("current_price", 0))
                unrealized_pl = float(pos.get("unrealized_pl", 0))
                if qty != 0:
                    apex_sym = self._from_alpaca_symbol(alpaca_sym)
                    result[apex_sym] = {
                        "qty": qty,
                        "avg_cost": avg_cost,
                        "current_price": current_price,
                        "unrealized_pl": unrealized_pl,
                    }
            return result
        except Exception as e:
            logger.error(f"Error getting detailed positions: {e}")
            return {}

    # ------------------------------------------------------------------
    # Account info
    # ------------------------------------------------------------------

    async def get_portfolio_value(self) -> float:
        """Get total portfolio value."""
        try:
            account = await self._request("GET", "/v2/account")
            equity = float(account.get("equity", 0))
            self._last_equity = equity  # Cache for sync dashboard export
            return equity
        except Exception:
            return getattr(self, '_last_equity', 0.0)

    async def get_account_cash(self) -> float:
        """Get available cash."""
        try:
            account = await self._request("GET", "/v2/account")
            return float(account.get("cash", 0))
        except Exception:
            return 0.0

    async def get_buying_power(self) -> float:
        """Get buying power."""
        try:
            account = await self._request("GET", "/v2/account")
            return float(account.get("buying_power", 0))
        except Exception:
            return 0.0

    # ------------------------------------------------------------------
    # Order management
    # ------------------------------------------------------------------

    def has_pending_order(self, symbol: str) -> bool:
        """Check for pending orders on a symbol."""
        normalized = normalize_symbol(symbol)
        return normalized in self._pending_orders

    def get_open_orders(self) -> list:
        """Get list of open order IDs."""
        return list(self._pending_orders.values())

    async def cancel_all_orders(self) -> None:
        """Cancel all open orders."""
        try:
            await self._request("DELETE", "/v2/orders")
            self._pending_orders.clear()
            logger.info("Cancelled all open Alpaca orders")
        except Exception as e:
            logger.error(f"Error cancelling orders: {e}")

    # ------------------------------------------------------------------
    # Execution metrics (same interface as IBKRConnector)
    # ------------------------------------------------------------------

    def record_execution_metrics(
        self,
        symbol: str,
        expected_price: float,
        fill_price: float,
        commission: float = 0.0,
    ) -> None:
        """Record execution metrics for performance analysis."""
        if expected_price > 0:
            slippage_bps = (
                (fill_price - expected_price) / expected_price
            ) * 10000
        else:
            slippage_bps = 0.0

        self.execution_metrics["total_trades"] += 1
        self.execution_metrics["total_slippage"] += abs(slippage_bps)
        self.execution_metrics["total_commission"] += commission

        self.execution_metrics["slippage_history"].append(
            {
                "timestamp": datetime.now().isoformat(),
                "symbol": symbol,
                "expected_price": expected_price,
                "fill_price": fill_price,
                "slippage_bps": slippage_bps,
                "commission": commission,
            }
        )

        if abs(slippage_bps) > 10:
            logger.warning(
                f"High slippage on {symbol}: {slippage_bps:+.1f} bps "
                f"(expected ${expected_price:.2f}, filled ${fill_price:.2f})"
            )

    def get_execution_summary(self) -> Dict:
        """Get summary of execution metrics."""
        metrics = self.execution_metrics
        n_trades = metrics["total_trades"]

        if n_trades == 0:
            return {
                "total_trades": 0,
                "avg_slippage_bps": 0.0,
                "total_commission": 0.0,
                "avg_commission": 0.0,
            }

        return {
            "total_trades": n_trades,
            "avg_slippage_bps": metrics["total_slippage"] / n_trades,
            "total_commission": metrics["total_commission"],
            "avg_commission": metrics["total_commission"] / n_trades,
            "recent_slippage": metrics["slippage_history"][-10:],
        }

    # ------------------------------------------------------------------
    # Contract compatibility stubs
    # ------------------------------------------------------------------

    async def get_contract(self, symbol: str) -> Optional[Any]:
        """Compatibility stub - Alpaca doesn't use contracts."""
        return {"symbol": self._to_alpaca_symbol(symbol)}
