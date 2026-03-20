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
import math
import time
from datetime import datetime
from typing import Dict, Optional, Callable, Any, List, Set

import httpx

from config import ApexConfig
from core.symbols import AssetClass, parse_symbol, normalize_symbol
from monitoring.prometheus_metrics import PrometheusMetrics

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
        self._price_cache_ts: Dict[str, float] = {}
        self._quote_task: Optional[asyncio.Task] = None
        self._reconnect_delay: float = 30.0

        # Pending orders tracking
        self._pending_orders: Dict[str, str] = {}  # symbol -> order_id

        # Reconnect health tracking
        self._portfolio_fail_count: int = 0  # Consecutive get_portfolio_value failures

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
        # Active Alpaca crypto symbols discovered at connect-time (e.g., BTC/USD).
        self._active_crypto_symbols: Set[str] = set()
        self._preferred_quote_currencies: tuple[str, ...] = ("USD", "USDT", "USDC")

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
        if mapped and (not self._active_crypto_symbols or mapped in self._active_crypto_symbols):
            return mapped

        if parsed.asset_class == AssetClass.CRYPTO:
            preferred = [
                f"{parsed.base}/{quote}"
                for quote in self._preferred_quote_currencies
            ]
            if self._active_crypto_symbols:
                for candidate in preferred:
                    if candidate in self._active_crypto_symbols:
                        return candidate
            return preferred[0]

        return parsed.base

    async def _refresh_active_crypto_symbols(self) -> None:
        """Discover currently active Alpaca crypto symbols (BTC/USD, ETH/USDT, ...)."""
        try:
            data = await self._request(
                "GET",
                "/v2/assets",
                params={"status": "active", "asset_class": "crypto"},
            )
        except Exception as exc:
            logger.debug("Failed to refresh Alpaca crypto symbols: %s", exc)
            return

        symbols: Set[str] = set()
        if isinstance(data, list):
            for row in data:
                if not isinstance(row, dict):
                    continue
                sym = str(row.get("symbol", "")).strip().upper()
                if not sym:
                    continue
                if "/" in sym:
                    symbols.add(sym)
                    continue
                # Normalize condensed symbol format (e.g. BTCUSD -> BTC/USD).
                for quote in self._preferred_quote_currencies:
                    if sym.endswith(quote) and len(sym) > len(quote):
                        base = sym[: -len(quote)]
                        if base:
                            symbols.add(f"{base}/{quote}")
                        break
        if symbols:
            self._active_crypto_symbols = symbols
            logger.info(
                "Alpaca active crypto symbols loaded: %d pairs",
                len(self._active_crypto_symbols),
            )

    def get_discovered_crypto_symbols(
        self,
        *,
        limit: int = 24,
        preferred_quotes: Optional[List[str]] = None,
    ) -> List[str]:
        """
        Return active Alpaca crypto pairs in APEX format (CRYPTO:BASE/QUOTE).
        """
        if not self._active_crypto_symbols:
            return []

        quotes = [
            q.strip().upper()
            for q in (preferred_quotes or getattr(ApexConfig, "ALPACA_DISCOVER_CRYPTO_PREFERRED_QUOTES", []))
            if str(q).strip()
        ]
        if not quotes:
            quotes = ["USD", "USDT", "USDC"]
        quote_rank = {q: idx for idx, q in enumerate(quotes)}

        ranked: List[tuple[int, str, str, str]] = []
        for raw in self._active_crypto_symbols:
            if "/" not in raw:
                continue
            base, quote = raw.split("/", 1)
            base = base.strip().upper()
            quote = quote.strip().upper()
            if not base or not quote or not base.isalnum() or not quote.isalnum():
                continue
            ranked.append((quote_rank.get(quote, len(quotes)), base, quote, f"CRYPTO:{base}/{quote}"))

        ranked.sort(key=lambda row: (row[0], row[1], row[2]))
        pairs = [row[3] for row in ranked]
        if limit > 0:
            pairs = pairs[:limit]
        return pairs

    @staticmethod
    def _fear_greed_score_mult(fg: float) -> float:
        """Contrarian F&G multiplier: 1.20 at extreme fear (fg=0), 0.80 at extreme greed (fg=100)."""
        return 1.20 - 0.40 * (fg / 100.0)

    async def scan_crypto_momentum_leaders(
        self,
        top_n: int = 8,
        min_volume_usd: float = 500_000.0,
        anchors: Optional[List[str]] = None,
        excluded: Optional[set] = None,
        fear_greed: Optional[float] = None,
    ) -> List[str]:
        """Rank all active Alpaca USD crypto pairs by 24-hour momentum × liquidity.

        Uses a single batched call to ``/v1beta3/crypto/us/snapshots`` per 50
        symbols — no per-symbol HTTP round-trips.

        Scoring: ``|Δ24h%| × log1p(vol_usd / 1_000_000)``
          - ``|Δ24h%|`` rewards fast-moving coins over static ones.
          - ``log1p`` ensures adequate liquidity without letting BTC's raw
            dollar volume overwhelm every other pair.

        Anchor symbols (BTC/USD, ETH/USD by default) are always included
        regardless of their score.  Symbols in ``excluded`` are silently
        skipped.

        Returns a list of ``CRYPTO:``-prefixed APEX symbols (e.g.
        ``["CRYPTO:BTC/USD", "CRYPTO:SOL/USD", ...]``).

        IBKR equity universe is completely untouched — this method only
        produces a list; the caller decides whether to update
        ``_dynamic_crypto_symbols``.
        """
        import math

        # ── 1. Ensure we have a fresh asset list ──────────────────────────
        if not self._active_crypto_symbols:
            await self._refresh_active_crypto_symbols()
        if not self._active_crypto_symbols:
            logger.warning("CryptoMomentumScan: no active crypto symbols known — aborting")
            return []

        _excluded: set = {s.upper() for s in (excluded or set())}
        _anchor_alpaca: List[str] = []
        for a in (anchors or ["BTC/USD", "ETH/USD"]):
            a_clean = a.upper().replace("CRYPTO:", "").strip()
            if a_clean not in _anchor_alpaca:
                _anchor_alpaca.append(a_clean)

        # ── 2. Build candidate list (USD-quoted only, not excluded) ───────
        candidates: List[str] = []
        for raw in sorted(self._active_crypto_symbols):
            if "/" not in raw:
                continue
            base, quote = raw.split("/", 1)
            base = base.strip().upper()
            quote = quote.strip().upper()
            if quote not in ("USD", "USDT", "USDC"):
                continue
            apex_key = f"CRYPTO:{base}/{quote}"
            if apex_key in _excluded or raw in _excluded:
                continue
            candidates.append(raw)  # Alpaca format, e.g. "SOL/USD"

        if not candidates:
            logger.warning("CryptoMomentumScan: candidate list empty after filtering")
            return []

        # ── 3. Batch snapshot requests (50 per call) ──────────────────────
        scores: dict = {}   # alpaca_sym -> float score
        BATCH = 50
        for i in range(0, len(candidates), BATCH):
            batch = candidates[i : i + BATCH]
            try:
                data = await self._request(
                    "GET",
                    "/v1beta3/crypto/us/snapshots",
                    base_url=self.DATA_BASE_URL,
                    params={"symbols": ",".join(batch)},
                )
            except Exception as exc:
                logger.warning("CryptoMomentumScan: snapshot request failed: %s", exc)
                continue

            snapshots = {}
            if isinstance(data, dict):
                # API returns {"snapshots": {...}} or the dict directly
                snapshots = data.get("snapshots", data)

            for sym, snap in snapshots.items():
                if not isinstance(snap, dict):
                    continue
                daily = snap.get("dailyBar") or {}
                o   = float(daily.get("o") or 0)
                c   = float(daily.get("c") or 0)
                v   = float(daily.get("v") or 0)   # base-currency volume
                vw  = float(daily.get("vw") or 0)  # volume-weighted avg price
                if o <= 0 or v <= 0:
                    continue
                price = vw if vw > 0 else c
                if price <= 0:
                    continue
                vol_usd = v * price
                if vol_usd < min_volume_usd:
                    continue
                delta_pct = abs(c - o) / o  # dimensionless; 0.05 = 5%
                fg_mult = self._fear_greed_score_mult(fear_greed) if fear_greed is not None else 1.0
                score = delta_pct * math.log1p(vol_usd / 1_000_000.0) * fg_mult
                scores[sym] = score

        if not scores:
            logger.warning(
                "CryptoMomentumScan: no qualifying pairs (min_vol_usd=%.0f) — "
                "returning anchors only", min_volume_usd
            )
            return [f"CRYPTO:{a}" for a in _anchor_alpaca if a in set(candidates)]

        # ── 4. Rank and select top_n ──────────────────────────────────────
        ranked = sorted(scores, key=lambda s: scores[s], reverse=True)
        selected_alpaca: List[str] = list(_anchor_alpaca)  # anchors first
        for sym in ranked:
            if len(selected_alpaca) >= top_n:
                break
            if sym not in selected_alpaca:
                selected_alpaca.append(sym)

        result = [f"CRYPTO:{s}" for s in selected_alpaca]
        fg_str = f" fg={fear_greed:.0f}" if fear_greed is not None else ""
        logger.info(
            "CryptoMomentumScan: top %d from %d candidates (min_vol=$%.0f%s): %s",
            len(result), len(scores), min_volume_usd, fg_str, ", ".join(result),
        )
        return result

    # Known crypto base assets for reverse mapping
    _CRYPTO_BASES = {
        "BTC", "ETH", "SOL", "ADA", "XRP", "DOT", "LTC", "BCH",
        "DOGE", "AVAX", "LINK", "MATIC", "XLM", "XMR", "ETC",
        "AAVE", "UNI", "SHIB", "ATOM", "FIL", "CRV", "BAT", "RENDER",
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

    @staticmethod
    def _safe_float(value: Any, default: float = 0.0) -> float:
        """Convert arbitrary API values to finite floats without raising."""
        try:
            parsed = float(value)
        except (TypeError, ValueError):
            return float(default)
        if not math.isfinite(parsed):
            return float(default)
        return float(parsed)

    async def _ensure_connected_for_account_reads(self) -> bool:
        """Reconnect lazily when account endpoints are queried before connect()."""
        if self.offline_mode:
            return False
        if self._client is not None and self._connected:
            return True
        try:
            await self.connect()
            return self._client is not None and self._connected
        except Exception as exc:
            logger.warning("Alpaca account read reconnect failed: %s", exc)
            return False

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
        if not self._client:
            # Client not yet initialized — connector hasn't connected yet.
            # Return None so callers like get_portfolio_value() can use their cache gracefully.
            logger.debug("Alpaca _request called before connect() — client not ready, returning None")
            return None
        # Allow initial account probe during connect() before _connected flips true.
        if not self._connected and path != "/v2/account":
            logger.debug("Alpaca _request called before connected flag set (path=%s) — returning None", path)
            return None


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
                    timeout=getattr(ApexConfig, "ALPACA_HTTP_TIMEOUT_SECONDS", 8.0),
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
            except (httpx.RequestError, asyncio.TimeoutError, RuntimeError) as e:
                if "client has been closed" in str(e).lower():
                    self._connected = False
                    self._client = None
                    logger.warning("Alpaca HTTP client was closed; marking connector disconnected")
                    return {}
                if "cannot send a request" in str(e).lower():
                    self._connected = False
                    self._client = None
                    logger.warning("Alpaca HTTP request rejected because client was closed; connector marked disconnected")
                    return {}
                if attempt < max_retries:
                    logger.warning(
                        f"Alpaca request failed (attempt {attempt + 1}): {type(e).__name__}: {e}"
                    )
                    await asyncio.sleep(2 ** attempt)
                else:
                    logger.error(
                        f"Alpaca request failed after {max_retries + 1} attempts: {type(e).__name__}: {e}"
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
            if self._client:
                await self._client.aclose()
                self._client = None
                
            self._client = httpx.AsyncClient(
                limits=httpx.Limits(max_keepalive_connections=20, max_connections=50),
                timeout=httpx.Timeout(
                    connect=10.0,   # TCP + TLS handshake
                    read=getattr(ApexConfig, "ALPACA_HTTP_TIMEOUT_SECONDS", 20.0),
                    write=10.0,
                    pool=30.0,      # max wait for a connection from pool
                ),
            )
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

            # Reconcile pending orders: sync local _pending_orders with
            # Alpaca's actual open orders to prevent ghost/duplicate orders.
            await self._reconcile_pending_orders()

            if account.get("crypto_status") == "ACTIVE":
                logger.info("  Crypto:  ACTIVE")
                await self._refresh_active_crypto_symbols()
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
                self._price_cache_ts[normalized] = time.time()
                if self.data_callback:
                    self.data_callback(normalized)
                return price

            logger.debug(f"No price for {symbol} from Alpaca (symbol may be delisted or market closed)")
            # Track dead symbols to avoid repeated warning spam
            if not hasattr(self, '_no_price_symbols'):
                self._no_price_symbols = set()
            if symbol not in self._no_price_symbols:
                self._no_price_symbols.add(symbol)
                logger.warning(f"No price for {symbol} from Alpaca (first miss — subsequent misses suppressed to debug)")
            return self._fallback_price(normalized)

        except Exception as e:
            msg = str(e).lower()
            if "client has been closed" in msg or "cannot send a request" in msg:
                logger.warning("Alpaca price request skipped for %s: connector client closed", symbol)
                return self._fallback_price(normalized)
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
        """Background polling loop for crypto quotes with stale-data reconnect watchdog."""
        chunk_size = 25
        cycle_count = 0
        # Stale-data watchdog: counts consecutive cycles where we received 0 fresh quotes
        _stale_cycles = 0
        _STALE_CYCLE_LIMIT = 3   # 3 × 10s = 30s of zero fresh data forces reconnect

        while True:
            try:
                cycle_count += 1
                if not symbols:
                    await asyncio.sleep(15)
                    continue
                if self.offline_mode:
                    await asyncio.sleep(15)
                    continue

                if not self._connected or self._client is None:
                    try:
                        logger.info("Alpaca quote loop disconnected. Attempting reconnect...")
                        await self.connect()
                        self._reconnect_delay = 30.0  # Reset on success
                        cycle_count = 0
                        _stale_cycles = 0
                    except Exception as exc:
                        logger.warning("Alpaca reconnect failed: %s — retry in %.0fs", exc, self._reconnect_delay)
                        await asyncio.sleep(self._reconnect_delay)
                        self._reconnect_delay = min(self._reconnect_delay * 2, 300.0)
                    continue

                # ─── Stale-data watchdog ──────────────────────────────────────
                # If we have symbols actively subscribed but ALL prices are older
                # than 90s, the HTTP client is silently broken (no exception raised,
                # but data has stopped flowing). Force a reconnect.
                if symbols and self._price_cache_ts:
                    now = time.time()
                    oldest_fresh = max(self._price_cache_ts.get(normalize_symbol(s), 0.0) for s in symbols)
                    if (now - oldest_fresh) > 90.0:
                        _stale_cycles += 1
                        if _stale_cycles >= _STALE_CYCLE_LIMIT:
                            logger.warning(
                                "Alpaca watchdog: prices stale for >90s (%d cycles) — forcing reconnect",
                                _stale_cycles,
                            )
                            self._connected = False
                            if self._client:
                                try:
                                    await self._client.aclose()
                                except Exception:
                                    pass
                            self._client = None
                            _stale_cycles = 0
                            continue
                    else:
                        _stale_cycles = 0  # Reset on fresh data

                # Periodically check account status (every 5 minutes)
                if cycle_count % 30 == 0:
                    account = await self._request("GET", "/v2/account")
                    if not account or "id" not in account:
                        logger.warning("Alpaca connection health check failed — marking disconnected")
                        self._connected = False
                        if self._client:
                            try:
                                await self._client.aclose()
                            except Exception:
                                pass
                        self._client = None
                        continue

                fresh_count = 0
                for i in range(0, len(symbols), chunk_size):
                    batch = symbols[i:i + chunk_size]
                    alpaca_to_normalized: Dict[str, str] = {}
                    for symbol in batch:
                        normalized = normalize_symbol(symbol)
                        self._price_cache.pop(normalized, None)
                        alpaca_to_normalized[self._to_alpaca_symbol(symbol)] = normalized

                    data = await self._request(
                        "GET",
                        "/v1beta3/crypto/us/latest/quotes",
                        base_url=self.DATA_BASE_URL,
                        params={"symbols": ",".join(alpaca_to_normalized.keys())},
                    )
                    quotes = data.get("quotes", {}) if isinstance(data, dict) else {}
                    for alpaca_sym, normalized in alpaca_to_normalized.items():
                        quote = quotes.get(alpaca_sym, {})
                        bp = float(quote.get("bp", 0.0) or 0.0)
                        ap = float(quote.get("ap", 0.0) or 0.0)
                        if bp > 0 and ap > 0:
                            self._price_cache[normalized] = (bp + ap) / 2.0
                            self._price_cache_ts[normalized] = time.time()
                            fresh_count += 1
                            if self.data_callback:
                                self.data_callback(normalized)
                        else:
                            # Fallback to per-symbol fetch to keep legacy behavior.
                            await self.get_market_price(normalized)

                if fresh_count == 0 and symbols:
                    _stale_cycles += 1
                    logger.debug("Alpaca poll: 0 fresh quotes in this cycle (stale=%d)", _stale_cycles)
                else:
                    _stale_cycles = 0

                await asyncio.sleep(10)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Quote polling error: {e}")
                await asyncio.sleep(30)

    def get_quote_age(self, symbol: str) -> float:
        """
        Returns the age of the cached quote in seconds. 
        Returns 999999.0 if no quote exists.
        """
        try:
            normalized = normalize_symbol(symbol)
            if normalized in self._price_cache_ts:
                return time.time() - self._price_cache_ts[normalized]
        except:
            pass
        return 999999.0

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

            # Guard: Alpaca connector only handles crypto; reject anything else at the gate.
            if parsed.asset_class != AssetClass.CRYPTO:
                logger.error(
                    "AlpacaConnector: non-crypto symbol %s (asset_class=%s) rejected — "
                    "route equities/forex to IBKR.",
                    symbol, parsed.asset_class.value,
                )
                return None

            if side not in ("BUY", "SELL"):
                logger.error(f"Invalid order side: {side}")
                return None
            if quantity <= 0:
                logger.error(f"Invalid quantity: {quantity}")
                return None

            expected_price = await self.get_market_price(symbol)
            if expected_price <= 0:
                logger.error(f"Cannot execute: no price for {symbol}")
                self.write_dead_letter(symbol, side, quantity, "no_price")
                return None

            # UPGRADE D: Pre-trade spread gate (entries only)
            if side == "BUY":
                allowed, spread_bps, spread_reason = await self.check_spread_gate(symbol)
                if not allowed:
                    logger.warning("🚫 %s: BUY blocked by spread gate (%s)", symbol, spread_reason)
                    self.write_dead_letter(symbol, side, quantity, f"spread_gate:{spread_reason}",
                                          {"spread_bps": spread_bps})
                    return None

            alpaca_sym = self._to_alpaca_symbol(symbol)

            # ✅ PRE-TRADE BUYING POWER GUARD
            # Alpaca crypto uses non-marginable buying power for fills.
            # If the notional exceeds available cash we'll get a 403 rejection.
            if side == "BUY":
                try:
                    account = await self._request("GET", "/v2/account")
                    # Crypto purchases are cash-funded only (no margin). Use cash balance
                    # directly. non_marginable_buying_power is unreliable in Alpaca paper
                    # trading (often reports ~$10 for crypto-heavy accounts regardless of
                    # actual cash balance).
                    cash_cap = float(account.get("cash", 0) or 0)
                    bp = float(account.get("buying_power", 0) or 0)
                    # buying_power = 2×cash for margin accounts; cap at cash to avoid over-spend
                    raw_avail = min(cash_cap, bp / 2.0) if min(cash_cap, bp / 2.0) > 0 else cash_cap if cash_cap > 0 else 0
                    
                    # Prevent concurrent overspend: track outgoing nominally over the last 10 seconds
                    now_ts = time.time()
                    if not hasattr(self, "_crypto_spend_blocks"):
                        self._crypto_spend_blocks = []
                    self._crypto_spend_blocks = [(ts, amt) for ts, amt in getattr(self, "_crypto_spend_blocks") if now_ts - ts < 10.0]
                    pending_spend = sum(amt for ts, amt in self._crypto_spend_blocks)
                    avail = max(0.0, raw_avail - pending_spend)

                    notional = quantity * expected_price
                    # Apply a 95% safety margin to avoid edge-case over-spend
                    max_notional = avail * 0.95
                    if notional > max_notional:
                        if max_notional <= 0:
                            logger.warning(
                                "🚫 %s: BUY blocked — Alpaca buying power is $0 (net of pending trades)",
                                symbol
                            )
                            self.write_dead_letter(symbol, side, quantity, "no_buying_power")
                            return None
                        # Scale down quantity to what we can afford
                        reduced_qty = max_notional / expected_price
                        reduced_qty = round(reduced_qty, 6)
                        
                        logger.warning(
                            "⚠️ %s: Reducing BUY qty %.4f→%.4f (notional $%.2f > available $%.2f @ 95%%)",
                            symbol, quantity, reduced_qty, notional, avail
                        )
                        quantity = reduced_qty
                        if quantity <= 0:
                            self.write_dead_letter(symbol, side, quantity, "no_buying_power")
                            return None
                    
                    # ✅ Alpaca minimum order notional check ($10)
                    ALPACA_MIN_NOTIONAL = 10.0
                    actual_notional = quantity * expected_price
                    if actual_notional < ALPACA_MIN_NOTIONAL:
                        logger.warning(
                            "⏭️ %s: Skipping — scaled notional $%.2f < $%.0f Alpaca minimum",
                            symbol, actual_notional, ALPACA_MIN_NOTIONAL
                        )
                        return None
                    
                    # Add to tracking blocks right before firing order
                    self._crypto_spend_blocks.append((time.time(), actual_notional))

                except Exception as bp_err:
                    logger.debug("Buying power check failed (non-fatal): %s", bp_err)


            # ── Exit limit order: try limit first, fall back to market ────────────
            # Data: P75 exit slippage = 29.2 bps. A sell-limit at -30 bps captures
            # ~75% of exits at reduced slippage; 25% fall back to market.
            _use_exit_limit = (
                side == "SELL"
                and getattr(ApexConfig, "ALPACA_EXIT_USE_LIMIT", True)
                and not force_market
            )
            _limit_offset_bps = float(getattr(ApexConfig, "ALPACA_EXIT_LIMIT_OFFSET_BPS", 30))
            _limit_price = round(expected_price * (1 - _limit_offset_bps / 10000), 6) if _use_exit_limit else None

            if _use_exit_limit and _limit_price:
                order_body = {
                    "symbol": alpaca_sym,
                    "qty": str(quantity),
                    "side": "sell",
                    "type": "limit",
                    "limit_price": str(_limit_price),
                    "time_in_force": "gtc",
                }
                logger.info(
                    "Placing SELL LIMIT %s %.6f %s @ $%.4f (limit, offset=%.0f bps, mid=$%.2f)",
                    symbol, quantity, alpaca_sym, _limit_price, _limit_offset_bps, expected_price,
                )
            else:
                order_body = {
                    "symbol": alpaca_sym,
                    "qty": str(quantity),
                    "side": side.lower(),
                    "type": "market",
                    "time_in_force": "gtc",
                }
                logger.info(
                    "Placing %s %s %.6f @ ~$%.2f (market)",
                    side, alpaca_sym, quantity, expected_price,
                )

            data = await self._request("POST", "/v2/orders", json_body=order_body)

            if not data or "id" not in data:
                logger.error(f"Order rejected by Alpaca: {data}")
                self.write_dead_letter(symbol, side, quantity, "alpaca_rejected",
                                      {"response": str(data)[:200]})
                return None

            order_id = data["id"]
            status = data.get("status", "")

            # Poll until filled; if limit times out, cancel and retry as market
            if status in ("new", "accepted", "pending_new"):
                _limit_wait = int(getattr(ApexConfig, "ALPACA_EXIT_LIMIT_WAIT_SECONDS", 8))
                _market_wait = int(getattr(ApexConfig, "ALPACA_FILL_WAIT_SECONDS", 10))
                _poll_timeout = _limit_wait if _use_exit_limit else _market_wait
                fill_data = await self._wait_for_fill(order_id, timeout=_poll_timeout)
                if fill_data:
                    data = fill_data
                elif _use_exit_limit and order_id:
                    # Limit missed — cancel and retry as market
                    try:
                        await self._request("DELETE", f"/v2/orders/{order_id}")
                        logger.info(
                            "AlpacaConnector: limit exit missed for %s after %ds — "
                            "retrying as market order",
                            symbol, _limit_wait,
                        )
                    except Exception as _ce:
                        logger.warning("AlpacaConnector: limit cancel failed: %s", _ce)
                    # Market fallback
                    market_body = {
                        "symbol": alpaca_sym,
                        "qty": str(quantity),
                        "side": "sell",
                        "type": "market",
                        "time_in_force": "gtc",
                    }
                    mkt_data = await self._request("POST", "/v2/orders", json_body=market_body)
                    if mkt_data and "id" in mkt_data:
                        mkt_fill = await self._wait_for_fill(mkt_data["id"], timeout=_market_wait)
                        data = mkt_fill or mkt_data
                    else:
                        logger.error("AlpacaConnector: market fallback order rejected: %s", mkt_data)
                elif order_id:
                    # Market order timed out — cancel dangling order
                    try:
                        await self._request("DELETE", f"/v2/orders/{order_id}")
                        logger.warning(
                            "AlpacaConnector: cancelled unfilled market order %s for %s after %ds",
                            order_id, symbol, _market_wait,
                        )
                        data = {
                            "id": order_id,
                            "status": "canceled",
                            "filled_qty": 0,
                            "filled_avg_price": 0,
                        }
                    except Exception as _ce:
                        logger.warning("AlpacaConnector: cancel order failed: %s", _ce)

            filled_qty = self._safe_float(data.get("filled_qty", 0))
            filled_price = self._safe_float(data.get("filled_avg_price", 0))
            status = str(data.get("status", status) or "").lower()

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

            if status in ("cancelled", "canceled", "expired", "rejected"):
                self._pending_orders.pop(symbol, None)
                logger.warning(
                    "AlpacaConnector: order %s for %s closed without a fill (status=%s)",
                    order_id,
                    symbol,
                    status,
                )
                return None

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
            self.write_dead_letter(symbol, side, quantity, f"exception:{type(e).__name__}:{e}")
            return None

    async def _wait_for_fill(
        self, order_id: str, timeout: int = 10
    ) -> Optional[dict]:
        """Poll for order fill status."""
        start = time.time()
        while time.time() - start < timeout:
            await asyncio.sleep(0.5)
            data = await self._request("GET", f"/v2/orders/{order_id}")
            if not isinstance(data, dict):
                continue
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
        """Get total portfolio value with reconnect circuit breaker."""
        reconnect_attempted = False
        for attempt in range(4): # 0-3, total 4 attempts
            try:
                if (self._client is None or not self._connected) and not reconnect_attempted:
                    reconnect_attempted = True
                    await self._ensure_connected_for_account_reads()
                account = await self._request("GET", "/v2/account")
                if isinstance(account, dict) and "equity" in account:
                    equity = self._safe_float(account.get("equity", 0))
                    if equity > 0:
                        self._last_equity = equity  # Cache for dashboard
                        self._portfolio_fail_count = 0  # Reset on success
                        return equity
                # If account is None or not a dict with 'equity', it's a soft failure, retry
                logger.error(f"Alpaca get_portfolio_value: _request returned invalid data (attempt {attempt+1}): {account}")
                if attempt == 0 and not reconnect_attempted:
                    reconnect_attempted = True
                    await self._ensure_connected_for_account_reads()
            except Exception as e:
                logger.error(f"Alpaca portfolio value fetch error (attempt {attempt+1}): {type(e).__name__}: {e}")
            
            if attempt < 3: # Don't sleep after the last attempt
                await asyncio.sleep(2.0)
        
        # If all retries fail or return invalid data, proceed to the original error handling
        # or return the cached value.
        # The original code had a broader try/except for the whole method.
        # We'll keep the circuit breaker logic for persistent failures.
        try:
            # This block is reached if the loop above didn't return a valid equity.
            # We'll treat this as a failure for the circuit breaker.
            raise RuntimeError("Failed to get valid portfolio value after multiple attempts")
        except Exception as e: # Catch the RuntimeError or any other exception that might have occurred
            self._portfolio_fail_count += 1
            cached = getattr(self, "_last_equity", 0.0)
            logger.warning(
                "Alpaca get_portfolio_value failed (%s: %s); returning cached $%.2f [fail#%d]",
                type(e).__name__, e, cached, self._portfolio_fail_count,
            )
            # After 3 consecutive failures, tear down the HTTP client so the
            # poll loop's reconnect logic will create a fresh one.
            if self._portfolio_fail_count >= 3:
                logger.warning(
                    "Alpaca: %d consecutive portfolio failures — forcing reconnect",
                    self._portfolio_fail_count,
                )
                self._connected = False
                if self._client:
                    try:
                        asyncio.get_event_loop().create_task(self._client.aclose())
                    except Exception:
                        pass
                self._client = None
                self._portfolio_fail_count = 0
            return cached

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
        """Get list of symbols with open (pending) orders."""
        return list(self._pending_orders.keys())

    async def cancel_all_orders(self) -> None:
        """Cancel all open orders."""
        try:
            await self._request("DELETE", "/v2/orders")
            self._pending_orders.clear()
            logger.info("Cancelled all open Alpaca orders")
        except Exception as e:
            logger.error(f"Error cancelling orders: {e}")

    # ------------------------------------------------------------------
    # UPGRADE B: Order persistence — save/load pending orders across restarts
    # ------------------------------------------------------------------

    def save_pending_orders(self, path) -> None:
        """Persist pending orders to disk so they survive engine restarts (Upgrade B)."""
        import json
        from pathlib import Path
        try:
            p = Path(path)
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(json.dumps(self._pending_orders, indent=2))
            logger.debug("Saved %d pending Alpaca orders to %s", len(self._pending_orders), p)
        except Exception as e:
            logger.warning("Failed to save pending orders: %s", e)

    def load_pending_orders(self, path) -> None:
        """Reload pending orders from disk on startup to avoid ghost orders (Upgrade B)."""
        import json
        from pathlib import Path
        try:
            p = Path(path)
            if not p.exists():
                return
            data = json.loads(p.read_text())
            if isinstance(data, dict):
                self._pending_orders.update(data)
                logger.info("Loaded %d pending Alpaca orders from %s", len(data), p)
        except Exception as e:
            logger.warning("Failed to load pending orders: %s", e)

    async def _reconcile_pending_orders(self) -> None:
        """Reconcile local _pending_orders with Alpaca's actual open orders.

        Removes entries from _pending_orders that no longer exist on Alpaca
        (filled, cancelled, or expired), preventing ghost orders and duplicate
        entries on reconnect.
        """
        if not self._pending_orders:
            return
        try:
            open_orders = await self._request("GET", "/v2/orders?status=open")
            if open_orders is None:
                logger.warning("Alpaca order reconciliation: could not fetch open orders")
                return
            # Build set of Alpaca open order IDs
            alpaca_open_ids = {o.get("id") for o in open_orders if isinstance(o, dict)}
            # Check each local pending order
            stale = []
            for sym, order_id in list(self._pending_orders.items()):
                if order_id not in alpaca_open_ids:
                    stale.append(sym)
            for sym in stale:
                self._pending_orders.pop(sym, None)
            if stale:
                logger.info(
                    "Alpaca order reconciliation: removed %d stale pending orders "
                    "(filled/cancelled): %s",
                    len(stale), stale,
                )
            else:
                logger.debug("Alpaca order reconciliation: all %d pending orders confirmed open",
                             len(self._pending_orders))
        except Exception as e:
            logger.warning("Alpaca order reconciliation failed: %s", e)

    # ------------------------------------------------------------------
    # UPGRADE D: Pre-trade spread gate — reject entry if spread too wide
    # ------------------------------------------------------------------

    async def check_spread_gate(self, symbol: str) -> tuple:
        """
        Check if bid-ask spread is within acceptable range before entry (Upgrade D).
        Returns (allowed: bool, spread_bps: float, reason: str).
        """
        if not getattr(ApexConfig, "LIQUIDITY_GATE_ENABLED", True):
            return True, 0.0, ""
        max_spread_bps = float(getattr(ApexConfig, "LIQUIDITY_SPREAD_MAX_BPS", 100.0))
        try:
            quote = await self.get_quote(symbol)
            if not quote:
                return True, 0.0, "no_quote"  # pass-through when quote unavailable
            bid, ask = quote.get("bid", 0.0), quote.get("ask", 0.0)
            if bid <= 0 or ask <= 0:
                return True, 0.0, "no_quote"
            mid = (bid + ask) / 2.0
            spread_bps = ((ask - bid) / mid) * 10_000 if mid > 0 else 0.0
            if spread_bps > max_spread_bps:
                return False, spread_bps, f"spread {spread_bps:.0f}bps > {max_spread_bps:.0f}bps limit"
            return True, spread_bps, ""
        except Exception as e:
            logger.debug("Spread gate check failed for %s: %s", symbol, e)
            return True, 0.0, "gate_error"

    # ------------------------------------------------------------------
    # UPGRADE I: Dead-letter queue — persist failed orders to JSONL
    # ------------------------------------------------------------------

    def write_dead_letter(self, symbol: str, side: str, quantity: float,
                          reason: str, context: dict = None) -> None:
        """Write a failed order to the dead-letter queue JSONL file (Upgrade I)."""
        if not getattr(ApexConfig, "DEAD_LETTER_QUEUE_ENABLED", True):
            return
        import json
        from pathlib import Path
        try:
            log_dir = Path(getattr(ApexConfig, "LOGS_DIR", "logs"))
            log_dir.mkdir(parents=True, exist_ok=True)
            entry = {
                "ts": datetime.utcnow().isoformat() + "Z",
                "broker": "alpaca",
                "symbol": symbol,
                "side": side,
                "quantity": quantity,
                "reason": reason,
                **(context or {}),
            }
            with open(log_dir / "dead_letter_orders.jsonl", "a") as f:
                f.write(json.dumps(entry) + "\n")
        except Exception as e:
            logger.debug("Dead-letter write failed: %s", e)

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

        # Record to Prometheus
        try:
            metrics = PrometheusMetrics()
            metrics.record_execution_slippage(abs(slippage_bps))
        except Exception:
            pass

        self.execution_metrics["slippage_history"].append(
            {
                "timestamp": datetime.utcnow().isoformat() + "Z",
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
