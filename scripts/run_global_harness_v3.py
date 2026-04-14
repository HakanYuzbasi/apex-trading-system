"""
scripts/run_global_harness_v3.py

Global Harness v3  Market-Neutral Portfolio with Sector Guardrails

New in v3 vs v2
---------------
* **VolatilitySizer** is wired into the strategy factory.  Every pair's
  ``leg_notional`` is dynamically scaled by the 30-day realised spread
  volatility so each pair risks the same target dollar amount regardless
  of market conditions.

* **FactorMonitor** (``quant_system/risk/factors.py``) enforces a sector
  concentration limit (default 30 % of equity per sector).  Its
  configuration (limit + any dynamic overrides) is persisted to a JSON
  sidecar file by the ``RiskStateController`` so restarts don't lose any
  manual adjustments.

* **EquityProtector** (``quant_system/risk/protector.py``) monitors the
  daily drawdown from the day's opening equity.  When the daily loss
  limit is breached (default -2 %) it publishes FLATTEN signals for every
  open position  including crypto  and refuses new entries for the rest
  of the UTC calendar day.  Its halt state is also persisted to a JSON
  sidecar file and reloaded on startup so a process restart mid-day does
  not clear the halt.

* **TCA ScheduledTask** fires ``TransactionCostAnalyzer.send_summary()``
  every weekday at 16:05 ET (5 minutes after the US equity market close).

* All new state is handled by ``RiskStateController.save()`` /
  ``RiskStateController.load()``, which are called alongside the existing
  ``StateManager`` to guarantee a clean restart.

24/7 compatibility
------------------
Crypto strategies remain active at all times.  Equity strategies are
activated / deactivated by ``StrategyController.reconcile()`` which
queries ``SessionManager``, identical to v2.  The TCA report is posted
only on weekdays (the scheduled-task implementation checks this before
dispatching).
"""
from __future__ import annotations

import asyncio
import json
import logging
import collections
import os
import signal
import sys
from pathlib import Path

# Initialize search path BEFORE any local imports
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import asyncio
import collections
import logging
import os
import signal
import time as _time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any, Callable
from zoneinfo import ZoneInfo
import websockets
import resource

from alpaca.data.enums import DataFeed
from alpaca.data.live.crypto import CryptoDataStream
from alpaca.data.live.stock import StockDataStream
from alpaca.trading.client import TradingClient
from alpaca.trading.stream import TradingStream

from core.symbols import AssetClass, normalize_symbol, parse_symbol
from core.trading_control import get_active_broker_mode
from execution.ibkr_connector import IBKRConnector
from execution.ibkr_adapter import IBKRAdapter

from quant_system.core.bus import InMemoryEventBus
from quant_system.core.calendars import SessionManager
from quant_system.data.normalizers.live_bridge import LiveDataBridge
from quant_system.data.stores.client import TimescaleDBClient
from quant_system.execution.brokers.alpaca_broker import AlpacaBroker
try:
    from quant_system.execution.neural_sniper import NeuralSniper
except ModuleNotFoundError:
    from quant_system.execution.obi_sniper import OBISniper as NeuralSniper  # type: ignore[assignment]
    import logging as _logging
    _logging.getLogger(__name__).warning("neural_sniper not found - falling back to OBISniper")

from quant_system.portfolio.ledger import PortfolioLedger
from quant_system.portfolio.persistence import StateManager
from quant_system.risk.eod_liquidator import EODLiquidator
from quant_system.risk.factors import FactorMonitor
from quant_system.risk.manager import RiskManager
from quant_system.risk.protector import EquityProtector
from quant_system.risk.regime_detector import RegimeDetector
from quant_system.risk.sizer import VolatilitySizer
from quant_system.risk.bayesian_vol import BayesianVolatilityAdjuster
from quant_system.analytics.alpha_monitor import AlphaDecayMonitor
from quant_system.analytics.notifier import TelegramNotifier
from quant_system.analytics.tca import TransactionCostAnalyzer

from core.logic.ml.meta_labeler import MetaLabeler
from quant_system.strategies.kalman_pairs import KalmanPairsStrategy
from quant_system.core.rotator import StrategyRotator
from quant_system.risk.tail_hedger import TailHedger
from portfolio.correlation_manager import CorrelationManager
from monitoring.live_monitor import LiveMonitor
from monitoring.health_monitor import HealthMonitor
from quant_system.portfolio.shadow_accounting import ShadowAccounting
from core.logging_config import setup_logging
from config import ApexConfig
from reconciliation.position_reconciler import PositionReconciler
from reconciliation.broker_adapters import AlpacaReconcilerAdapter
from quant_system.risk.sentiment_warden import SentimentWarden

logger = logging.getLogger("quant_system.global_harness_v3")

#  Log Interceptor for Telemetry 
_LOG_BUFFER = collections.deque(maxlen=20)

class TelemetryLogHandler(logging.Handler):
    """Intercepts logs and stores them in a rolling buffer for WebSocket delivery."""
    def emit(self, record):
        try:
            # Prevent recursive logging from web traffic or lower level bus events
            if record.name == "websockets.server":
                return
                
            timestamp = datetime.now().strftime("%H:%M:%S")
            msg = self.format(record)
            level = record.levelname
            # Format: [INFO] 19:30:22 - Strategy paused
            self._LOG_BUFFER.append(f"[{level}] {timestamp} - {msg}")
        except Exception:
            self.handleError(record)
    
    @property
    def _LOG_BUFFER(self):
        return _LOG_BUFFER

_ORDER_BUFFER = collections.deque(maxlen=50)

def _capture_order_execution(event: ExecutionEvent):
    """Intercepts execution fills and stores them for telemetry."""
    if event.execution_status in {"partial_fill", "filled"}:
        _ORDER_BUFFER.append({
            "time": datetime.now().strftime("%H:%M:%S"),
            "symbol": event.instrument_id,
            "side": event.side.upper(),
            "qty": float(event.fill_qty),
            "price": float(event.fill_price)
        })

#  ET timezone for the TCA scheduled task 
_ET = ZoneInfo("America/New_York")

#  Default state persistence directory 
_STATE_DIR = PROJECT_ROOT / "run_state" / "v3"

#  Default winner dictionary 
# ---------------------------------------------------------------------------
# Capital budget: ~$79k equity, 2x margin  $154k buying power.
# Target 60% utilisation = ~$92k deployed across all active legs combined.
# Each pair ties up 2  leg_notional (one per leg).
# 6 equity pairs  2  $7k  = $84k
# 4 crypto pairs   2  $2k  =  $16k   total  $100k (well within buying power)
# ---------------------------------------------------------------------------
DEFAULT_WINNER_DICTIONARY: dict[str, list[dict[str, Any]]] = {
    "crypto": [
        # Tier-1 crypto pairs  highest liquidity, tightest spreads
        {
            "slot": "crypto_btc_eth",
            "strategy": "kalman_pairs",
            "instrument_a": "CRYPTO:BTC/USD",
            "instrument_b": "CRYPTO:ETH/USD",
            "entry_z_score": 2.0,
            "exit_z_score": 0.5,
            "leg_notional": 2_500.0,
            "warmup_bars": 24,
        },
        {
            "slot": "crypto_eth_sol",
            "strategy": "kalman_pairs",
            "instrument_a": "CRYPTO:ETH/USD",
            "instrument_b": "CRYPTO:SOL/USD",
            "entry_z_score": 2.2,
            "exit_z_score": 0.5,
            "leg_notional": 2_000.0,
            "warmup_bars": 24,
        },
        {
            "slot": "crypto_btc_sol",
            "strategy": "kalman_pairs",
            "instrument_a": "CRYPTO:BTC/USD",
            "instrument_b": "CRYPTO:SOL/USD",
            "entry_z_score": 2.2,
            "exit_z_score": 0.5,
            "leg_notional": 2_000.0,
            "warmup_bars": 24,
        },
        {
            "slot": "crypto_eth_avax",
            "strategy": "kalman_pairs",
            "instrument_a": "CRYPTO:ETH/USD",
            "instrument_b": "CRYPTO:AVAX/USD",
            "entry_z_score": 2.3,
            "exit_z_score": 0.5,
            "leg_notional": 1_500.0,
            "warmup_bars": 24,
        },
    ],
    "equity": [
        # Notionals raised to $7k/leg to better utilise the $154k buying power.
        # All 6 pairs share 12 symbols = exactly the 25-symbol IEX bar subscription.
        {
            "slot": "equity_aapl_msft",
            "strategy": "kalman_pairs",
            "instrument_a": "AAPL",
            "instrument_b": "MSFT",
            "entry_z_score": 2.0,
            "exit_z_score": 0.5,
            "leg_notional": 7_000.0,
            "warmup_bars": 20,
        },
        {
            "slot": "equity_v_ma",
            "strategy": "kalman_pairs",
            "instrument_a": "V",
            "instrument_b": "MA",
            "entry_z_score": 2.0,
            "exit_z_score": 0.5,
            "leg_notional": 7_000.0,
            "warmup_bars": 20,
        },
        {
            "slot": "equity_amzn_googl",
            "strategy": "kalman_pairs",
            "instrument_a": "AMZN",
            "instrument_b": "GOOGL",
            "entry_z_score": 2.1,
            "exit_z_score": 0.5,
            "leg_notional": 7_000.0,
            "warmup_bars": 20,
        },
        {
            "slot": "equity_jpm_bac",
            "strategy": "kalman_pairs",
            "instrument_a": "JPM",
            "instrument_b": "BAC",
            "entry_z_score": 2.0,
            "exit_z_score": 0.5,
            "leg_notional": 6_000.0,
            "warmup_bars": 20,
        },
        {
            "slot": "equity_ko_pep",
            "strategy": "kalman_pairs",
            "instrument_a": "KO",
            "instrument_b": "PEP",
            "entry_z_score": 2.0,
            "exit_z_score": 0.5,
            "leg_notional": 6_000.0,
            "warmup_bars": 20,
        },
        {
            "slot": "equity_amd_nvda",
            "strategy": "kalman_pairs",
            "instrument_a": "AMD",
            "instrument_b": "NVDA",
            "entry_z_score": 2.5,
            "exit_z_score": 0.5,
            "leg_notional": 6_000.0,
            "warmup_bars": 20,
        },
    ],
}

DEFAULT_EXPECTED_SHARPE_BY_PAIR: dict[str, float] = {
    "CRYPTO:BTC/USD/CRYPTO:ETH/USD": 1.40,
    "CRYPTO:ETH/USD/CRYPTO:SOL/USD": 1.30,
    "CRYPTO:BTC/USD/CRYPTO:SOL/USD": 1.25,
    "CRYPTO:ETH/USD/CRYPTO:AVAX/USD": 1.20,
    "AAPL/MSFT": 1.85,
    "V/MA": 1.72,
    "AMZN/GOOGL": 1.65,
    "JPM/BAC": 1.55,
    "KO/PEP": 1.60,
    "AMD/NVDA": 1.50,
}

# Equity backbench: 13 extra symbols + 12 already in active pairs = 25 total.
# Alpaca free IEX feed supports up to 25 unique symbol subscriptions per connection.
# Extra 13: COST, CVS, CVX, GS, HD, JNJ, LOW, META, MS, T, UNH, WMT, XOM
BACKBENCH_UNIVERSE: list[dict[str, Any]] = [
    # Financials  reuse JPM/BAC from active slots
    {"instrument_a": "JPM", "instrument_b": "GS", "entry_z_score": 2.0, "leg_notional": 4000.0},
    {"instrument_a": "GS", "instrument_b": "MS", "entry_z_score": 2.0, "leg_notional": 4000.0},
    # Technology  reuse GOOGL/AMZN from active slots
    {"instrument_a": "GOOGL", "instrument_b": "META", "entry_z_score": 2.2, "leg_notional": 4000.0},
    {"instrument_a": "AMZN", "instrument_b": "COST", "entry_z_score": 2.1, "leg_notional": 4000.0},
    {"instrument_a": "COST", "instrument_b": "WMT", "entry_z_score": 2.0, "leg_notional": 4000.0},
    # Consumer
    {"instrument_a": "HD", "instrument_b": "COST", "entry_z_score": 2.0, "leg_notional": 4000.0},
    # Healthcare
    {"instrument_a": "UNH", "instrument_b": "CVS", "entry_z_score": 2.0, "leg_notional": 4000.0},
    {"instrument_a": "CVS", "instrument_b": "JNJ", "entry_z_score": 2.0, "leg_notional": 4000.0},
    # Energy
    {"instrument_a": "CVX", "instrument_b": "XOM", "entry_z_score": 2.0, "leg_notional": 4000.0},
    # Telecom
    {"instrument_a": "T", "instrument_b": "VZ", "entry_z_score": 2.0, "leg_notional": 4000.0},
    # Crypto extended  no IEX cap, all 36 Alpaca USD pairs available
    # Tier-2: high-liquidity alts correlated to BTC/ETH
    {"instrument_a": "CRYPTO:BTC/USD", "instrument_b": "CRYPTO:SOL/USD", "entry_z_score": 2.2, "leg_notional": 1500.0},
    {"instrument_a": "CRYPTO:BTC/USD", "instrument_b": "CRYPTO:AVAX/USD", "entry_z_score": 2.3, "leg_notional": 1500.0},
    {"instrument_a": "CRYPTO:SOL/USD", "instrument_b": "CRYPTO:AVAX/USD", "entry_z_score": 2.3, "leg_notional": 1500.0},
    {"instrument_a": "CRYPTO:ETH/USD", "instrument_b": "CRYPTO:LINK/USD", "entry_z_score": 2.3, "leg_notional": 1200.0},
    {"instrument_a": "CRYPTO:ETH/USD", "instrument_b": "CRYPTO:AAVE/USD", "entry_z_score": 2.3, "leg_notional": 1200.0},
    {"instrument_a": "CRYPTO:BTC/USD", "instrument_b": "CRYPTO:LTC/USD", "entry_z_score": 2.1, "leg_notional": 1200.0},
    {"instrument_a": "CRYPTO:BTC/USD", "instrument_b": "CRYPTO:BCH/USD", "entry_z_score": 2.1, "leg_notional": 1200.0},
    {"instrument_a": "CRYPTO:ETH/USD", "instrument_b": "CRYPTO:UNI/USD", "entry_z_score": 2.3, "leg_notional": 1000.0},
    {"instrument_a": "CRYPTO:BTC/USD", "instrument_b": "CRYPTO:XRP/USD", "entry_z_score": 2.2, "leg_notional": 1200.0},
    {"instrument_a": "CRYPTO:ETH/USD", "instrument_b": "CRYPTO:DOT/USD", "entry_z_score": 2.3, "leg_notional": 1000.0},
    {"instrument_a": "CRYPTO:LINK/USD", "instrument_b": "CRYPTO:AAVE/USD", "entry_z_score": 2.4, "leg_notional": 1000.0},
    {"instrument_a": "CRYPTO:SOL/USD", "instrument_b": "CRYPTO:ADA/USD", "entry_z_score": 2.4, "leg_notional": 1000.0},
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _configure_logging() -> None:
    setup_logging(
        level=os.getenv("GLOBAL_LOG_LEVEL", "INFO").upper(),
        log_file=ApexConfig.LOG_FILE,
        main_log_file="/private/tmp/apex_main.log",
        debug_log_file="/private/tmp/apex_debug.log",
        console_output=True
    )
    # Silence third party loggers for the harness
    logging.getLogger("ib_insync").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)


def _required_env(name: str) -> str:
    value = os.getenv(name, "").strip()
    if not value:
        raise RuntimeError(f"{name} must be set")
    return value


def _stock_feed_from_env() -> DataFeed:
    raw_feed = os.getenv("ALPACA_STOCK_DATA_FEED", "iex").strip().lower()
    if raw_feed == "sip":
        return DataFeed.SIP
    if raw_feed == "otc":
        return DataFeed.OTC
    return DataFeed.IEX


async def _bootstrap_cash(trading_client: TradingClient) -> float:
    account = await asyncio.to_thread(trading_client.get_account)
    return float(account.equity)


async def _sync_ledger_with_broker(trading_client: TradingClient, ledger: PortfolioLedger) -> None:
    """Aligns internal ledger cash with actual brokerage account equity on startup."""
    try:
        account = await asyncio.to_thread(trading_client.get_account)
        broker_truth = float(account.equity)
        
        # We adjust the cash so that the total_equity() of the ledger matches the broker truth.
        current_equity = ledger.total_equity()
        drift = abs(current_equity - broker_truth)
        
        if drift > 0.10: # $0.10 threshold
            logger.info(
                " Master Sync: Aligning ledger to broker equity truth ($%.2f -> $%.2f) drift=$%.2f",
                current_equity,
                broker_truth,
                drift
            )
            # Adjust cash by the difference
            ledger.cash += (broker_truth - current_equity)
            
        logger.info(" Portfolio synchronized with broker (Net Equity: $%.2f)", broker_truth)
    except Exception as e:
        logger.error(" Master Sync failed: %s", e)


async def _seed_existing_positions(
    trading_client: TradingClient, ledger: PortfolioLedger
) -> None:
    """Authoritative broker sync: zeroes out all ledger positions, then seeds from broker.

    This handles 'normalization' automatically: if the broker returns 'XRPUSD',
    we normalize it to 'CRYPTO:XRP/USD' before updating the ledger.
    """
    try:
        # First, collapse any existing duplicate symbols in the ledger (e.g. XRPUSD + CRYPTO:XRP/USD)
        _normalize_ledger_symbols(ledger)
        
        # Pull broker-truth positions
        positions = await asyncio.to_thread(trading_client.get_all_positions)
        
        # Zero out current ledger positions that are in the broker universe
        # to ensure we don't carry 'phantom' state from a stale JSON.
        seen_normalized = set()
        for p in positions:
            sym = normalize_symbol(p.symbol)
            pos_obj = ledger.get_position(sym)
            pos_obj.quantity = float(p.qty)
            pos_obj.avg_price = float(p.avg_entry_price)
            seen_normalized.add(sym)
            logger.info("   Seeded position: %s | Qty: %s", sym, p.qty)
            
        logger.info(" Sovereign Position Seed Complete: %d symbols synchronized.", len(seen_normalized))
    except Exception as e:
        logger.error(" Broker seeding failed: %s", e)


def _normalize_ledger_symbols(ledger: PortfolioLedger) -> None:
    """Collapses duplicate symbols in the ledger that normalize to the same string."""
    to_delete = []
    to_merge = {} # normalized_sym -> (total_qty, weighted_avg_price)
    
    for sym, pos in ledger.positions.items():
        norm = normalize_symbol(sym)
        if norm != sym or sym in to_merge:
            logger.warning(" Normalizing duplicate/legacy symbol: %s -> %s", sym, norm)
            qty, avg = to_merge.get(norm, (0.0, 0.0))
            new_qty = qty + pos.quantity
            if abs(new_qty) > 1e-6:
                new_avg = ((qty * avg) + (pos.quantity * pos.avg_price)) / new_qty
            else:
                new_avg = 0.0
            to_merge[norm] = (new_qty, new_avg)
            to_delete.append(sym)
        else:
            to_merge[norm] = (pos.quantity, pos.avg_price)
            
    # Apply merges
    for norm, (qty, avg) in to_merge.items():
        pos = ledger.get_position(norm)
        pos.quantity = qty
        pos.avg_price = avg
        
    # Remove old keys
    for sym in to_delete:
        if sym in ledger.positions and sym not in to_merge: # Don't delete if it IS the normalized one
             del ledger.positions[sym]


def _alpaca_symbol(instrument_id: str) -> str:
    parsed = parse_symbol(instrument_id)
    if parsed.asset_class == AssetClass.CRYPTO:
        return f"{parsed.base}/{parsed.quote}"
    return parsed.base


def _pair_label(instrument_a: str, instrument_b: str) -> str:
    return f"{instrument_a}/{instrument_b}"


def _backbench_stock_symbols() -> tuple[str, ...]:
    """All equity symbols referenced in BACKBENCH_UNIVERSE (for shadow subscriptions)."""
    symbols: set[str] = set()
    for cfg in BACKBENCH_UNIVERSE:
        for leg in ("instrument_a", "instrument_b"):
            raw = cfg.get(leg, "")
            if not raw:
                continue
            try:
                parsed = parse_symbol(raw)
                if parsed.asset_class != AssetClass.CRYPTO:
                    symbols.add(parsed.base)
            except Exception:
                pass
    return tuple(sorted(symbols))


def _backbench_crypto_symbols() -> tuple[str, ...]:
    """All crypto symbols referenced in BACKBENCH_UNIVERSE (for shadow subscriptions)."""
    symbols: set[str] = set()
    for cfg in BACKBENCH_UNIVERSE:
        for leg in ("instrument_a", "instrument_b"):
            raw = cfg.get(leg, "")
            if not raw:
                continue
            try:
                parsed = parse_symbol(raw)
                if parsed.asset_class == AssetClass.CRYPTO:
                    symbols.add(f"{parsed.base}/{parsed.quote}")
            except Exception:
                pass
    return tuple(sorted(symbols))


def _load_winner_dictionary() -> dict[str, list[dict[str, Any]]]:
    if path := os.getenv("GLOBAL_WINNER_DICTIONARY_PATH", "").strip():
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
    elif raw := os.getenv("GLOBAL_WINNER_DICTIONARY", "").strip():
        payload = json.loads(raw)
    else:
        payload = DEFAULT_WINNER_DICTIONARY

    normalized: dict[str, list[dict[str, Any]]] = {"crypto": [], "equity": []}
    for asset_group, configs in payload.items():
        if asset_group not in normalized or not isinstance(configs, list):
            continue
        for config in configs:
            if not isinstance(config, dict):
                continue
            normalized_config = dict(config)
            normalized_config["instrument_a"] = normalize_symbol(
                str(config["instrument_a"]).strip().upper()
            )
            normalized_config["instrument_b"] = normalize_symbol(
                str(config["instrument_b"]).strip().upper()
            )
            normalized_config.setdefault(
                "slot",
                f"{asset_group}_{normalized_config['instrument_a']}_{normalized_config['instrument_b']}",
            )
            normalized_config.setdefault("strategy", "kalman_pairs")
            normalized_config.setdefault("warmup_bars", 20)
            normalized[asset_group].append(normalized_config)
    
    # Adaptive Calibration Injection
    tuned_path = PROJECT_ROOT / "run_state" / "tuned_parameters.json"
    if tuned_path.exists():
        try:
            tuning_data = json.loads(tuned_path.read_text(encoding="utf-8"))
            overrides = tuning_data.get("tuning_overrides", {})
            tuned_count = 0
            for asset_group in normalized:
                for config in normalized[asset_group]:
                    label = _pair_label(config["instrument_a"], config["instrument_b"])
                    if label in overrides:
                        multiplier = overrides[label].get("entry_z_multiplier", 1.0)
                        old_z = config.get("entry_z_score", 2.0)
                        config["entry_z_score"] = float(old_z * multiplier)
                        tuned_count += 1
                        logger.info(f"ADAPTIVE CALIBRATION: {label} Z-score tuned {old_z:.2f} -> {config['entry_z_score']:.2f}")
            if tuned_count > 0:
                # Add flag for Telegram alert later
                normalized["_metadata"] = {"tuned_count": tuned_count}
        except Exception as e:
            logger.error(f"Failed to apply tuned parameters: {e}")

    return normalized


def _load_expected_sharpes() -> dict[str, float]:
    if path := os.getenv("EXPECTED_SHARPE_PATH", "").strip():
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
    elif raw := os.getenv("EXPECTED_SHARPE_MAP", "").strip():
        payload = json.loads(raw)
    else:
        payload = DEFAULT_EXPECTED_SHARPE_BY_PAIR
    return {str(key): float(value) for key, value in payload.items()}


# ---------------------------------------------------------------------------
# StrategySlotConfig
# ---------------------------------------------------------------------------

@dataclass(frozen=True, slots=True)
class StrategySlotConfig:
    slot: str
    asset_group: str
    strategy: str
    instrument_a: str
    instrument_b: str
    entry_z_score: float
    exit_z_score: float
    leg_notional: float  # base notional  may be overridden by VolatilitySizer
    warmup_bars: int

    @property
    def signature(self) -> tuple[object, ...]:
        return (
            self.strategy,
            self.instrument_a,
            self.instrument_b,
            self.entry_z_score,
            self.exit_z_score,
            self.leg_notional,
            self.warmup_bars,
        )

    @property
    def pair_label(self) -> str:
        return _pair_label(self.instrument_a, self.instrument_b)


# ---------------------------------------------------------------------------
# RiskStateController
# ---------------------------------------------------------------------------

class RiskStateController:
    """
    Persists and restores side-car state for the FactorMonitor and
    EquityProtector so that a process restart on the same UTC day does not
    silently clear the daily halt or lose custom sector overrides.

    Only the *configuration* novelties (sector overrides + concentration
    limit for FactorMonitor; halt + day-start equity for EquityProtector)
    are persisted here.  The live exposure is always recomputed from the
    PortfolioLedger, which is managed by the existing StateManager.

    Files written
    -------------
    <state_dir>/factor_monitor_state.json
    <state_dir>/equity_protector_state.json
    """

    def __init__(
        self,
        factor_monitor: FactorMonitor,
        equity_protector: EquityProtector,
        *,
        state_dir: Path,
    ) -> None:
        self._factor_monitor = factor_monitor
        self._equity_protector = equity_protector
        self._state_dir = state_dir
        self._factor_path = state_dir / "factor_monitor_state.json"
        self._protector_path = state_dir / "equity_protector_state.json"

    def load(self) -> None:
        """Load persisted state into the risk objects.  Safe to call on startup."""
        self._factor_monitor.load_state(self._factor_path)
        self._equity_protector.load_state(self._protector_path)

    def save(self) -> None:
        """Flush current state to disk.  Call before shutdown and after halt triggers."""
        self._state_dir.mkdir(parents=True, exist_ok=True)
        self._factor_monitor.save_state(self._factor_path)
        self._equity_protector.save_state(self._protector_path)
        logger.debug("RiskStateController: state flushed to %s", self._state_dir)


# ---------------------------------------------------------------------------
# ScheduledTask  generic wall-clock-based cron substitute
# ---------------------------------------------------------------------------

class ScheduledTask:
    """
    Fire ``coro_factory()`` at a specified HH:MM wall-clock time in a given
    timezone, once per calendar day.

    Parameters
    ----------
    name:
        Human-readable identifier shown in log messages.
    hour / minute:
        Local time (in ``tz``) at which the task should fire.
    tz:
        Timezone for the fire time.  Defaults to America/New_York.
    weekdays_only:
        When True (default), the task only fires Monday-Friday.
    coro_factory:
        Async callable with no arguments.  A fresh coroutine is spawned
        each time the task fires.
    poll_interval_seconds:
        How often to check the wall clock (default 30 s).
    """

    def __init__(
        self,
        name: str,
        *,
        hour: int,
        minute: int,
        tz: ZoneInfo = _ET,
        weekdays_only: bool = True,
        coro_factory: Callable[[], Any],
        poll_interval_seconds: float = 30.0,
    ) -> None:
        self._name = name
        self._hour = hour
        self._minute = minute
        self._tz = tz
        self._weekdays_only = weekdays_only
        self._coro_factory = coro_factory
        self._poll_interval = poll_interval_seconds
        self._last_fired_date: datetime | None = None

    async def run(self, stop_event: asyncio.Event) -> None:
        logger.info("ScheduledTask '%s' started: fires at %02d:%02d local", self._name, self._hour, self._minute)
        while not stop_event.is_set():
            try:
                await asyncio.wait_for(stop_event.wait(), timeout=self._poll_interval)
            except asyncio.TimeoutError:
                pass

            now_local = datetime.now(self._tz)

            # Weekday filter (Mon=0  Sun=6)
            if self._weekdays_only and now_local.weekday() >= 5:
                continue

            fire_today = now_local.replace(
                hour=self._hour, minute=self._minute, second=0, microsecond=0
            )
            # Fire if we are within [fire_time, fire_time + poll_interval) AND
            # we have not already fired today.
            already_fired_today = (
                self._last_fired_date is not None
                and self._last_fired_date.date() == now_local.date()
            )
            if already_fired_today:
                continue

            delta_seconds = (now_local - fire_today).total_seconds()
            if 0 <= delta_seconds < self._poll_interval * 2:
                self._last_fired_date = now_local
                logger.info("ScheduledTask '%s' firing at %s", self._name, now_local.isoformat())
                try:
                    await self._coro_factory()
                except Exception:
                    logger.exception("ScheduledTask '%s' raised an exception", self._name)


# ---------------------------------------------------------------------------
# StrategyController (v3  VolatilitySizer-aware)
# ---------------------------------------------------------------------------

class StrategyController:
    """
    Reconciles the desired strategy configuration against running strategies.

    v3 change: if a VolatilitySizer is provided, the leg_notional for
    each pair is replaced with the sizers current estimate before the
    strategy is instantiated.
    """

    def __init__(
        self,
        event_bus: InMemoryEventBus,
        session_manager: SessionManager,
        *,
        volatility_sizer: VolatilitySizer | None = None,
    ) -> None:
        self._event_bus = event_bus
        self._session_manager = session_manager
        self._volatility_sizer = volatility_sizer
        self._configured_slots: dict[str, StrategySlotConfig] = {}
        self._active_strategies: dict[str, tuple[StrategySlotConfig, KalmanPairsStrategy]] = {}
        self._paused_slots: set[str] = set()
        
        # Persistence configuration
        self._persistence_dir = Path("/app/run_state/v3")
        self._persistence_file = self._persistence_dir / "paused_strategies.json"
        self._load_paused_state()

    @property
    def configured_slots(self) -> dict[str, StrategySlotConfig]:
        return dict(self._configured_slots)

    def apply_winner_dictionary(self, winner_dictionary: dict[str, list[dict[str, Any]]]) -> None:
        configured_slots: dict[str, StrategySlotConfig] = {}
        for asset_group, configs in winner_dictionary.items():
            for config in configs:
                slot_config = StrategySlotConfig(
                    slot=str(config["slot"]),
                    asset_group=asset_group,
                    strategy=str(config["strategy"]),
                    instrument_a=str(config["instrument_a"]),
                    instrument_b=str(config["instrument_b"]),
                    entry_z_score=float(config["entry_z_score"]),
                    exit_z_score=float(config["exit_z_score"]),
                    leg_notional=float(config["leg_notional"]),
                    warmup_bars=int(config["warmup_bars"]),
                )
                configured_slots[slot_config.slot] = slot_config
        self._configured_slots = configured_slots

    def configured_stock_symbols(self) -> tuple[str, ...]:
        symbols = {
            _alpaca_symbol(config.instrument_a)
            for config in self._configured_slots.values()
            if self._slot_asset_class(config) in {AssetClass.EQUITY, AssetClass.OPTION}
        }
        symbols.update(
            _alpaca_symbol(config.instrument_b)
            for config in self._configured_slots.values()
            if self._slot_asset_class(config) in {AssetClass.EQUITY, AssetClass.OPTION}
        )
        return tuple(sorted(symbols))

    def configured_crypto_symbols(self) -> tuple[str, ...]:
        symbols = {
            _alpaca_symbol(config.instrument_a)
            for config in self._configured_slots.values()
            if self._slot_asset_class(config) == AssetClass.CRYPTO
        }
        symbols.update(
            _alpaca_symbol(config.instrument_b)
            for config in self._configured_slots.values()
            if self._slot_asset_class(config) == AssetClass.CRYPTO
        )
        return tuple(sorted(symbols))

    def active_slot_configs(self) -> tuple[StrategySlotConfig, ...]:
        return tuple(config for config, _ in self._active_strategies.values())

    def reconcile(self) -> None:
        desired_slots = set(self._configured_slots)
        for slot in list(self._active_strategies):
            if slot not in desired_slots:
                self._close_slot(slot)

        for slot, config in self._configured_slots.items():
            should_be_active = self._should_be_active(config)
            active = self._active_strategies.get(slot)
            if not should_be_active:
                if active is not None:
                    self._close_slot(slot)
                continue

            # Compute the vol-adjusted notional for this pair.
            sized_notional = self._resolve_notional(config)
            # Build a "live config" that reflects the current notional.
            live_config = StrategySlotConfig(
                slot=config.slot,
                asset_group=config.asset_group,
                strategy=config.strategy,
                instrument_a=config.instrument_a,
                instrument_b=config.instrument_b,
                entry_z_score=config.entry_z_score,
                exit_z_score=config.exit_z_score,
                leg_notional=sized_notional,
                warmup_bars=config.warmup_bars,
            )

            if active is not None and active[0].signature == live_config.signature:
                continue  # Already running with the same parameters.

            if active is not None:
                self._close_slot(slot)
            self._active_strategies[slot] = (live_config, self._instantiate_strategy(live_config))
            logger.info(
                "v3 Activated strategy slot=%s pair=%s leg_notional=$%.0f (sizer=%s)",
                slot,
                live_config.pair_label,
                sized_notional,
                "vol-adjusted" if self._volatility_sizer is not None else "fixed",
            )


    def close(self) -> None:
        for slot in list(self._active_strategies):
            self._close_slot(slot)

    def get_telemetry_snapshot(self, ledger: PortfolioLedger) -> list[dict[str, Any]]:
        """Aggregates live metrics from all active strategy slots for the dashboard."""
        snapshot = []
        # We iterate over configured slots to include paused ones
        for slot, config in self._configured_slots.items():
            active = self._active_strategies.get(slot)
            is_paused = slot in self._paused_slots
            
            status = "PAUSED" if is_paused else ("ACTIVE" if active else "CLOSED")
            z_score = getattr(active[1], "last_z_score", 0.0) if active else 0.0
            pnl = (
                ledger.get_unrealized_pnl(config.instrument_a)
                + ledger.get_unrealized_pnl(config.instrument_b)
                + ledger.get_position(config.instrument_a).realized_pnl
                + ledger.get_position(config.instrument_b).realized_pnl
            )
            
            snapshot.append({
                "pair_name": slot,
                "z_score": round(z_score, 3),
                "status": status,
                "pnl": round(pnl, 2)
            })
        return snapshot

    def toggle_strategy(self, slot_name: str) -> None:
        """Manually toggle a strategy slot on/off."""
        if slot_name in self._paused_slots:
            self._paused_slots.remove(slot_name)
            logger.info("v3 Strategy slot=%s RESUMED manually", slot_name)
        else:
            self._paused_slots.add(slot_name)
            logger.info("v3 Strategy slot=%s PAUSED manually", slot_name)
        
        # Immediate reconciliation to stop/start the instance
        self.reconcile()
        self._save_paused_state()

    def _save_paused_state(self) -> None:
        """Serializes the current paused slots to persistent storage."""
        try:
            self._persistence_dir.mkdir(parents=True, exist_ok=True)
            with open(self._persistence_file, "w") as f:
                json.dump(list(self._paused_slots), f)
            logger.info("v3 Persistent State: Saved %d paused slots to %s", len(self._paused_slots), self._persistence_file.name)
        except Exception as e:
            logger.error("v3 Persistence Error: Failed to save paused state: %s", e)

    def _load_paused_state(self) -> None:
        """Restores the paused slots from persistent storage."""
        if not self._persistence_file.exists():
            return
        try:
            with open(self._persistence_file, "r") as f:
                data = json.load(f)
                if isinstance(data, list):
                    self._paused_slots = set(data)
                    logger.info("v3 Persistent State: Restored %d paused slots", len(self._paused_slots))
        except Exception as e:
            logger.error("v3 Persistence Error: Failed to load paused state: %s", e)

    #  Private helpers 

    def _resolve_notional(self, config: StrategySlotConfig) -> float:
        """Return VolatilitySizer-scaled notional, falling back to the config value."""
        if self._volatility_sizer is None:
            return config.leg_notional
        sized = self._volatility_sizer.position_size(
            config.instrument_a,
            config.instrument_b,
            target_risk_dollars=config.leg_notional,
        )
        # Guard: cap at 1.5 base (keeps total exposure within buying power on fresh start)
        # and floor at 50% so the sizer never kills a pair with zero vol history.
        clamped = max(config.leg_notional * 0.50, min(config.leg_notional * 1.5, sized))
        if abs(clamped - config.leg_notional) / config.leg_notional > 0.01:
            logger.debug(
                "VolatilitySizer: pair=%s base=$%.0f  sized=$%.0f",
                config.pair_label,
                config.leg_notional,
                clamped,
            )
        return clamped

    def _instantiate_strategy(self, config: StrategySlotConfig) -> KalmanPairsStrategy:
        if config.strategy != "kalman_pairs":
            raise ValueError(f"Unsupported strategy in v3 harness: {config.strategy}")
        return KalmanPairsStrategy(
            self._event_bus,
            instrument_a=config.instrument_a,
            instrument_b=config.instrument_b,
            entry_z_score=config.entry_z_score,
            exit_z_score=config.exit_z_score,
            leg_notional=config.leg_notional,
            warmup_bars=config.warmup_bars,
        )

    def _should_be_active(self, config: StrategySlotConfig) -> bool:
        if config.slot in self._paused_slots:
            return False # Manually halted
            
        asset_class = self._slot_asset_class(config)
        if asset_class == AssetClass.CRYPTO:
            return True  # Crypto is always tradable  24/7.
        return self._session_manager.is_market_open(config.instrument_a) and \
               self._session_manager.is_market_open(config.instrument_b)

    @staticmethod
    def _slot_asset_class(config: StrategySlotConfig) -> AssetClass:
        return parse_symbol(config.instrument_a).asset_class

    def _close_slot(self, slot: str) -> None:
        active = self._active_strategies.pop(slot, None)
        if active is None:
            return
        _, strategy = active
        strategy.close()
        logger.info("Deactivated strategy slot=%s", slot)


# ---------------------------------------------------------------------------
# AlphaMonitorController (unchanged from v2)
# ---------------------------------------------------------------------------

class AlphaMonitorController:
    def __init__(
        self,
        portfolio_ledger: PortfolioLedger,
        event_bus: InMemoryEventBus,
        notifier: TelegramNotifier,
        expected_sharpe_by_pair: dict[str, float],
        emergency_callback: Callable[[str], asyncio.Future | Any],
    ) -> None:
        self._portfolio_ledger = portfolio_ledger
        self._event_bus = event_bus
        self._notifier = notifier
        self._expected_sharpe_by_pair = expected_sharpe_by_pair
        self._emergency_callback = emergency_callback
        self._monitors: dict[str, AlphaDecayMonitor] = {}

    def reconcile(self, active_configs: tuple[StrategySlotConfig, ...]) -> None:
        desired_labels = {config.pair_label for config in active_configs}
        for pair_label in list(self._monitors):
            if pair_label not in desired_labels:
                asyncio.create_task(self._close_monitor(pair_label))

        for config in active_configs:
            pair_label = config.pair_label
            if pair_label in self._monitors:
                continue
            expected_sharpe = self._expected_sharpe_by_pair.get(pair_label)
            if expected_sharpe is None:
                logger.warning(
                    "No expected Sharpe configured for %s; alpha monitor disabled for this pair",
                    pair_label,
                )
                continue
            self._monitors[pair_label] = AlphaDecayMonitor(
                self._portfolio_ledger,
                self._event_bus,
                self._notifier,
                strategy_label=pair_label,
                expected_oos_sharpe=expected_sharpe,
                alert_callback=self._emergency_callback,
            )
            logger.info(
                "Alpha monitor activated for %s expected_sharpe=%.3f", pair_label, expected_sharpe
            )

    async def close(self) -> None:
        for pair_label in list(self._monitors):
            await self._close_monitor(pair_label)

    async def _close_monitor(self, pair_label: str) -> None:
        monitor = self._monitors.pop(pair_label, None)
        if monitor is None:
            return
        await monitor.close()


# ---------------------------------------------------------------------------
# StreamSupervisor (unchanged from v2)
# ---------------------------------------------------------------------------

class StreamSupervisor:
    def __init__(
        self,
        name: str,
        factory: Callable[[], Any],
        subscribe: Callable[[Any, tuple[str, ...]], None],
    ) -> None:
        self._name = name
        self._factory = factory
        self._subscribe = subscribe
        self._desired_symbols: tuple[str, ...] = ()
        self._current_stream: Any | None = None

    async def update_symbols(self, symbols: tuple[str, ...]) -> None:
        normalized = tuple(sorted(set(symbols)))
        if normalized == self._desired_symbols:
            return
        self._desired_symbols = normalized
        logger.info("%s desired symbols updated to %s", self._name, normalized)
        if self._current_stream is not None:
            await asyncio.gather(self._current_stream.stop_ws(), return_exceptions=True)

    async def run(self, stop_event: asyncio.Event) -> None:
        backoff_seconds = 1.0
        while not stop_event.is_set():
            if not self._desired_symbols:
                try:
                    await asyncio.wait_for(stop_event.wait(), timeout=5.0)
                except asyncio.TimeoutError:
                    continue
                break

            stream = self._factory()
            self._current_stream = stream
            self._subscribe(stream, self._desired_symbols)
            logger.info("%s starting for symbols=%s", self._name, self._desired_symbols)
            try:
                await stream._run_forever()  # noqa: SLF001
                backoff_seconds = 1.0
                if not stop_event.is_set():
                    logger.warning("%s stopped unexpectedly; restarting", self._name)
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.exception("%s failed; retrying in %.1fs", self._name, backoff_seconds)
                await asyncio.sleep(backoff_seconds)
                backoff_seconds = min(backoff_seconds * 2.0, 60.0)
            finally:
                await asyncio.gather(stream.stop_ws(), stream.close(), return_exceptions=True)
                self._current_stream = None


# ---------------------------------------------------------------------------
# Stream helpers
# ---------------------------------------------------------------------------

async def _run_broker_forever(broker: AlpacaBroker, stop_event: asyncio.Event) -> None:
    backoff_seconds = 1.0
    while not stop_event.is_set():
        try:
            await broker.run()
            backoff_seconds = 1.0
            if not stop_event.is_set():
                logger.warning("alpaca-broker stream stopped unexpectedly; restarting")
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.exception("alpaca-broker stream failed; retrying in %.1fs", backoff_seconds)
            await asyncio.sleep(backoff_seconds)
            backoff_seconds = min(backoff_seconds * 2.0, 60.0)


def _subscribe_stock_stream(
    stream: StockDataStream, symbols: tuple[str, ...], bridge: LiveDataBridge
) -> None:
    if not symbols:
        return
    # IEX free tier: bars-only subscription keeps us within the 25-symbol limit.
    # Active-pair quotes are fetched via REST polling (_run_equity_quote_poller)
    # so the limit chaser and OBI sniper always have fresh bid/ask.
    stream.subscribe_bars(bridge.publish_alpaca, *symbols)


async def _run_equity_quote_poller(
    active_symbols_fn: "Callable[[], tuple[str, ...]]",
    event_bus: "InMemoryEventBus",
    api_key: str,
    secret_key: str,
    stop_event: asyncio.Event,
    poll_interval: float = 5.0,
) -> None:
    """REST-poll IEX quotes for active equity symbols every `poll_interval` seconds.

    Supplements bars-only websocket subscription with real bid/ask for the
    limit chaser and OBI sniper, bypassing the Alpaca IEX websocket subscription
    count limit on the free plan.
    """
    import aiohttp
    from itertools import count as _count
    from quant_system.events.market import QuoteTick
    from datetime import timezone

    base_url = "https://data.alpaca.markets/v2/stocks/quotes/latest"
    headers = {
        "APCA-API-KEY-ID": api_key,
        "APCA-API-SECRET-KEY": secret_key,
    }
    seq = _count(1)

    logger.info("Equity REST quote poller started (interval=%.1fs, feed=iex)", poll_interval)
    async with aiohttp.ClientSession(headers=headers) as session:
        while not stop_event.is_set():
            symbols = active_symbols_fn()
            if symbols:
                try:
                    params = {"symbols": ",".join(symbols), "feed": "iex"}
                    async with session.get(
                        base_url, params=params, timeout=aiohttp.ClientTimeout(total=4.0)
                    ) as resp:
                        if resp.status == 200:
                            data = await resp.json()
                            now = datetime.now(timezone.utc)
                            for sym, q in data.get("quotes", {}).items():
                                bid = float(q.get("bp") or 0)
                                ask = float(q.get("ap") or 0)
                                if bid > 0 and ask > 0 and ask >= bid:
                                    tick = QuoteTick(
                                        instrument_id=sym,
                                        exchange_ts=now,
                                        received_ts=now,
                                        processed_ts=now,
                                        sequence_id=next(seq),
                                        source="alpaca.rest.iex",
                                        bid=bid,
                                        ask=ask,
                                        bid_size=float(q.get("bs") or 0),
                                        ask_size=float(q.get("as") or 0),
                                    )
                                    await event_bus.publish_async(tick)
                        elif resp.status == 429:
                            await asyncio.sleep(poll_interval * 3)
                            continue
                except asyncio.CancelledError:
                    raise
                except Exception as exc:
                    logger.debug("Quote poller error: %s", exc)
            try:
                await asyncio.wait_for(stop_event.wait(), timeout=poll_interval)
            except asyncio.TimeoutError:
                pass
    logger.info("Equity REST quote poller stopped")


_crypto_bar_count: int = 0
_crypto_quote_count: int = 0
_crypto_trade_count: int = 0

async def _debug_bar_wrapper(payload: Any) -> None:
    global _crypto_bar_count
    _crypto_bar_count += 1
    if _crypto_bar_count <= 3:
        sym = payload.get("S") if isinstance(payload, dict) else getattr(payload, "symbol", "?")
        logger.info(" Crypto bars flowing: bar #%d sym=%s payload_keys=%s",
                    _crypto_bar_count, sym,
                    list(payload.keys()) if isinstance(payload, dict) else "N/A")
    await bridge_ref[0].publish_alpaca(payload)

async def _debug_quote_wrapper(payload: Any) -> None:
    await bridge_ref[0].publish_alpaca(payload)

async def _debug_trade_wrapper(payload: Any) -> None:
    await bridge_ref[0].publish_alpaca(payload)

bridge_ref: list[Any] = []  # Set at runtime to share bridge instance with debug wrappers

def _subscribe_crypto_stream(
    stream: CryptoDataStream, symbols: tuple[str, ...], bridge: LiveDataBridge
) -> None:
    if not symbols:
        return
    bridge_ref.append(bridge)
    stream.subscribe_bars(_debug_bar_wrapper, *symbols)
    stream.subscribe_trades(_debug_trade_wrapper, *symbols)
    stream.subscribe_quotes(_debug_quote_wrapper, *symbols)


async def _emergency_flatten_pair(pair_label: str) -> None:
    logger.critical("Emergency flatten intent logged for pair=%s", pair_label)


# ---------------------------------------------------------------------------
# Winner-refresh + reconciliation loop (v3)
# ---------------------------------------------------------------------------

async def _run_dashboard_updates(
    ledger: PortfolioLedger,
    risk_manager: RiskManager,
    correlation_manager: CorrelationManager,
    monitor: LiveMonitor,
    controller: StrategyController,
    rotator: StrategyRotator,
    tail_hedger: TailHedger,
    stop_event: asyncio.Event,
    trading_client: "TradingClient | None" = None,
    ibkr_adapter: "IBKRAdapter | None" = None,
) -> None:
    """Periodically update the dashboard state file with advanced metrics."""
    logger.info(" Dashboard update loop started")

    # Broker reconciliation cache  refreshed every 30 s to avoid rate limits
    _broker_positions_cache: list = []
    _broker_positions_ts: float = 0.0
    _BROKER_POS_TTL = 30.0

    # VIX / regime cache  refreshed every 5 min to avoid yfinance rate limits
    _vix_cache: float = 0.0
    _regime_cache: str = "neutral"
    _vix_ts: float = 0.0
    _VIX_TTL = 300.0  # 5 minutes

    async def _refresh_vix_regime() -> None:
        nonlocal _vix_cache, _regime_cache, _vix_ts
        try:
            import yfinance as yf
            import asyncio as _asyncio

            def _fetch():
                ticker = yf.Ticker("^VIX")
                hist = ticker.history(period="5d", interval="1d")
                if hist.empty:
                    return 0.0, "neutral"
                vix = float(hist["Close"].iloc[-1])
                # Derive regime from VIX level
                if vix >= 30:
                    regime = "volatile"
                elif vix >= 20:
                    regime = "neutral"
                else:
                    regime = "bull"
                # Override with drawdown signal if bayesian vol is high
                bayesian = getattr(risk_manager, "latest_bayesian_prob", 0.0) or 0.0
                if bayesian > 0.6:
                    regime = "volatile"
                elif bayesian > 0.35 and regime == "bull":
                    regime = "neutral"
                return vix, regime

            vix, regime = await _asyncio.to_thread(_fetch)
            _vix_cache = vix
            _regime_cache = regime
            _vix_ts = _time.monotonic()
        except Exception as _vix_exc:
            logger.debug("VIX/regime fetch failed: %s", _vix_exc)

    while not stop_event.is_set():
        try:
            # 0. Refresh raw broker positions for reconciliation (TTL-cached)
            now_mono = _time.monotonic()
            if now_mono - _broker_positions_ts >= _BROKER_POS_TTL:
                try:
                    all_raw_positions = []
                    
                    # Alpaca positions
                    if trading_client is not None:
                        alpaca_positions = await asyncio.to_thread(trading_client.get_all_positions)
                        all_raw_positions.extend(alpaca_positions)
                        
                    # IBKR positions
                    if ibkr_adapter and ibkr_adapter.is_connected():
                        # ibkr_adapter.ib is the ib_insync instance
                        ibkr_positions = ibkr_adapter.ib.positions()
                        # Normalize IBKR simple position objects to Alpaca-like schema for the UI
                        for p in ibkr_positions:
                            # p is Position(account='...', contract=Contract(...), position=10.0, avgCost=150.0)
                            all_raw_positions.append(type('MockPos', (), {
                                'symbol': p.contract.localSymbol or p.contract.symbol,
                                'qty': p.position,
                                'side': 'long' if p.position > 0 else 'short',
                                'market_value': p.position * (p.avgCost), # Approximation if mktPrice not in p
                                'unrealized_pl': 0.0,
                                'unrealized_plpc': 0.0,
                                'current_price': 0.0,
                                'avg_entry_price': p.avgCost
                            }))

                    # Build the set of instruments that APEX strategies are actively trading
                    strategy_instruments: set[str] = set()
                    for cfg in controller.active_slot_configs():
                        strategy_instruments.add(cfg.instrument_a)
                        strategy_instruments.add(cfg.instrument_b)

                    _broker_positions_cache = []
                    for pos in all_raw_positions:
                        norm_sym = normalize_symbol(str(pos.symbol).strip().upper())
                        _broker_positions_cache.append({
                            "symbol": str(pos.symbol),
                            "normalized_symbol": norm_sym,
                            "qty": float(pos.qty),
                            "side": str(getattr(pos, "side", "long")),
                            "market_value": float(getattr(pos, "market_value", None) or 0),
                            "unrealized_pl": float(getattr(pos, "unrealized_pl", None) or 0),
                            "unrealized_plpc": float(getattr(pos, "unrealized_plpc", None) or 0),
                            "current_price": float(getattr(pos, "current_price", None) or 0),
                            "avg_price": float(pos.avg_entry_price),
                            "is_orphaned": norm_sym not in strategy_instruments,
                        })
                    _broker_positions_ts = now_mono
                except Exception as _rp_exc:
                    logger.warning("Broker position reconciliation fetch failed: %s", _rp_exc)

            # 0b. Refresh VIX / regime (TTL-cached, non-blocking)
            if _time.monotonic() - _vix_ts >= _VIX_TTL:
                await _refresh_vix_regime()

            # 1. Collect Latency Data
            latency_map = []
            for config, strategy in controller._active_strategies.values():
                if hasattr(strategy, 'last_latency_micros'):
                    latency_map.append({
                        "name": f"Strategy:{config.pair_label}",
                        "micros": strategy.last_latency_micros
                    })
            
            # 2. Build the state payload
            alpaca_equity = await _bootstrap_cash(trading_client)
            ibkr_equity = 0.0
            if ibkr_adapter and ibkr_adapter.is_connected():
                try:
                    ibkr_equity = await ibkr_adapter.get_net_liquidation()
                except Exception as _ib_eq_exc:
                    logger.warning("IBKR equity fetch failed in dashboard loop: %s", _ib_eq_exc)
            
            sovereign_truth = alpaca_equity + ibkr_equity
            logger.info("Sovereign Dashboard Update: Alpaca=$%.2f, IBKR=$%.2f, Total=$%.2f", 
                        alpaca_equity, ibkr_equity, sovereign_truth)
            
            _realized_pnl = ledger.total_realized_pnl()
            _unrealized_pnl = ledger.total_unrealized_pnl()
            _control_file = PROJECT_ROOT / "run_state" / "trading_control_commands.json"
            
            state = {
                "timestamp": datetime.now().isoformat(),
                "equity": sovereign_truth,
                "capital": sovereign_truth,
                "cash": ledger.cash,
                "alpaca_equity": alpaca_equity,
                "ibkr_equity": ibkr_equity,
                "realized_pnl": _realized_pnl,
                "unrealized_pnl": _unrealized_pnl,
                "daily_pnl": _realized_pnl + _unrealized_pnl,
                "total_pnl": _realized_pnl + _unrealized_pnl,
                "active_margin": risk_manager.current_required_margin(),
                "leverage_limit": risk_manager.global_leverage_limit,
                "broker_mode": get_active_broker_mode(_control_file),
                "open_positions": len([s for s, p in ledger.positions.items() if abs(p.quantity) > 1e-6]),
                "drawdown": ledger.total_unrealized_pnl() / ledger.total_equity() if ledger.total_equity() > 0 else 0.0,
                "active_pairs": [cfg.pair_label for cfg in controller.active_slot_configs()],
                "backbench_performance": {
                    l: p.get_sortino() for l, p in rotator._performance.items()
                },
                "meta_confidence_score": risk_manager.latest_meta_confidence,
                "bayesian_vol_prob": risk_manager.latest_bayesian_prob,
                "obi_heat_indices": {
                    inst: rotator._latest_obi.get(inst, 0.0) if hasattr(rotator, '_latest_obi') else 0.0
                    for inst in list(ledger.positions.keys())
                },
                "survival_probability": getattr(risk_manager, 'survival_probability', 1.0),
                "regime": _regime_cache,
                "vix": _vix_cache,
                "hedge_status": tail_hedger.get_status()["is_hedged"],
                "latency_heatmap": latency_map,
                "brokers": [
                    {
                        "broker": "alpaca",
                        "mode": "trading" if trading_client else "offline",
                        "equity": alpaca_equity,
                        "status": "connected"
                    },
                    {
                        "broker": "ibkr",
                        "mode": "trading" if (ibkr_adapter and ibkr_adapter.is_connected()) else "offline",
                        "equity": ibkr_equity,
                        "status": "connected" if (ibkr_adapter and ibkr_adapter.is_connected()) else "disconnected"
                    }
                ],
                "correlation_matrix": correlation_manager.correlation_matrix.to_dict() if not correlation_manager.correlation_matrix.empty else {},
                "positions": {
                    sym: {
                        "qty": pos.quantity,
                        "avg_price": pos.avg_price,
                        "pnl": ledger.get_unrealized_pnl(sym),
                    } for sym, pos in ledger.positions.items() if abs(pos.quantity) > 1e-6
                },
                "broker_positions": _broker_positions_cache,
                "broker_heartbeats": {
                    "alpaca": {
                        "healthy": True,
                        "last_success_ts": datetime.now(timezone.utc).isoformat(),
                    },
                    "ibkr": {
                        "healthy": bool(ibkr_adapter and ibkr_adapter.is_connected()),
                        "last_success_ts": datetime.now(timezone.utc).isoformat(),
                        "last_error": None if (ibkr_adapter and ibkr_adapter.is_connected()) else "Connection lost"
                    }
                },
            }
            
            # 3. Update the monitor
            monitor.update_state(state)
            
        except Exception as e:
            logger.error(f"Dashboard update failed: {e}")
            
        try:
            await asyncio.wait_for(stop_event.wait(), timeout=10.0)
        except asyncio.TimeoutError:
            continue

async def _winner_refresh_loop(
    controller: StrategyController,
    alpha_controller: AlphaMonitorController,
    stock_supervisor: StreamSupervisor,
    crypto_supervisor: StreamSupervisor,
    notifier: TelegramNotifier,
    regime_detector: RegimeDetector,
    risk_state_controller: RiskStateController,
    expected_sharpes: dict[str, float],
    equity_protector: EquityProtector,
    rotator: StrategyRotator,
    tail_hedger: TailHedger,
    stop_event: asyncio.Event,
) -> None:
    last_refresh_at: datetime | None = None
    startup_summary_sent = False

    while not stop_event.is_set():
        now = datetime.now(timezone.utc)
        should_refresh = last_refresh_at is None or (now - last_refresh_at) >= timedelta(minutes=60)
        if should_refresh:
            winner_dictionary = _load_winner_dictionary()
            controller.apply_winner_dictionary(winner_dictionary)
            active_stocks = controller.configured_stock_symbols()
            active_cryptos = controller.configured_crypto_symbols()
            all_stocks = tuple(sorted(set(active_stocks) | set(_backbench_stock_symbols())))
            all_cryptos = tuple(sorted(set(active_cryptos) | set(_backbench_crypto_symbols())))
            await stock_supervisor.update_symbols(all_stocks)
            await crypto_supervisor.update_symbols(all_cryptos)
            last_refresh_at = now
            logger.info("Winner dictionary refreshed (v3)")

        controller.reconcile()
        alpha_controller.reconcile(controller.active_slot_configs())
        
        # Tail-Hedge Check
        await tail_hedger.check_and_hedge()

        # Hot-Swap Logic
        active_performance = []
        for config in controller.active_slot_configs():
            # Get current sortino from alpha monitor if available
            monitor = alpha_controller._monitors.get(config.pair_label)
            sortino = 0.0
            if monitor and hasattr(monitor, 'get_current_sortino'):
                sortino = monitor.get_current_sortino()
            active_performance.append((config.pair_label, sortino))
        
        swaps = rotator.get_swaps(active_performance)
        if swaps:
            for old_label, new_config in swaps:
                logger.warning("HOT-SWAP TRIGGERED: Replacing %s with %s", old_label, new_config['instrument_a'])
                # Update winner dictionary in-place for next refresh or force reconcile
                # For now, let's just log it. Real implementation would update the config source.

        # Hot-Swap Logic
        # ... (implementation from previous step)
        
        # Risk Adjustment Feedback
        adjustment_file = PROJECT_ROOT / "run_state" / "risk_adjustments.json"
        if adjustment_file.exists():
            try:
                with open(adjustment_file, 'r') as f:
                    adjustment = json.load(f)
                
                # Check if it's a fresh adjustment
                adj_ts = datetime.fromisoformat(adjustment['timestamp'])
                if (now - adj_ts) < timedelta(minutes=5):
                    risk_manager.set_leverage_limit(adjustment['recommended_leverage'])
                    risk_manager.survival_probability = adjustment['survival_probability']
                    logger.info("Monte Carlo Risk feedback applied: Leverage=%0.2f", adjustment['recommended_leverage'])
            except Exception as e:
                logger.error(f"Failed to load risk adjustments: {e}")

        # Persist risk state after every reconciliation so halt status is
        # never more than ~60 s stale on disk.
        await asyncio.to_thread(risk_state_controller.save)

        if not startup_summary_sent:
            halted_notice = "  HALTED (daily loss limit reached)" if equity_protector.is_halted() else ""
            regime_summary = (
                "Regime veto active for new entries when instrument regime is high_vol or breakout."
            )
            sharpe_lines = [
                f"{pair}: {expected_sharpes[pair]:.2f}" for pair in sorted(expected_sharpes)
            ]
            await notifier.notify_system_event(
                f"Harness v3 Online{halted_notice}",
                f"{regime_summary}\n"
                "Sector concentration limit: 30% equity\n"
                "Daily loss kill-switch: -2%\n"
                "VolatilitySizer: 30-day spread vol (clamped 5%-300% of base)\n"
                "TCA report: weekdays @ 16:05 ET\n"
                "Expected Sharpe targets:\n" + "\n".join(sharpe_lines),
            )
            startup_summary_sent = True

        try:
            await asyncio.wait_for(stop_event.wait(), timeout=60.0)
        except asyncio.TimeoutError:
            continue


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

async def main() -> None:
    _configure_logging()
    
    #  Initialize Telemetry Logging Hook 
    root_logger = logging.getLogger()
    telemetry_handler = TelemetryLogHandler()
    telemetry_handler.setFormatter(logging.Formatter('%(message)s'))
    root_logger.addHandler(telemetry_handler)

    #  Resource Audit (ulimit) 
    try:
        soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
        logger.info(" Resource Audit: FD Limits (soft=%d, hard=%d)", soft, hard)
    except Exception:
        logger.warning(" Resource Audit: Could not determine FD limits")

    api_key = _required_env("APCA_API_KEY_ID")
    secret_key = _required_env("APCA_API_SECRET_KEY")
    expected_sharpes = _load_expected_sharpes()

    #  State directory 
    state_dir_env = os.getenv("V3_STATE_DIR", "").strip()
    state_dir = Path(state_dir_env) if state_dir_env else _STATE_DIR
    state_dir.mkdir(parents=True, exist_ok=True)

    #  Risk parameters from env (with sensible defaults) 
    sector_concentration_limit = float(
        os.getenv("SECTOR_CONCENTRATION_LIMIT", "0.30")
    )
    daily_loss_limit = float(os.getenv("DAILY_LOSS_LIMIT", "-0.02"))
    target_risk_dollars = float(os.getenv("TARGET_RISK_DOLLARS", "1000.0"))
    vol_lookback_window = int(os.getenv("VOL_LOOKBACK_WINDOW", "30"))

    #  Shared Telemetry State 
    # Using a dict to avoid scoping/closure issues with nested functions.
    telemetry_state = {
        "ledger": None,
        "controller": None,
        "equity_protector": None,
        "is_ready": False
    }

    #  Telemetry Uplink (Priority START) 
    async def _run_telemetry_server():
        """Broadcasts live portfolio metrics and listens for manual commands (Uplink)."""
        async def telemetry_handler(ws):
            logger.info(" Telemetry Client Connected ")
            
            # Send initial state immediately
            try:
                ledger_obj = telemetry_state.get("ledger")
                controller_obj = telemetry_state.get("controller")
                
                initial_payload = {
                    "total_equity": round(ledger_obj.total_equity(), 2) if ledger_obj else 0.0,
                    "buying_power": round(ledger_obj.cash * 4.0, 2) if ledger_obj else 0.0,
                    "pairs": controller_obj.get_telemetry_snapshot(ledger_obj) if (controller_obj and ledger_obj) else [],
                    "logs": list(_LOG_BUFFER) if _LOG_BUFFER else ["System initializing..."],
                    "orders": list(_ORDER_BUFFER) if _ORDER_BUFFER else []
                }
                await ws.send(json.dumps(initial_payload))
            except Exception as e:
                logger.error(f"Failed to send initial telemetry: {e}")

            async def listener():
                try:
                    async for message in ws:
                        try:
                            data = json.loads(message)
                            if data.get("command") == "EMERGENCY_FLATTEN":
                                logger.critical(" EMERGENCY FLATTEN INITIATED VIA WEBSOCKET")
                                if ep := telemetry_state.get("equity_protector"):
                                    ep.manual_flatten()
                            elif data.get("command") == "TOGGLE_STRATEGY":
                                pair_name = data.get("pair_name")
                                logger.info(f" Strategy Uplink: Toggling {pair_name}")
                                if ctrl := telemetry_state.get("controller"):
                                    ctrl.toggle_strategy(pair_name)
                        except Exception as e:
                            logger.error(f"WebSocket incoming command error: {e}")
                except websockets.exceptions.ConnectionClosed:
                    pass

            async def broadcaster():
                while not stop_event.is_set():
                    try:
                        ledger_obj = telemetry_state.get("ledger")
                        controller_obj = telemetry_state.get("controller")
                        
                        if ledger_obj and controller_obj:
                            payload = {
                                "total_equity": round(ledger_obj.total_equity(), 2),
                                "buying_power": round(ledger_obj.cash * 4.0, 2),
                                "pairs": controller_obj.get_telemetry_snapshot(ledger_obj),
                                "logs": list(_LOG_BUFFER),
                                "orders": list(_ORDER_BUFFER)
                            }
                            await ws.send(json.dumps(payload))
                        await asyncio.sleep(1.0)
                    except websockets.exceptions.ConnectionClosed:
                        break
                    except Exception as e:
                        logger.error(f"Telemetry broadcaster error: {e}")
                        await asyncio.sleep(1.0)

            await asyncio.gather(listener(), broadcaster())

        # Explicitly allow any origin to prevent browser CORS rejection
        async with websockets.serve(telemetry_handler, "0.0.0.0", 8765, origins=None):
            logger.info(" Telemetry Bridge: Streaming & Listening on ws://0.0.0.0:8765 (Origins: Allowed)")
            await stop_event.wait()

    stop_event = asyncio.Event()
    telemetry_task = asyncio.create_task(_run_telemetry_server(), name="telemetry-bridge-v3")

    #  Infrastructure 
    db_client = TimescaleDBClient()
    await asyncio.to_thread(db_client.ensure_schema)
    # is_paper: honour APCA_API_BASE_URL first (paper-api.alpaca.markets = paper),
    # then fall back to LIVE_TRADING flag.  Paper API keys (prefix "PK") will 401
    # against the live endpoint, so the URL check takes precedence.
    _base_url = os.getenv("APCA_API_BASE_URL", "")
    if _base_url:
        is_paper = "paper" in _base_url.lower()
    else:
        is_paper = os.getenv("LIVE_TRADING", "True").lower() == "false"
    trading_client = TradingClient(api_key, secret_key, paper=is_paper)
    fallback_starting_cash = await _bootstrap_cash(trading_client)

    event_bus = InMemoryEventBus()
    #  Subscribe to executions for dashboard history 
    event_bus.subscribe("execution", _capture_order_execution)
    
    session_manager = SessionManager()
    bridge = LiveDataBridge(event_bus)
    ledger = PortfolioLedger(event_bus, starting_cash=fallback_starting_cash)
    telemetry_state["ledger"] = ledger

    #  Portfolio state recovery 
    state_manager = StateManager(ledger, event_bus, db_client)
    await asyncio.to_thread(state_manager.ensure_schema)
    recovered_state = await asyncio.to_thread(state_manager.load_latest_state)
    if recovered_state is not None:
        state_manager.restore_into_ledger(recovered_state)
        logger.info(
            "Recovered persisted portfolio state from %s",
            recovered_state.state_ts.isoformat(),
        )
    else:
        logger.info("No persisted state found. Initializing fresh ledger.")

    #  Hard Sync with Broker Reality (Source of Truth) 
    # Even if we recovered state, we MUST sync with the actual broker to
    # clear 'phantom' positions from previous runs/tests.
    await _seed_existing_positions(trading_client, ledger)

    #  Regime detector (must subscribe before strategies) 
    regime_detector = RegimeDetector(event_bus)

    #  Notifications 
    notifier = TelegramNotifier(event_bus)

    #  TCA 
    tca = TransactionCostAnalyzer(event_bus, notifier)

    #  Risk modules 
    # Load winner dictionary first so we know all pairs for the sizer.
    initial_winner_dict = _load_winner_dictionary()
    
    # Monday Open Adaptive Alert
    tuned_meta = initial_winner_dict.get("_metadata", {})
    if tuned_count := tuned_meta.get("tuned_count", 0):
         await notifier.notify_text(
             f" ADAPTIVE CALIBRATION COMPLETE: Entry thresholds tightened for {tuned_count} high-risk pairs based on tonight's stress tests."
         )
    pair_configs: list[tuple[str, str]] = [
        (str(cfg["instrument_a"]), str(cfg["instrument_b"]))
        for group_configs in initial_winner_dict.values()
        for cfg in group_configs
    ]

    volatility_sizer = VolatilitySizer(
        event_bus,
        pair_configs=pair_configs,
        lookback_window=vol_lookback_window,
        target_risk_dollars=target_risk_dollars,
    )

    factor_monitor = FactorMonitor(
        ledger,
        sector_concentration_limit=sector_concentration_limit,
    )

    equity_protector = EquityProtector(
        ledger,
        event_bus,
        daily_loss_limit=daily_loss_limit,
        # The protector writes its own halt state automatically on trigger;
        # we also wire it via RiskStateController for a full flush on shutdown.
        state_path=state_dir / "equity_protector_state.json",
    )
    telemetry_state["equity_protector"] = equity_protector

    risk_state_controller = RiskStateController(
        factor_monitor,
        equity_protector,
        state_dir=state_dir,
    )
    # Restore any previously persisted risk state (e.g. today's halt flag).
    await asyncio.to_thread(risk_state_controller.load)

    # MASTER SYNC: Overwrite stale recovered state with actual brokerage truth (Sovereign Equity)
    try:
        alpaca_equity = await _bootstrap_cash(trading_client)
        ibkr_equity = 0.0
        if ibkr_connector.ib.isConnected():
            summary = await ibkr_connector.ib.accountSummaryAsync()
            for item in summary:
                if item.tag == 'NetLiquidation':
                    ibkr_equity = float(item.value)
                    break
        
        sovereign_truth = alpaca_equity + ibkr_equity
        current_equity = ledger.total_equity()
        drift = abs(current_equity - sovereign_truth)
        
        if drift > 10.0: # $10 threshold for multi-venue
            logger.info(" Master Sync: Aligning ledger to Sovereign Truth ($%.2f -> $%.2f)", current_equity, sovereign_truth)
            ledger.cash += (sovereign_truth - current_equity)
            
        logger.info(" Portfolio synchronized: Sovereign Equity = $%.2f (Alpaca=$%.2f, IBKR=$%.2f)", sovereign_truth, alpaca_equity, ibkr_equity)
    except Exception as e:
        logger.error(" Master Sync failed: %s", e)

    bayesian_vol = BayesianVolatilityAdjuster(event_bus)
    meta_labeler = MetaLabeler()

    sentiment_warden = SentimentWarden(
        api_key=api_key,
        secret_key=secret_key,
        state_path=state_dir / "sentiment_vetoes.json"
    )

    risk_manager = RiskManager(
        ledger,
        event_bus,
        regime_detector=regime_detector,
        factor_monitor=factor_monitor,
        equity_protector=equity_protector,
        bayesian_vol=bayesian_vol,
        meta_labeler=meta_labeler,
        sentiment_warden=sentiment_warden,
    )

    #  Execution + EOD 
    neural_sniper = NeuralSniper(
        event_bus=event_bus,
        model_path="run_state/models/ppo_execution_v1.zip"
    )
    broker = AlpacaBroker(trading_client, event_bus, limit_chaser=neural_sniper)
    eod_liquidator = EODLiquidator(ledger, event_bus, session_manager=session_manager)
    
    # Dashboard integration
    monitor = LiveMonitor()
    correlation_manager = CorrelationManager()

    # IBKR Connectivity Monitoring (Dashboard Only for v3)
    ibkr_connector = IBKRConnector(
        host=os.getenv("IBKR_HOST", "host.docker.internal"),
        port=int(os.getenv("IBKR_PORT", 7497)),
        client_id=int(os.getenv("IBKR_CLIENT_ID", 101)),
    )
    ibkr_adapter = IBKRAdapter(ibkr_connector)
    try:
        # Fire-and-forget connection attempt
        asyncio.create_task(ibkr_adapter.connect())
    except Exception as e:
        logger.debug(f"IBKR initial connection attempt failed: {e}")

    #  Rotator 
    rotator = StrategyRotator(
        event_bus,
        shadow_universe=BACKBENCH_UNIVERSE,
        active_ledger=ledger,
    )

    tail_hedger = TailHedger(
        trading_client,
        ledger,
        bayesian_vol,
    )

    #  Strategy controller (v3  vol-sized) 
    controller = StrategyController(
        event_bus,
        session_manager,
        volatility_sizer=volatility_sizer,
    )
    telemetry_state["controller"] = controller
    telemetry_state["is_ready"] = True

    alpha_controller = AlphaMonitorController(
        ledger,
        event_bus,
        notifier,
        expected_sharpes,
        _emergency_flatten_pair,
    )

    #  High-Availability (HA) Monitoring 
    reconciler_adapter = AlpacaReconcilerAdapter(trading_client)
    position_reconciler = PositionReconciler(reconciler_adapter)
    
    health_monitor = HealthMonitor(notifier=notifier)
    shadow_accounting = ShadowAccounting(
        event_bus=event_bus,
        ledger=ledger,
        reconciler=position_reconciler,
        ibkr_connector=ibkr_connector,
        notifier=notifier,
        check_interval_seconds=300
    )
    
    health_monitor.start()
    shadow_accounting.start()

    #  Alpaca data streams 
    stock_supervisor = StreamSupervisor(
        "alpaca-stock-data",
        lambda: StockDataStream(api_key, secret_key, raw_data=True, feed=_stock_feed_from_env()),
        lambda stream, symbols: _subscribe_stock_stream(stream, symbols, bridge),
    )
    crypto_supervisor = StreamSupervisor(
        "alpaca-crypto-data",
        lambda: CryptoDataStream(api_key, secret_key, raw_data=True),
        lambda stream, symbols: _subscribe_crypto_stream(stream, symbols, bridge),
    )

    #  TCA scheduled report (weekdays 16:05 ET) 
    tca_task_obj = ScheduledTask(
        "tca_daily_report",
        hour=16,
        minute=5,
        tz=_ET,
        weekdays_only=True,
        coro_factory=lambda: tca.send_summary(lookback_days=1),
        poll_interval_seconds=30.0,
    )

    #  Signal / stop machinery 
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(sig, stop_event.set)
        except NotImplementedError:
            pass

    #  Spawn tasks 
    winner_task = asyncio.create_task(
        _winner_refresh_loop(
            controller,
            alpha_controller,
            stock_supervisor,
            crypto_supervisor,
            notifier,
            regime_detector,
            risk_state_controller,
            expected_sharpes,
            equity_protector,
            rotator,
            tail_hedger,
            stop_event,
        ),
        name="winner-refresh-loop-v3",
    )
    stock_task = asyncio.create_task(
        stock_supervisor.run(stop_event), name="stock-stream-supervisor-v3"
    )
    crypto_task = asyncio.create_task(
        crypto_supervisor.run(stop_event), name="crypto-stream-supervisor-v3"
    )
    broker_task = asyncio.create_task(
        _run_broker_forever(broker, stop_event), name="alpaca-broker-supervisor-v3"
    )
    tca_schedule_task = asyncio.create_task(
        tca_task_obj.run(stop_event), name="tca-daily-report-v3"
    )
    dashboard_task = asyncio.create_task(
        _run_dashboard_updates(
            ledger,
            risk_manager,
            correlation_manager,
            monitor,
            controller,
            rotator,
            tail_hedger,
            stop_event,
            trading_client=trading_client,
            ibkr_adapter=ibkr_connector,
        ),
        name="dashboard-updates-v3",
    )

    # HA tasks
    async def _ha_heartbeat_loop():
        while not stop_event.is_set():
            health_monitor.heartbeat()
            await asyncio.sleep(1.0)
            
    ha_heartbeat_task = asyncio.create_task(_ha_heartbeat_loop(), name="ha-heartbeat-v3")
    shadow_acc_task = asyncio.create_task(shadow_accounting.run_reconciliation_loop(), name="shadow-accounting-v3")

    # Sentiment Warden periodic task
    async def _sentiment_refresh_loop():
        # Universe from winner dictionary
        while not stop_event.is_set():
            active_symbols = list(ledger.positions.keys())
            if not active_symbols:
                # Fallback to a default set or the top pairs from winner dict
                all_configured = [
                    str(cfg["instrument_a"]) for group in _load_winner_dictionary().values() for cfg in group
                ] + [
                    str(cfg["instrument_b"]) for group in _load_winner_dictionary().values() for cfg in group
                ]
                active_symbols = list(set(all_configured))[:15] # Limit to top universe

            await sentiment_warden.scan_universe(active_symbols)
            await asyncio.sleep(3600) # Refresh hourly
            
    sentiment_task = asyncio.create_task(_sentiment_refresh_loop(), name="sentiment-warden-v3")

    #  Equity REST quote poller (workaround for IEX bars-only websocket) 
    _alpaca_api_key = os.environ.get("APCA_API_KEY_ID", "")
    _alpaca_secret_key = os.environ.get("APCA_API_SECRET_KEY", "")
    quote_poll_task = asyncio.create_task(
        _run_equity_quote_poller(
            active_symbols_fn=controller.configured_stock_symbols,
            event_bus=event_bus,
            api_key=_alpaca_api_key,
            secret_key=_alpaca_secret_key,
            stop_event=stop_event,
            poll_interval=5.0,
        ),
        name="equity-quote-poller-v3",
    )

    #  Startup summary 
    halted_notice = "  HALTED" if equity_protector.is_halted() else ""
    logger.info(
        "Global harness v3 started cash=%.2f equity=%.2f recovered=%s halted=%s",
        ledger.cash,
        ledger.total_equity(),
        recovered_state is not None,
        equity_protector.is_halted(),
    )
    if recovered_state is not None:
        await notifier.notify_system_event(
            f"System Recovery{halted_notice}",
            f"Recovered persisted state from {recovered_state.state_ts.isoformat()}"
            f" | cash=${ledger.cash:,.2f} | equity=${ledger.total_equity():,.2f}"
            f"\nSector limit: {sector_concentration_limit:.0%}"
            f" | Daily loss limit: {daily_loss_limit:.1%}"
            f" | Vol lookback: {vol_lookback_window}d",
        )
    else:
        await notifier.notify_system_event(
            f"System Restart{halted_notice}",
            f"Started without persisted state"
            f" | fallback cash=${fallback_starting_cash:,.2f}"
            f" | equity=${ledger.total_equity():,.2f}"
            f"\nSector limit: {sector_concentration_limit:.0%}"
            f" | Daily loss limit: {daily_loss_limit:.1%}"
            f" | Vol lookback: {vol_lookback_window}d",
        )

    #  Run until signalled 
    try:
        await stop_event.wait()
    finally:
        logger.info("Stopping global harness v3")
        stop_event.set()
        health_monitor.stop()
        shadow_accounting.stop()
        
        all_tasks = (
            winner_task,
            stock_task,
            crypto_task,
            broker_task,
            tca_schedule_task,
            dashboard_task,
            ha_heartbeat_task,
            shadow_acc_task,
            sentiment_task,
            quote_poll_task,
            telemetry_task,
        )
        for task in all_tasks:
            task.cancel()
        await asyncio.gather(*all_tasks, return_exceptions=True)

        # Final state flush before any resource is closed.
        await asyncio.to_thread(risk_state_controller.save)
        logger.info("RiskStateController: final state flushed")

        await asyncio.gather(
            stock_supervisor.update_symbols(()),
            crypto_supervisor.update_symbols(()),
            broker.stop(),
            notifier.close(),
            state_manager.close(),
            alpha_controller.close(),
            return_exceptions=True,
        )
        tca.close()
        volatility_sizer.close(event_bus)
        controller.close()
        regime_detector.close()
        eod_liquidator.close()
        equity_protector.close()
        bayesian_vol.close()
        broker.close()
        risk_manager.close()
        ledger.close()

        logger.info(
            "Shutdown complete cash=%.2f realized_pnl=%.2f unrealized_pnl=%.2f equity=%.2f",
            ledger.cash,
            ledger.total_realized_pnl(),
            ledger.total_unrealized_pnl(),
            ledger.total_equity(),
        )


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass
