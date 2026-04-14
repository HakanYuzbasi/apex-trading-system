from __future__ import annotations

import asyncio
import json
import logging
import os
import signal
import sys
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable

from alpaca.data.enums import DataFeed
from alpaca.data.live.crypto import CryptoDataStream
from alpaca.data.live.stock import StockDataStream
from alpaca.trading.client import TradingClient

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.symbols import AssetClass, normalize_symbol, parse_symbol
from quant_system.analytics.notifier import TelegramNotifier
from quant_system.core.bus import InMemoryEventBus
from quant_system.core.calendars import SessionManager
from quant_system.data.normalizers.live_bridge import LiveDataBridge
from quant_system.data.stores.client import TimescaleDBClient
from quant_system.execution.brokers.alpaca_broker import AlpacaBroker
from quant_system.portfolio.ledger import PortfolioLedger
from quant_system.portfolio.persistence import StateManager
from quant_system.risk.eod_liquidator import EODLiquidator
from quant_system.risk.manager import RiskManager
from quant_system.strategies.pairs_stat_arb import PairsStatArbStrategy


logger = logging.getLogger("quant_system.global_harness")


DEFAULT_WINNER_DICTIONARY: dict[str, list[dict[str, Any]]] = {
    "crypto": [
        {
            "slot": "crypto_primary",
            "strategy": "pairs_stat_arb",
            "instrument_a": "CRYPTO:BTC/USD",
            "instrument_b": "CRYPTO:ETH/USD",
            "lookback_window": 50,
            "entry_z_score": 2.0,
            "exit_z_score": 0.5,
            "leg_notional": 2_000.0,
        }
    ],
    "equity": [
        {
            "slot": "equity_primary",
            "strategy": "pairs_stat_arb",
            "instrument_a": "AAPL",
            "instrument_b": "MSFT",
            "lookback_window": 50,
            "entry_z_score": 2.0,
            "exit_z_score": 0.5,
            "leg_notional": 5_000.0,
        }
    ],
}


def _configure_logging() -> None:
    logging.basicConfig(
        level=os.getenv("GLOBAL_LOG_LEVEL", "INFO").upper(),
        format="%(asctime)s %(levelname)s %(name)s - %(message)s",
    )


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
    return float(account.cash)


async def _seed_existing_positions(trading_client: TradingClient, ledger: PortfolioLedger) -> None:
    try:
        positions = await asyncio.to_thread(trading_client.get_all_positions)
    except Exception:
        logger.exception("Unable to seed existing Alpaca positions into the ledger")
        return

    for remote_position in positions:
        instrument_id = normalize_symbol(str(remote_position.symbol).strip().upper())
        position = ledger.get_position(instrument_id)
        position.quantity = float(remote_position.qty)
        position.avg_price = float(remote_position.avg_entry_price)
        current_price = getattr(remote_position, "current_price", None)
        if current_price is not None:
            ledger.last_price_by_instrument[instrument_id] = float(current_price)
        logger.info(
            "Seeded existing position instrument=%s qty=%.6f avg_price=%.4f",
            instrument_id,
            position.quantity,
            position.avg_price,
        )


def _alpaca_symbol(instrument_id: str) -> str:
    parsed = parse_symbol(instrument_id)
    if parsed.asset_class == AssetClass.CRYPTO:
        return f"{parsed.base}/{parsed.quote}"
    return parsed.base


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
            normalized_config["instrument_a"] = normalize_symbol(str(config["instrument_a"]).strip().upper())
            normalized_config["instrument_b"] = normalize_symbol(str(config["instrument_b"]).strip().upper())
            normalized_config.setdefault("slot", f"{asset_group}_{normalized_config['instrument_a']}_{normalized_config['instrument_b']}")
            normalized_config.setdefault("strategy", "pairs_stat_arb")
            normalized[asset_group].append(normalized_config)
    return normalized


@dataclass(frozen=True, slots=True)
class StrategySlotConfig:
    slot: str
    asset_group: str
    strategy: str
    instrument_a: str
    instrument_b: str
    lookback_window: int
    entry_z_score: float
    exit_z_score: float
    leg_notional: float

    @property
    def signature(self) -> tuple[object, ...]:
        return (
            self.strategy,
            self.instrument_a,
            self.instrument_b,
            self.lookback_window,
            self.entry_z_score,
            self.exit_z_score,
            self.leg_notional,
        )


class StrategyController:
    def __init__(self, event_bus: InMemoryEventBus, session_manager: SessionManager) -> None:
        self._event_bus = event_bus
        self._session_manager = session_manager
        self._configured_slots: dict[str, StrategySlotConfig] = {}
        self._active_strategies: dict[str, tuple[StrategySlotConfig, PairsStatArbStrategy]] = {}

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
                    lookback_window=int(config["lookback_window"]),
                    entry_z_score=float(config["entry_z_score"]),
                    exit_z_score=float(config["exit_z_score"]),
                    leg_notional=float(config["leg_notional"]),
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

            if active is not None and active[0].signature == config.signature:
                continue

            if active is not None:
                self._close_slot(slot)
            self._active_strategies[slot] = (config, self._instantiate_strategy(config))
            logger.info("Activated strategy slot=%s instruments=%s/%s", slot, config.instrument_a, config.instrument_b)

    def close(self) -> None:
        for slot in list(self._active_strategies):
            self._close_slot(slot)

    def _instantiate_strategy(self, config: StrategySlotConfig) -> PairsStatArbStrategy:
        if config.strategy != "pairs_stat_arb":
            raise ValueError(f"Unsupported strategy in global harness: {config.strategy}")
        return PairsStatArbStrategy(
            self._event_bus,
            instrument_a=config.instrument_a,
            instrument_b=config.instrument_b,
            lookback_window=config.lookback_window,
            entry_z_score=config.entry_z_score,
            exit_z_score=config.exit_z_score,
            leg_notional=config.leg_notional,
        )

    def _should_be_active(self, config: StrategySlotConfig) -> bool:
        asset_class = self._slot_asset_class(config)
        if asset_class == AssetClass.CRYPTO:
            return True
        return (
            self._session_manager.is_market_open(config.instrument_a)
            and self._session_manager.is_market_open(config.instrument_b)
        )

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
                await stream._run_forever()  # noqa: SLF001 - Alpaca stream exposes coroutine runner
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


def _subscribe_stock_stream(stream: StockDataStream, symbols: tuple[str, ...], bridge: LiveDataBridge) -> None:
    if not symbols:
        return
    stream.subscribe_bars(bridge.publish_alpaca, *symbols)
    stream.subscribe_trades(bridge.publish_alpaca, *symbols)
    stream.subscribe_quotes(bridge.publish_alpaca, *symbols)


def _subscribe_crypto_stream(stream: CryptoDataStream, symbols: tuple[str, ...], bridge: LiveDataBridge) -> None:
    if not symbols:
        return
    stream.subscribe_bars(bridge.publish_alpaca, *symbols)
    stream.subscribe_trades(bridge.publish_alpaca, *symbols)
    stream.subscribe_quotes(bridge.publish_alpaca, *symbols)


async def _winner_refresh_loop(
    controller: StrategyController,
    stock_supervisor: StreamSupervisor,
    crypto_supervisor: StreamSupervisor,
    stop_event: asyncio.Event,
) -> None:
    last_refresh_at: datetime | None = None
    while not stop_event.is_set():
        now = datetime.now(timezone.utc)
        should_refresh = last_refresh_at is None or (now - last_refresh_at) >= timedelta(minutes=60)
        if should_refresh:
            winner_dictionary = _load_winner_dictionary()
            controller.apply_winner_dictionary(winner_dictionary)
            await stock_supervisor.update_symbols(controller.configured_stock_symbols())
            await crypto_supervisor.update_symbols(controller.configured_crypto_symbols())
            last_refresh_at = now
            logger.info("Winner dictionary refreshed")

        controller.reconcile()
        try:
            await asyncio.wait_for(stop_event.wait(), timeout=60.0)
        except asyncio.TimeoutError:
            continue


async def main() -> None:
    _configure_logging()

    api_key = _required_env("APCA_API_KEY_ID")
    secret_key = _required_env("APCA_API_SECRET_KEY")

    db_client = TimescaleDBClient()
    await asyncio.to_thread(db_client.ensure_schema)
    trading_client = TradingClient(api_key, secret_key, paper=True)
    fallback_starting_cash = await _bootstrap_cash(trading_client)

    event_bus = InMemoryEventBus()
    session_manager = SessionManager()
    bridge = LiveDataBridge(event_bus)
    ledger = PortfolioLedger(event_bus, starting_cash=fallback_starting_cash)
    state_manager = StateManager(ledger, event_bus, db_client)
    await asyncio.to_thread(state_manager.ensure_schema)
    recovered_state = await asyncio.to_thread(state_manager.load_latest_state)
    recovered = recovered_state is not None
    if recovered_state is not None:
        state_manager.restore_into_ledger(recovered_state)
        logger.info("Recovered persisted portfolio state from %s", recovered_state.state_ts.isoformat())
    else:
        await _seed_existing_positions(trading_client, ledger)
    risk_manager = RiskManager(ledger, event_bus)
    broker = AlpacaBroker(trading_client, event_bus)
    notifier = TelegramNotifier(event_bus)
    eod_liquidator = EODLiquidator(ledger, event_bus, session_manager=session_manager)
    controller = StrategyController(event_bus, session_manager)

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

    stop_event = asyncio.Event()
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(sig, stop_event.set)
        except NotImplementedError:
            pass

    winner_task = asyncio.create_task(
        _winner_refresh_loop(controller, stock_supervisor, crypto_supervisor, stop_event),
        name="winner-refresh-loop",
    )
    stock_task = asyncio.create_task(stock_supervisor.run(stop_event), name="stock-stream-supervisor")
    crypto_task = asyncio.create_task(crypto_supervisor.run(stop_event), name="crypto-stream-supervisor")
    broker_task = asyncio.create_task(_run_broker_forever(broker, stop_event), name="alpaca-broker-supervisor")

    logger.info(
        "Global harness started cash=%.2f equity=%.2f",
        ledger.cash,
        ledger.total_equity(),
    )
    if recovered_state is not None:
        await notifier.notify_system_event(
            "System Recovery",
            f"Recovered persisted state from {recovered_state.state_ts.isoformat()} | cash=${ledger.cash:,.2f} | equity=${ledger.total_equity():,.2f}",
        )
    else:
        await notifier.notify_system_event(
            "System Restart",
            f"Started without persisted state | fallback cash=${fallback_starting_cash:,.2f} | equity=${ledger.total_equity():,.2f}",
        )

    try:
        await stop_event.wait()
    finally:
        logger.info("Stopping global harness")
        stop_event.set()
        for task in (winner_task, stock_task, crypto_task, broker_task):
            task.cancel()
        await asyncio.gather(winner_task, stock_task, crypto_task, broker_task, return_exceptions=True)
        await asyncio.gather(
            stock_supervisor.update_symbols(()),
            crypto_supervisor.update_symbols(()),
            broker.stop(),
            notifier.close(),
            state_manager.close(),
            return_exceptions=True,
        )
        controller.close()
        eod_liquidator.close()
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
