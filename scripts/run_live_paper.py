from __future__ import annotations

import asyncio
import logging
import os
import signal
import sys
from pathlib import Path

from alpaca.data.enums import DataFeed
from alpaca.data.live.stock import StockDataStream
from alpaca.trading.client import TradingClient

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.symbols import normalize_symbol
from quant_system.analytics.notifier import DiscordNotifier
from quant_system.core.bus import InMemoryEventBus
from quant_system.data.normalizers.live_bridge import LiveDataBridge
from quant_system.execution.brokers.alpaca_broker import AlpacaBroker
from quant_system.portfolio.ledger import PortfolioLedger
from quant_system.risk.eod_liquidator import EODLiquidator
from quant_system.risk.manager import RiskManager
from quant_system.strategies.sma_crossover import SMACrossoverStrategy


logger = logging.getLogger("quant_system.live_paper")


def _configure_logging() -> None:
    logging.basicConfig(
        level=os.getenv("LIVE_LOG_LEVEL", "INFO").upper(),
        format="%(asctime)s %(levelname)s %(name)s - %(message)s",
    )


def _required_env(name: str) -> str:
    value = os.getenv(name, "").strip()
    if not value:
        raise RuntimeError(f"{name} must be set")
    return value


def _data_feed_from_env() -> DataFeed:
    raw_feed = os.getenv("ALPACA_DATA_FEED", "iex").strip().lower()
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

    if not positions:
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
            "Seeded live position instrument=%s qty=%.6f avg_price=%.4f",
            instrument_id,
            position.quantity,
            position.avg_price,
        )


async def main() -> None:
    _configure_logging()

    api_key = _required_env("APCA_API_KEY_ID")
    secret_key = _required_env("APCA_API_SECRET_KEY")
    instrument_id = normalize_symbol(os.getenv("LIVE_SYMBOL", "AAPL").strip().upper())

    # Use the latest WFO winner here by environment override.
    short_window = int(os.getenv("LIVE_SMA_SHORT_WINDOW", "20"))
    long_window = int(os.getenv("LIVE_SMA_LONG_WINDOW", "50"))
    long_notional = float(os.getenv("LIVE_SMA_LONG_NOTIONAL", "5000"))

    trading_client = TradingClient(api_key, secret_key, paper=True)
    starting_cash = await _bootstrap_cash(trading_client)

    event_bus = InMemoryEventBus()
    bridge = LiveDataBridge(event_bus)
    ledger = PortfolioLedger(event_bus, starting_cash=starting_cash)
    await _seed_existing_positions(trading_client, ledger)
    risk_manager = RiskManager(ledger, event_bus)
    eod_liquidator = EODLiquidator(ledger, event_bus)
    broker = AlpacaBroker(trading_client, event_bus)
    notifier = DiscordNotifier(event_bus)
    strategy = SMACrossoverStrategy(
        event_bus,
        instrument_id=instrument_id,
        short_window=short_window,
        long_window=long_window,
        long_notional=long_notional,
    )

    data_stream = StockDataStream(
        api_key,
        secret_key,
        raw_data=True,
        feed=_data_feed_from_env(),
    )
    data_stream.subscribe_bars(bridge.publish_alpaca, instrument_id)
    data_stream.subscribe_trades(bridge.publish_alpaca, instrument_id)
    data_stream.subscribe_quotes(bridge.publish_alpaca, instrument_id)

    logger.info(
        "Starting live paper trading symbol=%s cash=%.2f short_window=%d long_window=%d long_notional=%.2f",
        instrument_id,
        ledger.cash,
        short_window,
        long_window,
        long_notional,
    )

    stop_event = asyncio.Event()
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(sig, stop_event.set)
        except NotImplementedError:
            pass

    data_task = asyncio.create_task(data_stream._run_forever(), name="alpaca-market-data")  # noqa: SLF001
    broker_task = asyncio.create_task(broker.run(), name="alpaca-trade-updates")

    try:
        await stop_event.wait()
    finally:
        logger.info("Stopping live paper trading")
        data_task.cancel()
        broker_task.cancel()
        await asyncio.gather(data_task, broker_task, return_exceptions=True)
        await asyncio.gather(data_stream.stop_ws(), broker.stop(), return_exceptions=True)
        strategy.close()
        await notifier.close()
        broker.close()
        eod_liquidator.close()
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
