import pandas as pd

from scripts.god_level_backtest import GodLevelBacktester, Position, Trade


def _price_series(length: int = 80, start: float = 100.0) -> pd.Series:
    idx = pd.bdate_range("2025-01-01", periods=length)
    vals = [start + 0.1 * i for i in range(length)]
    return pd.Series(vals, index=idx)


def test_short_cashflow_and_equity_accounting():
    backtester = GodLevelBacktester(initial_capital=100_000)
    backtester.risk_manager = None  # deterministic sizing path

    date = pd.Timestamp("2025-05-01")
    row = pd.Series({"Close": 100.0, "High": 101.0, "Low": 99.0})
    prices = _price_series()

    entered = backtester._enter_position(
        symbol="AAPL",
        row=row,
        signal=-0.9,
        confidence=0.9,
        regime="bear",
        date=date,
        prices=prices,
        current_equity=100_000.0,
    )
    assert entered is True
    assert "AAPL" in backtester.positions

    position = backtester.positions["AAPL"]
    assert position.direction == "short"
    assert position.shares > 0

    marked_equity = backtester._calculate_portfolio_value({"AAPL": pd.Series({"Close": 100.0})})
    assert marked_equity < backtester.capital  # short liability is subtracted

    backtester._exit_position("AAPL", exit_price=90.0, exit_reason="Test exit", exit_date=date + pd.Timedelta(days=10))
    assert len(backtester.trades) == 1
    trade = backtester.trades[0]
    assert trade.pnl > 0  # profitable short should produce positive net PnL
    assert backtester.capital == backtester.initial_capital + trade.pnl


def test_time_exit_uses_actual_holding_days():
    backtester = GodLevelBacktester(initial_capital=100_000)
    backtester.risk_manager = None

    entry_date = pd.Timestamp("2025-01-01")
    backtester.positions["MSFT"] = Position(
        symbol="MSFT",
        entry_date=entry_date,
        entry_price=100.0,
        shares=20,
        direction="long",
        stop_loss=50.0,
        take_profit=150.0,
        trailing_stop_pct=0.03,
        entry_commission=1.0,
        highest_price=100.0,
        lowest_price=100.0,
        regime="neutral",
    )

    exit_date = pd.Timestamp("2025-04-10")
    day_data = {"MSFT": pd.Series({"Close": 100.0, "High": 101.0, "Low": 99.0})}
    backtester._update_positions(day_data, exit_date)

    assert len(backtester.trades) == 1
    trade = backtester.trades[0]
    assert trade.exit_reason == "Max hold period"
    assert trade.hold_days >= 90


def test_monte_carlo_bootstrap_has_nonzero_dispersion():
    backtester = GodLevelBacktester(initial_capital=100_000)
    backtester.trades = [
        Trade(
            symbol="AAPL",
            entry_date=pd.Timestamp("2025-01-01"),
            exit_date=pd.Timestamp("2025-01-06"),
            entry_price=100.0,
            exit_price=102.0,
            shares=10,
            direction="long",
            pnl=pnl,
            pnl_percent=0.0,
            exit_reason="test",
            regime="neutral",
            hold_days=5,
        )
        for pnl in [200.0, -150.0, 300.0, -250.0, 100.0, -80.0, 180.0, -120.0, 220.0, -90.0]
    ]

    mc = backtester.run_monte_carlo(n_simulations=300)
    assert mc is not None
    assert mc["return_std"] > 0
