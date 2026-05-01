"""Portfolio-realistic daily backtester for Apex research.

This module is intentionally independent of the older round scripts.  It
models the pieces those scripts compress away: daily marked equity, concurrent
positions, cash, entry/exit costs, fee-aware entry filtering, signal-aware HRP
sizing, volatility circuit breakers, and configurable exits.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional

import numpy as np
import pandas as pd

from risk.fee_aware_edge_gate import FeeAwareEdgeGate
from risk.signal_portfolio_constructor import SignalPortfolioConstructor


@dataclass(frozen=True)
class VolCircuitConfig:
    enabled: bool = True
    ratio_scale: float = 2.0
    ratio_halt: float = 3.0
    ratio_close: float = 4.0
    scale_multiplier: float = 0.25


@dataclass(frozen=True)
class ExitConfig:
    stop_loss_pct: float = 0.03
    take_profit_pct: float = 0.06
    trailing_pct: float = 0.03
    signal_exit_prob: float = 0.50
    max_hold_bars: int = 20


@dataclass(frozen=True)
class PortfolioBacktestConfig:
    initial_capital: float = 100_000.0
    signal_threshold: float = 0.55
    kelly_fraction: float = 0.50
    max_positions: int = 8
    max_position_pct: float = 0.15
    base_position_pct: Optional[float] = None
    allow_shorts: bool = False
    expected_return_scale_pct: float = 0.02
    fee_gate_enabled: bool = True
    fee_min_edge_cost_ratio: float = 1.50
    one_way_cost_bps: float = 8.0
    half_spread_bps: float = 1.0
    hrp_enabled: bool = True
    hrp_signal_blend: float = 0.40
    hrp_min_history: int = 20
    return_lookback: int = 90
    atr_fast: int = 5
    atr_slow: int = 60
    vol_circuit: VolCircuitConfig = field(default_factory=VolCircuitConfig)
    exits: ExitConfig = field(default_factory=ExitConfig)


@dataclass
class Position:
    symbol: str
    side: int
    shares: float
    entry_price: float
    entry_date: pd.Timestamp
    entry_notional: float
    entry_cost: float
    high_water_mark: float
    low_water_mark: float
    bars_held: int = 0


@dataclass
class Trade:
    symbol: str
    entry_date: pd.Timestamp
    exit_date: pd.Timestamp
    side: str
    entry_price: float
    exit_price: float
    shares: float
    pnl: float
    pnl_pct: float
    entry_cost: float
    exit_cost: float
    exit_reason: str
    bars_held: int


@dataclass
class BacktestResult:
    total_return_pct: float
    sharpe: float
    max_drawdown_pct: float
    trades: int
    win_rate: float
    profit_factor: float
    exposure_mean: float
    final_equity: float
    equity_curve: pd.Series
    daily_returns: pd.Series
    trade_log: List[Trade]
    diagnostics: Dict[str, float | int | Dict[str, int]]


class _VolCircuit:
    def __init__(self, config: VolCircuitConfig) -> None:
        self._config = config

    def evaluate(self, atr_fast: float, atr_slow: float) -> tuple[float, str, float]:
        if not self._config.enabled or atr_slow <= 0 or not np.isfinite(atr_slow):
            return 1.0, "normal", 0.0
        ratio = float(atr_fast) / float(atr_slow)
        if ratio > self._config.ratio_close:
            return 0.0, "close", ratio
        if ratio > self._config.ratio_halt:
            return 0.0, "halt", ratio
        if ratio > self._config.ratio_scale:
            return self._config.scale_multiplier, "scale", ratio
        return 1.0, "normal", ratio


class RealisticPortfolioBacktester:
    """Daily, long-biased portfolio simulator.

    Args:
        price_data: symbol -> OHLCV frame with at least Close/High/Low.
        probabilities: symbol -> probability/confidence series indexed by date.
        config: portfolio and risk configuration.
    """

    def __init__(
        self,
        price_data: Dict[str, pd.DataFrame],
        probabilities: Dict[str, pd.Series],
        config: Optional[PortfolioBacktestConfig] = None,
    ) -> None:
        self.price_data = {s: df.sort_index() for s, df in price_data.items()}
        self.probabilities = {s: p.sort_index() for s, p in probabilities.items()}
        self.config = config or PortfolioBacktestConfig()
        self.cash = float(self.config.initial_capital)
        self.positions: Dict[str, Position] = {}
        self.trades: List[Trade] = []
        self._block_reasons: Dict[str, int] = {}
        self._entries_by_mode: Dict[str, int] = {}
        self._vol_circuit = _VolCircuit(self.config.vol_circuit)
        self._fee_gate = FeeAwareEdgeGate()
        self._fee_gate._min_ratio = max(1.0, float(self.config.fee_min_edge_cost_ratio))
        self._portfolio_constructor = SignalPortfolioConstructor(
            max_single_weight=self.config.max_position_pct,
            min_history=self.config.hrp_min_history,
            signal_blend=self.config.hrp_signal_blend,
            recompute_interval_s=0.0,
        )
        self._atr_fast = {
            sym: self._atr(df, self.config.atr_fast) for sym, df in self.price_data.items()
        }
        self._atr_slow = {
            sym: self._atr(df, self.config.atr_slow) for sym, df in self.price_data.items()
        }

    def run(self) -> BacktestResult:
        dates = self._common_dates()
        if dates.empty:
            return self._empty_result()

        equity_points: List[tuple[pd.Timestamp, float]] = []
        exposure_points: List[float] = []

        for date in dates:
            self._update_portfolio_constructor(date)
            self._process_exits(date)
            equity_before_entries = self._mark_equity(date)
            self._process_entries(date, equity_before_entries)
            equity = self._mark_equity(date)
            exposure_points.append(self._gross_exposure(date, equity))
            equity_points.append((date, equity))

        last_date = dates[-1]
        for symbol in list(self.positions):
            self._exit_position(symbol, last_date, "end_of_test")

        if equity_points:
            equity_points[-1] = (last_date, self._mark_equity(last_date))

        equity_curve = pd.Series(
            [v for _, v in equity_points],
            index=pd.DatetimeIndex([d for d, _ in equity_points]),
            dtype=float,
        )
        return self._build_result(equity_curve, exposure_points)

    def _process_entries(self, date: pd.Timestamp, portfolio_value: float) -> None:
        slots = self.config.max_positions - len(self.positions)
        if slots <= 0 or portfolio_value <= 0:
            return

        candidates = []
        for symbol, prob_series in self.probabilities.items():
            if symbol in self.positions:
                continue
            prob = self._value_at(prob_series, date)
            if prob is None:
                continue
            side = 1
            edge_prob = float(prob) - 0.5
            if edge_prob < self.config.signal_threshold - 0.5:
                if self.config.allow_shorts and prob < 1.0 - self.config.signal_threshold:
                    side = -1
                    edge_prob = 0.5 - float(prob)
                else:
                    self._block("signal_threshold")
                    continue
            candidates.append((abs(edge_prob), symbol, float(prob), side))

        candidates.sort(reverse=True)
        for _, symbol, prob, side in candidates[:slots]:
            price = self._price(symbol, date)
            if price is None or price <= 0:
                self._block("missing_price")
                continue

            mult, mode, _ = self._vol_mode(symbol, date)
            if mode in ("halt", "close") or mult <= 0:
                self._block(f"vol_{mode}")
                continue

            expected_return = (abs(prob - 0.5) / 0.5) * self.config.expected_return_scale_pct
            confidence = max(prob, 1.0 - prob)
            if self.config.fee_gate_enabled:
                decision = self._fee_gate.evaluate(
                    expected_return_pct=expected_return,
                    confidence=confidence,
                    asset_class="equity",
                    realised_one_way_bps=self.config.one_way_cost_bps,
                    half_spread_bps=self.config.half_spread_bps,
                )
                if not decision.allowed:
                    self._block("fee_edge")
                    continue

            base_pct = self.config.base_position_pct
            if base_pct is None:
                base_pct = min(
                    self.config.max_position_pct,
                    1.0 / max(1, self.config.max_positions),
                )
            notional = portfolio_value * base_pct * self.config.kelly_fraction * mult
            notional = min(notional, portfolio_value * self.config.max_position_pct)
            if self.config.hrp_enabled:
                scale = self._portfolio_constructor.get_sizing_scale(
                    symbol=symbol,
                    proposed_notional=notional,
                    portfolio_value=portfolio_value,
                )
                notional *= scale

            if notional <= 0:
                self._block("zero_notional")
                continue
            one_way_cost = self._one_way_cost_pct()
            entry_cost = notional * one_way_cost
            if side < 0:
                self._block("shorts_disabled")
                continue
            if self.cash < notional + entry_cost:
                self._block("cash")
                continue

            shares = notional / price
            self.cash -= notional + entry_cost
            self.positions[symbol] = Position(
                symbol=symbol,
                side=side,
                shares=shares,
                entry_price=price,
                entry_date=date,
                entry_notional=notional,
                entry_cost=entry_cost,
                high_water_mark=price,
                low_water_mark=price,
            )
            self._entries_by_mode[mode] = self._entries_by_mode.get(mode, 0) + 1

    def _process_exits(self, date: pd.Timestamp) -> None:
        for symbol in list(self.positions):
            pos = self.positions[symbol]
            price = self._price(symbol, date)
            if price is None or price <= 0:
                continue
            pos.bars_held += 1
            pos.high_water_mark = max(pos.high_water_mark, price)
            pos.low_water_mark = min(pos.low_water_mark, price)

            mult, mode, _ = self._vol_mode(symbol, date)
            if mode == "close":
                self._exit_position(symbol, date, "vol_close")
                continue

            prob = self._value_at(self.probabilities.get(symbol), date)
            pnl_pct = (price / pos.entry_price - 1.0) * pos.side
            trail_drawdown = price / pos.high_water_mark - 1.0

            reason = None
            if pnl_pct <= -self.config.exits.stop_loss_pct:
                reason = "stop_loss"
            elif pnl_pct >= self.config.exits.take_profit_pct:
                reason = "take_profit"
            elif trail_drawdown <= -self.config.exits.trailing_pct:
                reason = "trailing_stop"
            elif prob is not None and prob < self.config.exits.signal_exit_prob:
                reason = "signal_exit"
            elif pos.bars_held >= self.config.exits.max_hold_bars:
                reason = "time_exit"

            if reason:
                self._exit_position(symbol, date, reason)

    def _exit_position(self, symbol: str, date: pd.Timestamp, reason: str) -> None:
        pos = self.positions.pop(symbol, None)
        if pos is None:
            return
        price = self._price(symbol, date)
        if price is None or price <= 0:
            price = pos.entry_price
        gross = pos.shares * price
        exit_cost = gross * self._one_way_cost_pct()
        proceeds = gross - exit_cost
        self.cash += proceeds
        pnl = proceeds - pos.entry_notional - pos.entry_cost
        denom = max(pos.entry_notional + pos.entry_cost, 1e-9)
        self.trades.append(
            Trade(
                symbol=symbol,
                entry_date=pos.entry_date,
                exit_date=date,
                side="long" if pos.side > 0 else "short",
                entry_price=pos.entry_price,
                exit_price=price,
                shares=pos.shares,
                pnl=float(pnl),
                pnl_pct=float(pnl / denom),
                entry_cost=pos.entry_cost,
                exit_cost=float(exit_cost),
                exit_reason=reason,
                bars_held=pos.bars_held,
            )
        )

    def _update_portfolio_constructor(self, date: pd.Timestamp) -> None:
        if not self.config.hrp_enabled:
            return
        signals = {}
        returns = {}
        for symbol, df in self.price_data.items():
            prob = self._value_at(self.probabilities.get(symbol), date)
            if prob is not None:
                signals[symbol] = (float(prob) - 0.5) * 2.0
            close = df["Close"].loc[:date].tail(self.config.return_lookback + 1)
            if len(close) >= 2:
                returns[symbol] = close.pct_change().dropna().tail(
                    self.config.return_lookback
                ).tolist()
        self._portfolio_constructor.update_signals(signals)
        self._portfolio_constructor.update_returns(returns)
        self._portfolio_constructor.maybe_recompute()

    def _build_result(
        self,
        equity_curve: pd.Series,
        exposure_points: Iterable[float],
    ) -> BacktestResult:
        if equity_curve.empty:
            return self._empty_result()
        daily_returns = equity_curve.pct_change().replace([np.inf, -np.inf], np.nan).dropna()
        sharpe = 0.0
        if len(daily_returns) > 1 and float(daily_returns.std()) > 0:
            sharpe = float(daily_returns.mean() / daily_returns.std() * np.sqrt(252))
        drawdowns = equity_curve / equity_curve.cummax() - 1.0
        wins = [t for t in self.trades if t.pnl > 0]
        losses = [t for t in self.trades if t.pnl < 0]
        gross_win = sum(t.pnl for t in wins)
        gross_loss = abs(sum(t.pnl for t in losses))
        return BacktestResult(
            total_return_pct=float(equity_curve.iloc[-1] / self.config.initial_capital - 1.0),
            sharpe=sharpe,
            max_drawdown_pct=float(drawdowns.min()) if not drawdowns.empty else 0.0,
            trades=len(self.trades),
            win_rate=float(len(wins) / len(self.trades)) if self.trades else 0.0,
            profit_factor=float(gross_win / gross_loss) if gross_loss > 0 else float("inf"),
            exposure_mean=float(np.mean(list(exposure_points))) if exposure_points else 0.0,
            final_equity=float(equity_curve.iloc[-1]),
            equity_curve=equity_curve,
            daily_returns=daily_returns,
            trade_log=list(self.trades),
            diagnostics={
                "blocks": dict(self._block_reasons),
                "entries_by_vol_mode": dict(self._entries_by_mode),
            },
        )

    def _empty_result(self) -> BacktestResult:
        empty = pd.Series(dtype=float)
        return BacktestResult(
            total_return_pct=0.0,
            sharpe=0.0,
            max_drawdown_pct=0.0,
            trades=0,
            win_rate=0.0,
            profit_factor=0.0,
            exposure_mean=0.0,
            final_equity=self.config.initial_capital,
            equity_curve=empty,
            daily_returns=empty,
            trade_log=[],
            diagnostics={"blocks": {}, "entries_by_vol_mode": {}},
        )

    def _common_dates(self) -> pd.DatetimeIndex:
        indexes = []
        for symbol in self.price_data:
            if symbol in self.probabilities:
                indexes.append(self.price_data[symbol].index.intersection(self.probabilities[symbol].index))
        if not indexes:
            return pd.DatetimeIndex([])
        dates = indexes[0]
        for idx in indexes[1:]:
            dates = dates.union(idx)
        return dates.sort_values()

    def _mark_equity(self, date: pd.Timestamp) -> float:
        value = self.cash
        for symbol, pos in self.positions.items():
            price = self._price(symbol, date)
            if price is None:
                price = pos.entry_price
            value += pos.shares * price
        return float(value)

    def _gross_exposure(self, date: pd.Timestamp, equity: float) -> float:
        if equity <= 0:
            return 0.0
        gross = 0.0
        for symbol, pos in self.positions.items():
            price = self._price(symbol, date)
            if price is not None:
                gross += abs(pos.shares * price)
        return float(gross / equity)

    def _vol_mode(self, symbol: str, date: pd.Timestamp) -> tuple[float, str, float]:
        a_fast = self._value_at(self._atr_fast.get(symbol), date) or 0.0
        a_slow = self._value_at(self._atr_slow.get(symbol), date) or 0.0
        return self._vol_circuit.evaluate(a_fast, a_slow)

    def _price(self, symbol: str, date: pd.Timestamp) -> Optional[float]:
        df = self.price_data.get(symbol)
        if df is None or "Close" not in df:
            return None
        return self._value_at(df["Close"], date)

    def _one_way_cost_pct(self) -> float:
        return max(0.0, self.config.one_way_cost_bps + self.config.half_spread_bps) / 10_000.0

    def _block(self, reason: str) -> None:
        self._block_reasons[reason] = self._block_reasons.get(reason, 0) + 1

    @staticmethod
    def _value_at(series: Optional[pd.Series], date: pd.Timestamp) -> Optional[float]:
        if series is None or series.empty:
            return None
        try:
            value = series.loc[date]
        except KeyError:
            prior = series.loc[:date]
            if prior.empty:
                return None
            value = prior.iloc[-1]
        if isinstance(value, pd.Series):
            value = value.iloc[-1]
        value = float(value)
        if not np.isfinite(value):
            return None
        return value

    @staticmethod
    def _atr(df: pd.DataFrame, n: int) -> pd.Series:
        high = df["High"]
        low = df["Low"]
        close = df["Close"]
        tr = pd.concat(
            [high - low, (high - close.shift()).abs(), (low - close.shift()).abs()],
            axis=1,
        ).max(axis=1)
        return tr.rolling(n).mean()
