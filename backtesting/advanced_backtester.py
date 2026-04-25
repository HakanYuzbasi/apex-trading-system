from typing import Dict, Any
"""
backtesting/advanced_backtester.py
PROFESSIONAL BACKTESTING ENGINE
- Proper time-series handling
- Transaction costs
- Slippage modeling
- Realistic order fills
"""

import numpy as np
import pandas as pd
import logging
from typing import List, Optional, Tuple
from datetime import datetime
from collections import defaultdict

from core.symbols import AssetClass, parse_symbol, is_market_open
from config import ApexConfig
from core.logging_config import setup_logging

try:
    from scipy.stats import norm as _scipy_norm
    _norm_cdf = _scipy_norm.cdf
    _norm_ppf = _scipy_norm.ppf
except ImportError:
    import math

    def _norm_cdf(x):
        return 0.5 * (1 + math.erf(x / math.sqrt(2)))

    def _norm_ppf(p):
        # Rational approximation (Abramowitz & Stegun 26.2.23)
        if p <= 0:
            return float('-inf')
        if p >= 1:
            return float('inf')
        if p < 0.5:
            return -_norm_ppf(1 - p)
        t = math.sqrt(-2 * math.log(1 - p))
        c0, c1, c2 = 2.515517, 0.802853, 0.010328
        d1, d2, d3 = 1.432788, 0.189269, 0.001308
        return t - (c0 + c1 * t + c2 * t * t) / (1 + d1 * t + d2 * t * t + d3 * t * t * t)

EULER_MASCHERONI = 0.5772156649

logger = logging.getLogger(__name__)


class AdvancedBacktester:
    """
    Professional-grade backtesting engine.
    
    Features:
    - Walk-forward analysis
    - Transaction costs modeling
    - Realistic slippage
    - Position tracking
    - Portfolio-level statistics
    - Risk-adjusted returns
    """
    
    def __init__(
        self,
        initial_capital: float = 1_000_000,
        commission_per_trade: float = 1.0,
        slippage_bps: float = 5.0,
        fx_commission_bps: Optional[float] = None,
        crypto_commission_bps: Optional[float] = None,
        fx_spread_bps: Optional[float] = None,
        crypto_spread_bps: Optional[float] = None,
        short_borrow_rate: float = 0.005,
        max_adv_participation: float = 0.05,
    ):
        self.initial_capital = initial_capital
        self.commission_per_trade = commission_per_trade
        self.slippage_bps = slippage_bps / 10000  # Convert to decimal
        self.fx_commission_bps = fx_commission_bps if fx_commission_bps is not None else ApexConfig.FX_COMMISSION_BPS
        self.crypto_commission_bps = crypto_commission_bps if crypto_commission_bps is not None else ApexConfig.CRYPTO_COMMISSION_BPS
        self.fx_spread_bps = fx_spread_bps if fx_spread_bps is not None else ApexConfig.FX_SPREAD_BPS
        self.crypto_spread_bps = crypto_spread_bps if crypto_spread_bps is not None else ApexConfig.CRYPTO_SPREAD_BPS
        self.short_borrow_rate = short_borrow_rate
        self.max_adv_participation = max_adv_participation

        self.reset()
        
        logger.info(
            "AdvancedBacktester initialized: capital=$%s, commission=$%.2f, slippage=%.1fbps",
            f"{initial_capital:,.0f}", commission_per_trade, slippage_bps,
        )
    
    def reset(self):
        """Reset backtest state."""
        self.cash = self.initial_capital
        self.positions = {}  # {symbol: quantity}
        self.trades = []
        self.equity_curve = []
        self.daily_returns = []
        self.current_date = None
        self._data_symbols = []
        self._n_sharpe_trials = 1
        self._borrow_costs_total = 0.0
        # ── Round 11: position metadata (entry price, initial risk in $ per
        # share, stop level, partial-exit stage, initial shares) indexed by
        # symbol. A dict is maintained only for symbols whose position is
        # currently non-zero; entries are created by ``_check_entries`` and
        # removed when the position is fully closed.
        self._position_meta: Dict[str, Dict[str, Any]] = {}
        # Correlation/deployment diagnostics (populated once per entry attempt).
        self._portfolio_corr_samples: List[float] = []
        self._deployed_capital_samples: List[float] = []
        # Signal aggregator instance used for Kelly edge stats. Wired by
        # ``run_backtest`` when the signal generator carries an aggregator.
        self._shared_aggregator = None
        # Round 12 diagnostics.
        self._macro_regime_samples: List[int] = []  # 1=RISK_ON, 0=RISK_OFF
        self._entry_block_reasons: Dict[str, int] = {}
        # Round 13 — Reg-T style short margin ledger. Keyed by symbol,
        # records the total USD margin currently posted for the open short.
        self._short_margin_posted: Dict[str, float] = {}
        # Gross-exposure snapshots (sum of |notional| / portfolio_value)
        # captured once per successful entry so the report can surface a
        # gross-deployed metric rather than the net (cash-based) view.
        self._gross_deployed_samples: List[float] = []

    def _annualization_factor(self, symbols: List[str]) -> int:
        classes = set()
        for symbol in symbols:
            try:
                classes.add(parse_symbol(symbol).asset_class)
            except ValueError:
                continue
        if classes == {AssetClass.CRYPTO}:
            return 365
        if classes == {AssetClass.FOREX}:
            return 260
        return 252

    def _is_market_open(self, symbol: str, date: datetime) -> bool:
        try:
            parsed = parse_symbol(symbol)
        except ValueError:
            return False
        return is_market_open(parsed, date, assume_daily=True)

    def _get_slippage_pct(self, asset_class: AssetClass) -> float:
        if asset_class == AssetClass.FOREX:
            return self.fx_spread_bps / 10000.0
        if asset_class == AssetClass.CRYPTO:
            return self.crypto_spread_bps / 10000.0
        return self.slippage_bps

    def _get_prev_date(self, data: Dict[str, pd.DataFrame], symbol: str, date: datetime) -> Optional[datetime]:
        if symbol not in data:
            return None
        idx = data[symbol].index
        if len(idx) == 0:
            return None
        if date in idx:
            pos = idx.get_loc(date)
            if isinstance(pos, slice):
                pos = pos.start
            if pos == 0:
                return None
            return idx[pos - 1]
        prior = idx[idx < date]
        if len(prior) == 0:
            return None
        return prior[-1]

    def _estimate_slippage_pct(
        self,
        symbol: str,
        data: Dict[str, pd.DataFrame],
        date: datetime,
        asset_class: AssetClass,
        quantity: float
    ) -> float:
        base = self._get_slippage_pct(asset_class)
        if symbol not in data or date not in data[symbol].index:
            return base
        hist = data[symbol].loc[:date].tail(20)
        if hist.empty:
            return base
        returns = hist['Close'].pct_change().dropna() if 'Close' in hist.columns else pd.Series(dtype=float)
        vol = returns.std() if len(returns) > 0 else 0.0
        vol_mult = getattr(ApexConfig, "SLIPPAGE_VOL_MULT", 2.0)
        adv_mult = getattr(ApexConfig, "SLIPPAGE_ADV_MULT", 5.0)
        slip = base * (1 + vol * vol_mult)
        if 'Volume' in hist.columns:
            adv = hist['Volume'].mean()
            if adv and adv > 0:
                ratio = abs(quantity) / adv
                slip += base * ratio * adv_mult
        max_bps = getattr(ApexConfig, "BACKTEST_MAX_SLIPPAGE_BPS", 50)
        slip = min(slip, max_bps / 10000.0)
        return slip

    def _validate_data(self, data: Dict[str, pd.DataFrame]) -> Dict[str, List[str]]:
        """Validate price data for survivorship bias and corporate actions.

        Returns dict with 'errors' (NaN/inf/splits → WARNING) and
        'info' (stale prices → INFO).
        """
        errors: List[str] = []
        info: List[str] = []
        stale_symbols: List[str] = []
        for symbol, df in data.items():
            if df.empty:
                errors.append(f"{symbol}: empty DataFrame")
                continue
            cols = [c for c in ['Open', 'High', 'Low', 'Close'] if c in df.columns]
            if df[cols].isnull().any().any():
                errors.append(f"{symbol}: contains NaN prices")
            if np.isinf(df[cols].values).any():
                errors.append(f"{symbol}: contains infinite prices")
            # Likely unadjusted stock splits (overnight return >50%)
            if 'Close' in df.columns and len(df) > 1:
                rets = df['Close'].pct_change(fill_method=None).dropna()
                splits = rets[rets.abs() > 0.5]
                for dt, ret in splits.items():
                    errors.append(
                        f"{symbol} @ {dt.date()}: {ret*100:+.0f}% overnight — likely unadjusted split"
                    )
            # Stale prices (5+ identical closes) — informational, not an error
            if 'Close' in df.columns and len(df) > 5:
                stale_run = (df['Close'].diff().eq(0)).astype(int)
                stale_groups = stale_run.groupby((stale_run != stale_run.shift()).cumsum()).cumsum()
                if (stale_groups >= 5).any():
                    stale_symbols.append(symbol)
        if stale_symbols:
            info.append(
                f"{len(stale_symbols)} symbol(s) have stale-price runs (5+ identical closes): "
                + ", ".join(stale_symbols[:10])
                + ("..." if len(stale_symbols) > 10 else "")
            )
        return {"errors": errors, "info": info}

    def _apply_borrow_costs(self, data: Dict[str, pd.DataFrame], date: datetime):
        """Deduct daily borrow cost for short positions."""
        if self.short_borrow_rate <= 0:
            return
        ann_factor = self._annualization_factor(self._data_symbols)
        for symbol, qty in list(self.positions.items()):
            if qty >= 0:
                continue
            if symbol not in data or date not in data[symbol].index:
                continue
            price = data[symbol].loc[date, 'Close']
            daily_cost = abs(qty) * price * self.short_borrow_rate / ann_factor
            self.cash -= daily_cost
            self._borrow_costs_total += daily_cost

    def _force_close_delisted(self, data: Dict[str, pd.DataFrame], date: datetime, tradeable: set):
        """Force-close positions for symbols that left the point-in-time universe."""
        for symbol in list(self.positions.keys()):
            if self.positions[symbol] == 0:
                continue
            if symbol in tradeable:
                continue
            qty = abs(self.positions[symbol])
            side = 'SELL' if self.positions[symbol] > 0 else 'BUY'
            # Use Open if available today, else last known Close
            if symbol in data and date in data[symbol].index:
                price = data[symbol].loc[date, 'Open']
            elif symbol in data and not data[symbol].empty:
                prior = data[symbol].index[data[symbol].index <= date]
                if len(prior) > 0:
                    price = data[symbol].loc[prior[-1], 'Close']
                else:
                    continue
            else:
                continue
            self._execute_order(symbol, side, qty, price, date, "Universe exit (delisting)")
            logger.warning("DELISTING: Force-closed %s %s @ %.2f", qty, symbol, price)

    def _calculate_commission(
        self,
        symbol: str,
        notional: float,
        is_maker: bool = False,
    ) -> float:
        """
        Compute commission for a simulated fill.

        Routes through :func:`execution.cost_model.fee_bps` so the backtester
        and live connectors share the same taker/maker split. Equities still
        pay the flat ``commission_per_trade`` floor on top of the bps fee to
        match retail brokers' per-ticket minimum.

        Args:
            symbol: Instrument symbol (used for asset-class routing).
            notional: Order notional in USD.
            is_maker: True for passive limit fills, False for aggressive.
                Defaults to False since backtests fill at Open / Close.

        Returns:
            Commission in USD.
        """
        from execution.cost_model import fee_bps as _fee_bps
        try:
            parsed = parse_symbol(symbol)
            asset_key = (
                "equity" if parsed.asset_class == AssetClass.EQUITY
                else "fx" if parsed.asset_class == AssetClass.FOREX
                else "crypto"
            )
        except ValueError:
            asset_key = "equity"

        bps = _fee_bps(asset_key, is_maker=is_maker)
        bps_cost = abs(float(notional)) * (bps / 10_000.0)
        if asset_key == "equity":
            return float(self.commission_per_trade) + bps_cost
        return bps_cost
    
    def run_backtest(
        self,
        data: Dict[str, pd.DataFrame],
        signal_generator,
        start_date: str,
        end_date: str,
        position_size_usd: float = 5000,
        max_positions: int = 15,
        universe_schedule: Optional[Dict[str, List[str]]] = None,
        n_sharpe_trials: int = 1,
    ) -> Dict:
        """
        Run complete backtest.

        Args:
            data: {symbol: DataFrame with OHLCV}
            signal_generator: Signal generator instance
            start_date: Backtest start date
            end_date: Backtest end date
            position_size_usd: Position size in dollars
            max_positions: Maximum concurrent positions
            universe_schedule: Optional point-in-time universe {date_str: [symbols]}.
                On each day, only symbols in the most recent universe entry are tradeable.
                Positions in symbols that leave the universe are force-closed (delisting).
            n_sharpe_trials: Number of strategy variants tested (for deflated Sharpe).

        Returns:
            Backtest results with metrics
        """
        logger.info("Running backtest: %s to %s", start_date, end_date)

        self.reset()
        self._data_symbols = list(data.keys())
        self._n_sharpe_trials = n_sharpe_trials
        # Round 11: pick up the signal generator's aggregator so Kelly sizing
        # can read per-source edge stats populated at exit time.
        self._shared_aggregator = getattr(signal_generator, "_aggregator", None)

        # Validate data for common issues (splits, stale prices, NaN)
        validation = self._validate_data(data)
        for w in validation["errors"]:
            logger.warning("DATA: %s", w)
        for w in validation["info"]:
            logger.info("DATA: %s", w)

        # GAP-7A: survivorship bias guard. Without a point-in-time universe
        # the backtester blindly trades whatever symbols are in `data`, which
        # typically reflects a *current* universe (survivors). Log once so
        # the run is not silently over-optimistic.
        if not universe_schedule:
            logger.warning(
                "SURVIVORSHIP: run_backtest called without universe_schedule — "
                "results may overstate returns because delisted/replaced symbols "
                "are not represented. Pass universe_schedule={date: [symbols]} "
                "to enforce a point-in-time universe."
            )

        # Pre-process universe schedule into sorted list for fast lookup
        universe_timeline = None
        if universe_schedule:
            universe_timeline = sorted(
                (pd.Timestamp(k), set(v)) for k, v in universe_schedule.items()
            )

        # Get date range
        dates = pd.date_range(start_date, end_date, freq='D')

        # GAP-7C: indicator warm-up. Skip the first MAX_INDICATOR_LOOKBACK
        # bars so no signal is generated on a data tail shorter than the
        # deepest indicator window. Portfolio bookkeeping still runs so
        # equity is continuous across the warm-up window.
        warmup_bars = int(getattr(ApexConfig, "MAX_INDICATOR_LOOKBACK", 0))
        if warmup_bars > 0 and warmup_bars < len(dates):
            logger.info(
                "Warm-up: suppressing signals for the first %d bars", warmup_bars,
            )

        # Track metrics
        peak_equity = self.initial_capital

        for idx, date in enumerate(dates):
            self.current_date = date
            in_warmup = idx < warmup_bars

            # Resolve current point-in-time universe
            tradeable = None
            if universe_timeline:
                for ud, symbols in reversed(universe_timeline):
                    if ud <= date:
                        tradeable = symbols
                        break

            # Deduct daily borrow costs for short positions
            self._apply_borrow_costs(data, date)

            # Update portfolio value
            portfolio_value = self._calculate_portfolio_value(data, date)

            # Record equity
            self.equity_curve.append({
                'date': date,
                'equity': portfolio_value,
                'cash': self.cash,
                'positions_value': portfolio_value - self.cash
            })

            # Calculate daily return
            if len(self.equity_curve) > 1:
                prev_equity = self.equity_curve[-2]['equity']
                daily_return = (portfolio_value / prev_equity - 1) if prev_equity > 0 else 0
                self.daily_returns.append(daily_return)

            # Update peak
            if portfolio_value > peak_equity:
                peak_equity = portfolio_value

            # Generate signals and execute trades. During warm-up we still
            # track portfolio value but skip signal generation to prevent
            # indicator look-ahead bias on undersized windows.
            if not in_warmup:
                self._process_trading_day(
                    data,
                    date,
                    signal_generator,
                    position_size_usd,
                    max_positions,
                    portfolio_value,
                    tradeable
                )
        
        # Calculate final metrics
        results = self._calculate_metrics()
        
        logger.info(
            "BACKTEST RESULTS | %s to %s | Return=%.2f%% | Sharpe=%.2f | "
            "MaxDD=%.2f%% | WinRate=%.1f%% | Trades=%d | Final=$%s",
            start_date, end_date,
            results['total_return'] * 100,
            results['sharpe_ratio'],
            results['max_drawdown'] * 100,
            results['win_rate'] * 100,
            results['total_trades'],
            f"{results['final_value']:,.0f}",
        )
        
        return results
    
    def run_walk_forward(
        self,
        data: Dict[str, pd.DataFrame],
        signal_generator,
        start_date: str,
        end_date: str,
        position_size_usd: float = 5000,
        max_positions: int = 15,
        universe_schedule: Optional[Dict[str, List[str]]] = None,
        is_bars: Optional[int] = None,
        oos_bars: Optional[int] = None,
        step_bars: Optional[int] = None,
    ) -> Dict:
        """
        Run an anchored walk-forward backtest and aggregate per-fold OOS metrics.

        At each fold the IS window ends, the OOS window begins, and signals
        inside the OOS window are scored on IS-unseen bars. The backtester
        itself is stateless across folds — each fold spins up a fresh
        equity curve so the OOS metrics are not contaminated by IS PnL.

        Args:
            data: ``{symbol: DataFrame with OHLCV}``. Must cover the full
                ``start_date..end_date`` range.
            signal_generator: Signal generator instance passed through to
                :meth:`run_backtest`.
            start_date: Inclusive start of the walk-forward span.
            end_date: Inclusive end of the walk-forward span.
            position_size_usd: Position size passed to each fold.
            max_positions: Cap on concurrent positions per fold.
            universe_schedule: Optional point-in-time universe; applied to
                each OOS fold identically.
            is_bars: In-sample bars per fold. Fallback: ``ApexConfig.WF_IS_BARS``.
            oos_bars: Out-of-sample bars per fold. Fallback: ``ApexConfig.WF_OOS_BARS``.
            step_bars: Step size between fold starts. Fallback: ``ApexConfig.WF_STEP_BARS``.

        Returns:
            Dict with keys:
                ``folds``: list of per-fold result dicts
                    (``{"oos_start", "oos_end", "sharpe_ratio", "total_return",
                    "max_drawdown", "win_rate", "profit_factor", "total_trades"}``)
                ``aggregate``: aggregated OOS metrics
                    (``mean_sharpe``, ``median_sharpe``, ``compounded_return``,
                    ``worst_fold_drawdown``, ``folds_run``, ``positive_folds``).
        """
        is_n = int(is_bars if is_bars is not None else ApexConfig.WF_IS_BARS)
        oos_n = int(oos_bars if oos_bars is not None else ApexConfig.WF_OOS_BARS)
        step_n = int(step_bars if step_bars is not None else ApexConfig.WF_STEP_BARS)
        if min(is_n, oos_n, step_n) < 1:
            raise ValueError(
                f"walk-forward requires positive is/oos/step bar counts; "
                f"got is={is_n} oos={oos_n} step={step_n}"
            )

        full_dates = pd.date_range(start_date, end_date, freq='D')
        if len(full_dates) < is_n + oos_n:
            raise ValueError(
                f"walk-forward range {start_date}..{end_date} is too short "
                f"({len(full_dates)} bars) for is={is_n} + oos={oos_n}"
            )

        fold_results: List[Dict] = []
        fold_start = 0
        while fold_start + is_n + oos_n <= len(full_dates):
            oos_from = full_dates[fold_start + is_n]
            oos_to = full_dates[min(fold_start + is_n + oos_n - 1, len(full_dates) - 1)]
            logger.info(
                "WF fold #%d — OOS %s → %s",
                len(fold_results) + 1,
                oos_from.date(),
                oos_to.date(),
            )
            fold_metrics = self.run_backtest(
                data=data,
                signal_generator=signal_generator,
                start_date=str(oos_from.date()),
                end_date=str(oos_to.date()),
                position_size_usd=position_size_usd,
                max_positions=max_positions,
                universe_schedule=universe_schedule,
                n_sharpe_trials=1,
            )
            # NaN-safe fold record: preserve NaN Sharpe so the aggregate can
            # exclude insufficient-data folds from mean/median instead of
            # silently coercing them to 0.
            raw_sharpe = fold_metrics.get("sharpe_ratio")
            try:
                sharpe_val = float(raw_sharpe) if raw_sharpe is not None else float("nan")
            except (TypeError, ValueError):
                sharpe_val = float("nan")
            fold_results.append({
                "oos_start": str(oos_from.date()),
                "oos_end": str(oos_to.date()),
                "sharpe_ratio": sharpe_val,
                "total_return": float(fold_metrics.get("total_return", 0.0) or 0.0),
                "max_drawdown": float(fold_metrics.get("max_drawdown", 0.0) or 0.0),
                "win_rate": float(fold_metrics.get("win_rate", 0.0) or 0.0),
                "profit_factor": float(fold_metrics.get("profit_factor", 0.0) or 0.0),
                "total_trades": int(fold_metrics.get("total_trades", 0) or 0),
            })
            fold_start += step_n

        if not fold_results:
            return {"folds": [], "aggregate": {"folds_run": 0}}

        sharpes_all = np.array(
            [f["sharpe_ratio"] for f in fold_results], dtype=float
        )
        valid_mask = np.isfinite(sharpes_all)
        valid_sharpes = sharpes_all[valid_mask]
        insufficient_data_folds = int((~valid_mask).sum())

        compounded = 1.0
        for f in fold_results:
            compounded *= 1.0 + f["total_return"]
        compounded -= 1.0
        worst_dd = min((f["max_drawdown"] for f in fold_results), default=0.0)
        positive = sum(1 for f in fold_results if f["total_return"] > 0)
        negative = sum(1 for f in fold_results if f["total_return"] < 0)

        return {
            "folds": fold_results,
            "aggregate": {
                "folds_run": len(fold_results),
                "mean_sharpe": (
                    float(valid_sharpes.mean()) if valid_sharpes.size else float("nan")
                ),
                "median_sharpe": (
                    float(np.median(valid_sharpes))
                    if valid_sharpes.size else float("nan")
                ),
                "compounded_return": float(compounded),
                "worst_fold_drawdown": float(worst_dd),
                "positive_folds": int(positive),
                "negative_folds": int(negative),
                "insufficient_data_folds": insufficient_data_folds,
            },
        }

    # ─────────────────────────────────────────────────────────────────────
    # Round 11 helpers — ATR, Kelly sizing, correlation gating
    # ─────────────────────────────────────────────────────────────────────

    def _compute_atr(
        self,
        data: Dict[str, pd.DataFrame],
        symbol: str,
        date: datetime,
        period: int = 14,
    ) -> float:
        """Wilder ATR(14) computed from bars strictly before ``date``."""
        if symbol not in data:
            return 0.0
        hist = data[symbol].loc[:date].iloc[:-1]  # exclude current bar
        if len(hist) < period + 1:
            return 0.0
        high = hist["High"].astype(float)
        low = hist["Low"].astype(float)
        close = hist["Close"].astype(float)
        prev_close = close.shift(1)
        tr = pd.concat(
            [(high - low).abs(), (high - prev_close).abs(), (low - prev_close).abs()],
            axis=1,
        ).max(axis=1)
        atr = tr.ewm(alpha=1.0 / period, adjust=False).mean()
        val = float(atr.iloc[-1]) if len(atr) else 0.0
        return val if np.isfinite(val) and val > 0.0 else 0.0

    def _portfolio_max_correlation(
        self,
        data: Dict[str, pd.DataFrame],
        candidate: str,
        date: datetime,
        lookback: int,
    ) -> float:
        """
        Return the maximum rolling-return correlation between ``candidate``
        and any currently-open position.

        Args:
            data: Panel.
            candidate: Symbol about to be opened.
            date: Decision timestamp.
            lookback: Rolling window (bars).

        Returns:
            Maximum absolute pairwise correlation in ``[0, 1]``. When there
            are no open positions or the lookback has too few overlapping
            returns the function returns ``0.0`` — i.e. no correlation
            penalty.
        """
        open_symbols = [
            s for s, q in self.positions.items()
            if q != 0 and s != candidate and s in data
        ]
        if not open_symbols or candidate not in data:
            return 0.0
        cand_hist = data[candidate].loc[:date]
        if len(cand_hist) < lookback + 1:
            return 0.0
        cand_ret = cand_hist["Close"].pct_change().dropna().tail(lookback)
        if len(cand_ret) < max(5, lookback // 4):
            return 0.0

        max_corr = 0.0
        for sym in open_symbols:
            sym_hist = data[sym].loc[:date]
            if len(sym_hist) < lookback + 1:
                continue
            sym_ret = sym_hist["Close"].pct_change().dropna().tail(lookback)
            aligned = pd.concat([cand_ret, sym_ret], axis=1, join="inner").dropna()
            if len(aligned) < max(5, lookback // 4):
                continue
            a = aligned.iloc[:, 0].to_numpy()
            b = aligned.iloc[:, 1].to_numpy()
            if np.std(a) <= 0.0 or np.std(b) <= 0.0:
                continue
            rho = float(np.corrcoef(a, b)[0, 1])
            if np.isfinite(rho) and abs(rho) > max_corr:
                max_corr = abs(rho)
        return max_corr

    def _gross_exposure(
        self,
        data: Dict[str, pd.DataFrame],
        date: datetime,
        *,
        additional_notional: float = 0.0,
    ) -> float:
        """
        Round 13 FIX 1 — total gross USD exposure across all open positions.

        ``Σ |quantity × close_price|`` using each symbol's most recent close
        at ``date``. ``additional_notional`` adds a hypothetical pending
        order (in absolute USD) so the caller can check the leverage cap
        BEFORE admitting a new entry. Falls back to the position's entry
        price when the symbol is absent from ``data`` (defensive — normal
        production flow always has data).
        """
        total = abs(float(additional_notional or 0.0))
        for sym, qty in self.positions.items():
            if qty == 0:
                continue
            price: Optional[float] = None
            if sym in data:
                hist = data[sym].loc[:date]
                if not hist.empty and "Close" in hist.columns:
                    val = float(hist["Close"].iloc[-1])
                    if np.isfinite(val) and val > 0.0:
                        price = val
            if price is None:
                meta = self._position_meta.get(sym, {})
                price = float(meta.get("entry_price", 0.0) or 0.0)
            total += abs(float(qty)) * float(price)
        return float(total)

    def _count_correlated_pairs(
        self,
        data: Dict[str, pd.DataFrame],
        date: datetime,
        lookback: int,
        threshold: float,
    ) -> int:
        """
        Count pairs of currently-open positions whose pairwise rolling
        correlation exceeds ``threshold``.
        """
        open_symbols = [
            s for s, q in self.positions.items()
            if q != 0 and s in data
        ]
        if len(open_symbols) < 2:
            return 0
        count = 0
        for i in range(len(open_symbols)):
            for j in range(i + 1, len(open_symbols)):
                a_hist = data[open_symbols[i]].loc[:date]
                b_hist = data[open_symbols[j]].loc[:date]
                if len(a_hist) < lookback + 1 or len(b_hist) < lookback + 1:
                    continue
                a_ret = a_hist["Close"].pct_change().dropna().tail(lookback)
                b_ret = b_hist["Close"].pct_change().dropna().tail(lookback)
                aligned = pd.concat([a_ret, b_ret], axis=1, join="inner").dropna()
                if len(aligned) < max(5, lookback // 4):
                    continue
                x = aligned.iloc[:, 0].to_numpy()
                y = aligned.iloc[:, 1].to_numpy()
                if np.std(x) <= 0.0 or np.std(y) <= 0.0:
                    continue
                rho = float(np.corrcoef(x, y)[0, 1])
                if np.isfinite(rho) and abs(rho) > threshold:
                    count += 1
        return count

    def _macro_regime(
        self,
        data: Dict[str, pd.DataFrame],
        date: datetime,
    ) -> str:
        """
        Round 12 FIX 4 — portfolio-level RISK_ON / RISK_OFF classifier
        using daily SPY bars.

        Returns ``"RISK_ON"`` when:
          * SPY ``N``-bar return > 0 (``N`` = ``MACRO_REGIME_RETURN_LOOKBACK``)
          * SPY ATR(14) / close < ``MACRO_REGIME_VOL_MAX``
        Returns ``"RISK_OFF"`` otherwise, or ``"RISK_ON"`` as a safe
        default when SPY data is missing (so the gate fails open rather
        than freezing all new entries on a data gap).
        """
        if "SPY" not in data:
            return "RISK_ON"
        spy = data["SPY"].loc[:date]
        lookback = int(getattr(ApexConfig, "MACRO_REGIME_RETURN_LOOKBACK", 20))
        vol_max = float(getattr(ApexConfig, "MACRO_REGIME_VOL_MAX", 0.015))
        if len(spy) < max(lookback + 1, 15):
            return "RISK_ON"
        close = spy["Close"].astype(float)
        if close.iloc[-1] <= 0:
            return "RISK_ON"
        ret = float(close.iloc[-1] / close.iloc[-lookback - 1] - 1.0)
        high = spy["High"].astype(float)
        low = spy["Low"].astype(float)
        prev_close = close.shift(1)
        tr = pd.concat(
            [(high - low).abs(), (high - prev_close).abs(), (low - prev_close).abs()],
            axis=1,
        ).max(axis=1)
        atr = tr.ewm(alpha=1.0 / 14, adjust=False).mean()
        atr_pct = float(atr.iloc[-1] / close.iloc[-1]) if close.iloc[-1] else 0.0
        if ret > 0.0 and np.isfinite(atr_pct) and atr_pct < vol_max:
            return "RISK_ON"
        return "RISK_OFF"

    def _kelly_notional(
        self,
        source: str,
        portfolio_value: float,
    ) -> Optional[float]:
        """
        Compute the half-Kelly notional for ``source`` using the shared
        :class:`~signals.signal_aggregator.SignalAggregator`'s rolling
        realised-PnL buffer.

        Returns:
            The Kelly-sized notional in USD after applying
            ``ApexConfig.KELLY_FRACTION_R11`` and the
            ``MIN_POSITION_USD`` / ``MAX_POSITION_PCT`` clamps. Returns
            ``None`` when Kelly sizing is disabled, no aggregator is
            wired, the source has fewer than ``KELLY_MIN_SAMPLES``
            observations, or the computed edge is non-positive. The
            caller must then fall back to the ATR / quality path.
        """
        if not getattr(ApexConfig, "KELLY_ENABLED", False):
            return None
        aggregator = getattr(self, "_shared_aggregator", None)
        if aggregator is None:
            return None
        stats = aggregator.get_source_edge_stats(source)
        n = int(stats.get("n_samples", 0))
        if n < int(ApexConfig.KELLY_MIN_SAMPLES):
            return None
        win_rate = float(stats.get("win_rate", 0.0))
        avg_win = float(stats.get("avg_win", 0.0))
        avg_loss = float(stats.get("avg_loss", 0.0))
        if avg_win <= 0.0:
            return None
        kelly_f = win_rate - ((1.0 - win_rate) * avg_loss) / avg_win
        if kelly_f <= 0.0:
            return None
        kelly_f = kelly_f * float(ApexConfig.KELLY_FRACTION_R11)
        raw_notional = kelly_f * float(portfolio_value)
        min_usd = float(ApexConfig.MIN_POSITION_USD)
        max_pct = float(ApexConfig.MAX_POSITION_PCT)
        max_usd = max_pct * float(portfolio_value)
        notional = max(min_usd, min(raw_notional, max_usd))
        return float(notional)

    # ─────────────────────────────────────────────────────────────────────

    def _process_trading_day(
        self,
        data: Dict[str, pd.DataFrame],
        date: datetime,
        signal_generator,
        position_size_usd: float,
        max_positions: int,
        portfolio_value: float,
        tradeable: Optional[set] = None
    ):
        """Process one trading day."""

        # Force-close positions for symbols that left the universe (delisting)
        if tradeable is not None:
            self._force_close_delisted(data, date, tradeable)

        # Exit positions first
        self._check_exits(data, date, signal_generator)

        # Enter new positions
        self._check_entries(
            data,
            date,
            signal_generator,
            position_size_usd,
            max_positions,
            portfolio_value,
            tradeable
        )
    
    def _check_exits(
        self,
        data: Dict[str, pd.DataFrame],
        date: datetime,
        signal_generator
    ):
        """
        Exit logic with Round 11 partial-exit state machine.

        Each open position carries metadata
        (entry_price, side, initial_shares, risk_per_share, stop_price,
        stage, atr_at_entry). On every bar:

          Stage 0 (just entered):
            * if close <= stop_price (long) / >= stop_price (short)
              → close FULL position (stop-out).
            * if close crosses +1R → close ``PARTIAL_EXIT_R1`` fraction,
              move stop to break-even, advance to stage 1.
            * bearish/bullish opposing signal > 0.30 closes full position.

          Stage 1 (+1R hit, stop at BE):
            * close <= stop_price → close remainder at break-even.
            * close crosses +2R → close ``PARTIAL_EXIT_R2`` fraction,
              trail stop at 1R below current close, advance to stage 2.

          Stage 2 (+2R hit, trailing):
            * stop trails at ``PARTIAL_EXIT_ATR_MULT * ATR(14)`` below
              current close (long) / above (short).
            * stop-out closes remainder.
        """
        partial_enabled = bool(getattr(ApexConfig, "PARTIAL_EXIT_ENABLED", False))
        pe_r1 = float(getattr(ApexConfig, "PARTIAL_EXIT_R1", 0.33))
        pe_r2 = float(getattr(ApexConfig, "PARTIAL_EXIT_R2", 0.33))
        pe_atr_mult = float(getattr(ApexConfig, "PARTIAL_EXIT_ATR_MULT", 2.0))

        for symbol in list(self.positions.keys()):
            qty = self.positions.get(symbol, 0)
            if qty == 0:
                continue
            if not self._is_market_open(symbol, date):
                continue
            prev_date = self._get_prev_date(data, symbol, date)
            if not prev_date:
                continue
            if symbol not in data or date not in data[symbol].index:
                continue

            current_bar = data[symbol].loc[date]
            price = float(current_bar["Open"])
            close_price = float(current_bar["Close"])

            try:
                prices = data[symbol].loc[:prev_date, "Close"]
                signal_data = signal_generator.generate_ml_signal(symbol, prices)
                signal = float(signal_data.get("signal", 0.0))
            except Exception:
                signal = 0.0

            meta = self._position_meta.get(symbol)

            # Legacy reversal exit — kept for safety and back-compat with
            # positions opened before the partial-exit metadata existed.
            if meta is None:
                if qty > 0 and signal < -0.30:
                    self._close_position(symbol, data, date, price, "Bearish signal")
                elif qty < 0 and signal > 0.30:
                    self._close_position(symbol, data, date, price, "Bullish signal")
                continue

            side = meta["side"]  # +1 long / -1 short
            entry = float(meta["entry_price"])
            r_per_share = float(meta["risk_per_share"])
            stop = float(meta["stop_price"])
            stage = int(meta["stage"])
            initial_shares = float(meta["initial_shares"])

            # R-multiple of the current bar (positive when in-profit).
            if r_per_share <= 0.0:
                r_mult = 0.0
            else:
                r_mult = ((price - entry) * side) / r_per_share

            # Partial-exit / stop / trail checks ---------------------------------
            should_full_exit = False
            full_exit_reason = ""

            # Stop-out using bar's low (long) or high (short) to catch
            # intra-bar touches; fall back to open if unavailable.
            low_price = float(current_bar.get("Low", price))
            high_price = float(current_bar.get("High", price))
            stopped_out = (
                (side > 0 and low_price <= stop)
                or (side < 0 and high_price >= stop)
            )

            if stopped_out:
                should_full_exit = True
                full_exit_reason = f"Stop hit @R={r_mult:.2f} (stage {stage})"
            elif (side > 0 and signal < -0.30) or (side < 0 and signal > 0.30):
                should_full_exit = True
                full_exit_reason = "Opposing signal"

            if should_full_exit:
                # Record R-multiple at the fill price for reporting.
                fill_price = stop if stopped_out else price
                realised_r = (
                    ((fill_price - entry) * side) / r_per_share
                    if r_per_share > 0.0 else 0.0
                )
                meta["realised_r"] = float(realised_r)
                self._close_position(
                    symbol, data, date, fill_price, full_exit_reason, realised_r=realised_r,
                )
                continue

            if not partial_enabled:
                continue

            # Stage 0 → 1: +1R partial + move stop to break-even.
            if stage == 0 and r_mult >= 1.0:
                partial_qty = abs(self._quantize_shares(symbol, initial_shares * pe_r1))
                if partial_qty > 0 and partial_qty < abs(qty):
                    partial_r = 1.0
                    self._execute_partial_exit(
                        symbol, data, date, price, partial_qty,
                        reason="Partial +1R exit", realised_r=partial_r,
                    )
                    meta["stop_price"] = entry  # break-even
                    meta["stage"] = 1
                continue

            # Stage 1 → 2: +2R partial + trail stop at 1R below close.
            if stage == 1 and r_mult >= 2.0:
                partial_qty = abs(self._quantize_shares(symbol, initial_shares * pe_r2))
                if partial_qty > 0 and partial_qty < abs(qty):
                    partial_r = 2.0
                    self._execute_partial_exit(
                        symbol, data, date, price, partial_qty,
                        reason="Partial +2R exit", realised_r=partial_r,
                    )
                    # Trail stop at 1R below current close (long) / above (short).
                    new_stop = close_price - side * r_per_share
                    meta["stop_price"] = float(new_stop)
                    meta["stage"] = 2
                continue

            # Stage 2: ATR-multiple trailing stop.
            if stage == 2:
                atr_now = self._compute_atr(data, symbol, date)
                if atr_now > 0.0:
                    trail_stop = close_price - side * pe_atr_mult * atr_now
                    # Only ratchet stop in the favourable direction.
                    if side > 0 and trail_stop > meta["stop_price"]:
                        meta["stop_price"] = float(trail_stop)
                    elif side < 0 and trail_stop < meta["stop_price"]:
                        meta["stop_price"] = float(trail_stop)
                continue

    def _quantize_shares(self, symbol: str, shares: float) -> float:
        """Round ``shares`` per the symbol's asset class (equities = int)."""
        try:
            ac = parse_symbol(symbol).asset_class
        except ValueError:
            ac = AssetClass.EQUITY
        if ac == AssetClass.EQUITY:
            return float(int(shares))
        return float(shares)

    def _execute_partial_exit(
        self,
        symbol: str,
        data: Dict[str, pd.DataFrame],
        date: datetime,
        price: float,
        exit_qty: float,
        *,
        reason: str,
        realised_r: float,
    ) -> None:
        """Close ``exit_qty`` of ``symbol`` without discarding position meta."""
        if exit_qty <= 0.0:
            return
        pos_qty = self.positions.get(symbol, 0)
        if pos_qty == 0:
            return
        side = "SELL" if pos_qty > 0 else "BUY"
        try:
            ac = parse_symbol(symbol).asset_class
        except ValueError:
            ac = AssetClass.EQUITY
        slip = self._estimate_slippage_pct(symbol, data, date, ac, exit_qty)
        ok = self._execute_order(
            symbol, side, exit_qty, price, date,
            f"{reason} @R={realised_r:.2f}",
            slippage_pct=slip,
            data=data,
        )
        if not ok:
            return
        # Annotate the most recent trade record with R-multiple for reporting.
        if self.trades:
            self.trades[-1]["partial"] = True
            self.trades[-1]["realised_r"] = float(realised_r)

    def _close_position(
        self,
        symbol: str,
        data: Dict[str, pd.DataFrame],
        date: datetime,
        price: float,
        reason: str,
        *,
        realised_r: Optional[float] = None,
    ) -> None:
        """Close the full remaining quantity of ``symbol`` and drop its meta."""
        pos_qty = self.positions.get(symbol, 0)
        if pos_qty == 0:
            return
        exit_qty = abs(pos_qty)
        side = "SELL" if pos_qty > 0 else "BUY"
        try:
            ac = parse_symbol(symbol).asset_class
        except ValueError:
            ac = AssetClass.EQUITY
        slip = self._estimate_slippage_pct(symbol, data, date, ac, exit_qty)
        ok = self._execute_order(
            symbol, side, exit_qty, price, date, reason,
            slippage_pct=slip,
            data=data,
        )
        if not ok:
            return
        if realised_r is not None and self.trades:
            self.trades[-1]["realised_r"] = float(realised_r)
        # Feed realised PnL back into the shared aggregator so Kelly
        # sizing learns from closed outcomes.
        meta = self._position_meta.pop(symbol, None)
        if meta is not None:
            self._feed_source_outcome(meta, price)

    def _feed_source_outcome(
        self, meta: Dict[str, Any], exit_price: float,
    ) -> None:
        """Push a realised-PnL% sample to the shared aggregator."""
        aggregator = getattr(self, "_shared_aggregator", None)
        if aggregator is None:
            return
        entry = float(meta.get("entry_price", 0.0) or 0.0)
        if entry <= 0.0:
            return
        side = int(meta.get("side", 1) or 1)
        pnl_pct = (exit_price - entry) / entry * side
        source_label = str(meta.get("source", "primary") or "primary")
        try:
            aggregator.record_source_outcome(source_label, float(pnl_pct))
        except Exception as exc:
            logger.debug("Aggregator record_source_outcome failed: %s", exc)
    
    def _check_entries(
        self,
        data: Dict[str, pd.DataFrame],
        date: datetime,
        signal_generator,
        position_size_usd: float,
        max_positions: int,
        portfolio_value: float,
        tradeable: Optional[set] = None
    ):
        """Check entry conditions for new positions."""

        # Count current positions
        current_positions = sum(1 for qty in self.positions.values() if qty != 0)

        if current_positions >= max_positions:
            return

        # Check each symbol
        for symbol in data.keys():
            # Only consider symbols in point-in-time universe
            if tradeable is not None and symbol not in tradeable:
                continue

            if date not in data[symbol].index:
                continue

            if not self._is_market_open(symbol, date):
                continue

            prev_date = self._get_prev_date(data, symbol, date)
            if not prev_date:
                continue
            
            # Skip if already have position
            if symbol in self.positions and self.positions[symbol] != 0:
                continue
            
            current_bar = data[symbol].loc[date]
            price = current_bar['Open']

            # Generate signal from data up to prev day (no lookahead)
            try:
                prices = data[symbol].loc[:prev_date, 'Close']
                signal_data = signal_generator.generate_ml_signal(symbol, prices)
                signal = signal_data.get('signal', 0.0)
                confidence = signal_data.get('confidence', 0.5)
                quality = signal_data.get('quality', confidence)
            except:
                self._entry_block_reasons["signal_error"] = (
                    self._entry_block_reasons.get("signal_error", 0) + 1
                )
                continue

            # Entry logic
            # Dynamic quality gating for better Sharpe
            min_conf = 0.35
            if confidence < min_conf or quality < 0.35:
                self._entry_block_reasons["confidence_gate"] = (
                    self._entry_block_reasons.get("confidence_gate", 0) + 1
                )
                continue

            dynamic_threshold = 0.45 + (0.10 if confidence < 0.55 else 0.0)
            if abs(signal) <= dynamic_threshold:
                self._entry_block_reasons["dynamic_threshold"] = (
                    self._entry_block_reasons.get("dynamic_threshold", 0) + 1
                )
                continue

            try:
                asset_class = parse_symbol(symbol).asset_class
            except ValueError:
                self._entry_block_reasons["invalid_symbol"] = (
                    self._entry_block_reasons.get("invalid_symbol", 0) + 1
                )
                continue

            # ── Round 12 FIX 4: Macro regime gate ─────────────────────────
            macro_enabled = bool(getattr(ApexConfig, "MACRO_REGIME_ENABLED", False))
            risk_off_mult = float(getattr(ApexConfig, "RISK_OFF_SIZE_MULT", 0.5))
            high_beta = set(getattr(ApexConfig, "HIGH_BETA_SYMBOLS", []) or [])
            regime_multiplier = 1.0
            if macro_enabled:
                regime = self._macro_regime(data, date)
                self._macro_regime_samples.append(1 if regime == "RISK_ON" else 0)
                if regime == "RISK_OFF":
                    # Block new long entries on high-beta symbols.
                    if signal > 0 and symbol in high_beta:
                        self._entry_block_reasons["risk_off_highbeta"] = (
                            self._entry_block_reasons.get("risk_off_highbeta", 0) + 1
                        )
                        logger.debug(
                            "event=entry_blocked symbol=%s reason=risk_off_highbeta",
                            symbol,
                        )
                        continue
                    regime_multiplier = risk_off_mult

            # ── FIX 3: correlation-adjusted concurrent sizing ─────────────
            corr_threshold = float(getattr(ApexConfig, "CORR_THRESHOLD", 0.70))
            corr_lookback = int(getattr(ApexConfig, "CORR_LOOKBACK_BARS", 60))
            # Hard block: 3+ already-open positions with pairwise corr above threshold.
            corr_blocked = (
                self._count_correlated_pairs(
                    data, date, corr_lookback, corr_threshold,
                ) >= 3
            )
            if corr_blocked:
                self._entry_block_reasons["corr_blocked"] = (
                    self._entry_block_reasons.get("corr_blocked", 0) + 1
                )
                logger.debug(
                    "event=entry_blocked symbol=%s reason=portfolio_correlated_cluster",
                    symbol,
                )
                continue
            max_corr = self._portfolio_max_correlation(
                data, symbol, date, corr_lookback,
            )
            self._portfolio_corr_samples.append(float(max_corr))
            corr_size_multiplier = 1.0
            if max_corr > corr_threshold:
                corr_size_multiplier = max(0.0, 1.0 - max_corr)
                if corr_size_multiplier <= 0.0:
                    self._entry_block_reasons["corr_size_zeroed"] = (
                        self._entry_block_reasons.get("corr_size_zeroed", 0) + 1
                    )
                    logger.debug(
                        "event=entry_blocked symbol=%s reason=corr_size_zeroed corr=%.2f",
                        symbol, max_corr,
                    )
                    continue

            # ── FIX 1: Kelly-derived notional, quality-scaled fallback ────
            source_label = "primary"
            if hasattr(signal_generator, "_source_hits"):
                # RealSignalAdapter exposes its per-source tally;
                # Kelly edge stats key off the aggregator's source labels
                # (we use ``primary`` for the ML/momentum primary signal).
                source_label = "primary"
            kelly_notional = self._kelly_notional(source_label, portfolio_value)
            if kelly_notional is not None:
                base_notional = kelly_notional
                sizing_mode = "kelly"
            else:
                # Fallback: caller-supplied position_size_usd scaled by quality.
                size_mult = 0.5 + 0.5 * float(np.clip(quality, 0.0, 1.0))
                base_notional = float(position_size_usd) * size_mult
                sizing_mode = "atr_quality"

            base_notional *= corr_size_multiplier * regime_multiplier
            # Enforce absolute floors/ceilings.
            min_usd = float(getattr(ApexConfig, "MIN_POSITION_USD", 500.0))
            max_pct = float(getattr(ApexConfig, "MAX_POSITION_PCT", 0.15))
            max_usd = max_pct * float(portfolio_value)
            if base_notional < min_usd:
                base_notional = min_usd
            if base_notional > max_usd:
                base_notional = max_usd

            if asset_class == AssetClass.EQUITY:
                shares = int(base_notional / price) if price > 0 else 0
                shares = min(shares, 200)  # Keep legacy max-share guard
            else:
                shares = (base_notional / price) if price > 0 else 0.0

            if asset_class == AssetClass.EQUITY and shares < 1:
                self._entry_block_reasons["share_size_zero"] = (
                    self._entry_block_reasons.get("share_size_zero", 0) + 1
                )
                continue
            if asset_class != AssetClass.EQUITY and shares < 0.0001:
                self._entry_block_reasons["share_size_zero"] = (
                    self._entry_block_reasons.get("share_size_zero", 0) + 1
                )
                continue

            # ── Round 13 FIX 1: portfolio gross-leverage cap ──────────────
            leverage_cap = float(
                getattr(ApexConfig, "PORTFOLIO_GROSS_LEVERAGE_MAX", 1.5)
            )
            candidate_notional = float(shares) * float(price)
            gross_after = self._gross_exposure(
                data, date, additional_notional=candidate_notional,
            )
            if leverage_cap > 0.0 and gross_after > portfolio_value * leverage_cap:
                self._entry_block_reasons["leverage_cap"] = (
                    self._entry_block_reasons.get("leverage_cap", 0) + 1
                )
                logger.debug(
                    "event=entry_blocked symbol=%s reason=leverage_cap "
                    "gross_after=%.2f cap=%.2f",
                    symbol, gross_after, portfolio_value * leverage_cap,
                )
                continue

            # ADV capacity constraint — cap at max_adv_participation % of 20-day ADV
            if self.max_adv_participation > 0 and 'Volume' in data[symbol].columns:
                hist_vol = data[symbol].loc[:date, 'Volume'].tail(20)
                if len(hist_vol) >= 5:
                    adv = hist_vol.mean()
                    if adv and adv > 0:
                        max_shares = adv * self.max_adv_participation
                        if shares > max_shares:
                            if asset_class == AssetClass.EQUITY:
                                shares = max(1, int(max_shares))
                            else:
                                shares = max_shares
                            logger.debug(
                                "ADV cap: %s capped to %.0f shares (%.1f%% of ADV %.0f)",
                                symbol, shares, self.max_adv_participation * 100, adv,
                            )

            # Check cash — LONGs need full notional, SHORTs need short margin.
            slippage_pct = self._estimate_slippage_pct(symbol, data, date, asset_class, shares)
            notional = shares * price * (1 + slippage_pct)
            commission = self._calculate_commission(symbol, notional)
            short_margin_pct = float(getattr(ApexConfig, "SHORT_MARGIN_PCT", 0.50))
            if signal > 0:
                cash_required = notional + commission
            else:
                cash_required = (abs(notional) * short_margin_pct) + commission
            if cash_required > self.cash:
                self._entry_block_reasons["cash_insufficient"] = (
                    self._entry_block_reasons.get("cash_insufficient", 0) + 1
                )
                continue

            side_str = 'BUY' if signal > 0 else 'SELL'
            side_sign = 1 if signal > 0 else -1

            # Compute initial risk R per share = ATR × PARTIAL_EXIT_R_STOP_MULT.
            atr_now = self._compute_atr(data, symbol, date)
            r_stop_mult = float(getattr(ApexConfig, "PARTIAL_EXIT_R_STOP_MULT", 2.5))
            risk_per_share = atr_now * r_stop_mult if atr_now > 0 else price * 0.02
            stop_price = price - side_sign * risk_per_share

            ok = self._execute_order(
                symbol, side_str, shares, price, date,
                f"Signal: {signal:.3f} [{sizing_mode}]",
                slippage_pct=slippage_pct,
                data=data,
            )
            if not ok:
                continue

            # Record metadata for the partial-exit state machine.
            self._position_meta[symbol] = {
                "entry_price": float(price),
                "side": side_sign,
                "initial_shares": float(shares),
                "risk_per_share": float(risk_per_share),
                "stop_price": float(stop_price),
                "stage": 0,
                "atr_at_entry": float(atr_now),
                "source": source_label,
                "sizing_mode": sizing_mode,
                "entry_corr": float(max_corr),
                "entry_date": date,
            }

            # Deployment sample taken AFTER the order fills so it reflects
            # actual committed capital. Round 13: switch from the net
            # (1 - cash/portfolio) formulation — which went negative on
            # short proceeds — to a gross view summing |notional| / PV.
            gross_after_fill = self._gross_exposure(data, date)
            gross_deployed = (
                gross_after_fill / portfolio_value
                if portfolio_value > 0 else 0.0
            )
            self._gross_deployed_samples.append(float(gross_deployed))
            # Preserve the legacy net figure for regression analysis.
            deployed = 1.0 - (self.cash / portfolio_value) if portfolio_value > 0 else 0.0
            self._deployed_capital_samples.append(float(deployed))

            current_positions += 1
            if current_positions >= max_positions:
                break
    
    def _compute_exec_cost_bps(
        self,
        symbol: str,
        data: Optional[Dict[str, pd.DataFrame]],
        date: Optional[datetime],
        asset_class: AssetClass,
        notional: float,
    ) -> Tuple[float, float, float]:
        """
        Round 12 FIX 5 — additive execution-cost model.

        Returns ``(spread_bps, impact_bps, commission_bps)`` for the given
        order. ``commission_bps`` already excludes the flat equity
        commission floor (which is captured elsewhere in dollar terms);
        it is exposed here only so the caller can reconstruct the full
        per-trade bps for reporting.
        """
        etf_symbols = set(getattr(ApexConfig, "ETF_SYMBOLS", []) or [])
        spread_etf = float(getattr(ApexConfig, "SPREAD_BPS_ETF", 1.0))
        spread_default = float(getattr(ApexConfig, "SPREAD_BPS_DEFAULT", 3.0))
        impact_mult = float(getattr(ApexConfig, "MARKET_IMPACT_MULT", 0.1))

        if asset_class == AssetClass.EQUITY and symbol in etf_symbols:
            spread_bps = spread_etf
        elif asset_class == AssetClass.EQUITY:
            spread_bps = spread_default
        elif asset_class == AssetClass.CRYPTO:
            spread_bps = float(self.crypto_spread_bps)
        elif asset_class == AssetClass.FOREX:
            spread_bps = float(self.fx_spread_bps)
        else:
            spread_bps = spread_default

        impact_bps = 0.0
        if impact_mult > 0.0 and data is not None and date is not None and symbol in data:
            hist = data[symbol].loc[:date]
            if (
                len(hist) >= 5
                and "Volume" in hist.columns
                and "Close" in hist.columns
            ):
                vol = hist["Volume"].tail(20).astype(float)
                px = hist["Close"].tail(20).astype(float)
                adv_usd = float((vol * px).mean())
                if adv_usd > 0.0 and abs(notional) > 0.0:
                    ratio = abs(float(notional)) / adv_usd
                    # 0.1 * sqrt(ratio) * 1e4 → 100 bps at 1% of ADV, 316 bps at 10%.
                    impact_bps = impact_mult * float(np.sqrt(ratio)) * 10_000.0

        # Commission-bps is just the asset-class bps component (ex the flat
        # equity commission floor) for reporting; the dollar commission is
        # still computed in ``_calculate_commission``.
        if asset_class == AssetClass.CRYPTO:
            commission_bps = float(self.crypto_commission_bps)
        elif asset_class == AssetClass.FOREX:
            commission_bps = float(self.fx_commission_bps)
        else:
            commission_bps = 0.0
        return float(spread_bps), float(impact_bps), float(commission_bps)

    def _execute_order(
        self,
        symbol: str,
        side: str,
        quantity: float,
        price: float,
        date: datetime,
        reason: str = "",
        slippage_pct: Optional[float] = None,
        data: Optional[Dict[str, pd.DataFrame]] = None,
    ):
        """Execute an order with realistic fills."""

        if not self._is_market_open(symbol, date):
            logger.debug("event=order_rejected symbol=%s reason=market_closed", symbol)
            return False

        try:
            parsed = parse_symbol(symbol)
            asset_class = parsed.asset_class
        except ValueError:
            logger.debug("event=order_rejected symbol=%s reason=invalid_symbol", symbol)
            return False

        logger.info(
            "event=symbol_normalization input=%s normalized=%s broker=%s",
            symbol,
            parsed.normalized,
            parsed.normalized,
        )

        if asset_class == AssetClass.EQUITY and isinstance(quantity, float) and not quantity.is_integer():
            logger.debug("event=order_rejected symbol=%s reason=fractional_equity quantity=%s", symbol, quantity)
            return False

        # Apply slippage (use pre-computed dynamic slippage if provided)
        if slippage_pct is None:
            slippage_pct = self._get_slippage_pct(asset_class)

        # Round 12 FIX 5 — additive spread + market-impact bps costs.
        notional_pre_slip = quantity * price
        spread_bps, impact_bps, _commission_bps = self._compute_exec_cost_bps(
            symbol,
            data,
            date,
            asset_class,
            notional_pre_slip,
        )
        extra_bps = spread_bps + impact_bps
        extra_pct = extra_bps / 10_000.0

        if side == 'BUY':
            execution_price = price * (1 + slippage_pct + extra_pct)
        else:
            execution_price = price * (1 - slippage_pct - extra_pct)

        # Calculate costs
        gross_value = quantity * execution_price
        commission = self._calculate_commission(symbol, gross_value)
        # Total bps of execution friction: slippage + spread + impact.
        # The flat equity commission shows up in dollar terms only.
        total_exec_bps = float((slippage_pct * 10_000.0) + extra_bps)
        logger.info(
            "event=fee_model asset=%s symbol=%s notional=%.2f commission=%.4f "
            "slippage_bps=%.2f spread_bps=%.2f impact_bps=%.2f total_bps=%.2f",
            asset_class.value,
            symbol,
            gross_value,
            commission,
            slippage_pct * 10_000.0,
            spread_bps,
            impact_bps,
            total_exec_bps,
        )

        # Round 13 FIX 1 — Reg-T-style short-margin accounting.
        # Branch based on the position BEFORE the fill:
        #   BUY  with pre_pos >= 0  : opening/extending long  (standard cash debit)
        #   BUY  with pre_pos <  0  : covering a short        (release margin + PnL)
        #   SELL with pre_pos >  0  : closing a long          (standard cash credit)
        #   SELL with pre_pos <= 0  : opening/extending short (debit margin, not credit)
        pre_pos = self.positions.get(symbol, 0)
        short_margin_pct = float(getattr(ApexConfig, "SHORT_MARGIN_PCT", 0.50))

        if side == 'BUY':
            if pre_pos < 0:
                # Cover a short (quantity must not exceed abs(pre_pos) — mixed
                # orders are not supported; guard explicitly).
                if quantity > abs(pre_pos) + 1e-9:
                    logger.error(
                        "event=order_rejected symbol=%s reason=mixed_cover_not_supported "
                        "qty=%s pre_pos=%s", symbol, quantity, pre_pos,
                    )
                    return False
                posted = float(self._short_margin_posted.get(symbol, 0.0))
                cover_fraction = quantity / abs(pre_pos) if pre_pos else 0.0
                margin_released = posted * cover_fraction
                entry_price = float(
                    self._position_meta.get(symbol, {}).get(
                        "entry_price", execution_price,
                    )
                )
                # Short PnL = (entry - cover) * qty, minus commission.
                short_pnl = (entry_price - execution_price) * quantity - commission
                self.cash += margin_released + short_pnl
                new_posted = max(0.0, posted - margin_released)
                if new_posted < 1e-6:
                    self._short_margin_posted.pop(symbol, None)
                else:
                    self._short_margin_posted[symbol] = new_posted
                self.positions[symbol] = pre_pos + quantity
            else:
                total_cost = gross_value + commission
                if total_cost > self.cash:
                    logger.debug(
                        f"Insufficient cash for {symbol}: need ${total_cost:,.2f}, "
                        f"have ${self.cash:,.2f}"
                    )
                    return False
                self.cash -= total_cost
                self.positions[symbol] = pre_pos + quantity

        else:  # SELL
            if pre_pos > 0:
                # Closing an existing long (quantity must not exceed pre_pos).
                if quantity > pre_pos + 1e-9:
                    logger.error(
                        "event=order_rejected symbol=%s reason=mixed_flip_not_supported "
                        "qty=%s pre_pos=%s", symbol, quantity, pre_pos,
                    )
                    return False
                total_proceeds = gross_value - commission
                self.cash += total_proceeds
                self.positions[symbol] = pre_pos - quantity
            else:
                # Opening/extending a short — debit margin rather than credit
                # proceeds so gross exposure stays capped by cash.
                margin_required = abs(gross_value) * short_margin_pct + commission
                if margin_required > self.cash:
                    logger.debug(
                        f"Insufficient margin for short {symbol}: need "
                        f"${margin_required:,.2f}, have ${self.cash:,.2f}"
                    )
                    return False
                self.cash -= margin_required
                self._short_margin_posted[symbol] = (
                    self._short_margin_posted.get(symbol, 0.0)
                    + abs(gross_value) * short_margin_pct
                )
                self.positions[symbol] = pre_pos - quantity

        # Record trade
        trade = {
            'date': date,
            'symbol': symbol,
            'side': side,
            'quantity': quantity,
            'price': price,
            'execution_price': execution_price,
            'commission': commission,
            'slippage': abs(execution_price - price) * quantity,
            'slippage_bps': float(slippage_pct * 10_000.0),
            'spread_bps': float(spread_bps),
            'impact_bps': float(impact_bps),
            'total_cost_bps': total_exec_bps,
            'reason': reason,
            'cash_after': self.cash
        }

        self.trades.append(trade)
        
        logger.debug(f"{date.date()}: {side} {quantity} {symbol} @ ${execution_price:.2f}")
        
        return True
    
    def _calculate_portfolio_value(
        self,
        data: Dict[str, pd.DataFrame],
        date: datetime
    ) -> float:
        """Calculate total portfolio value."""
        
        positions_value = 0
        
        for symbol, qty in self.positions.items():
            if qty == 0:
                continue
            
            if symbol not in data or date not in data[symbol].index:
                continue
            
            price = data[symbol].loc[date, 'Close']
            positions_value += qty * price
        
        return self.cash + positions_value
    
    def _calculate_metrics(self) -> Dict:
        """Calculate comprehensive backtest metrics."""
        
        if not self.equity_curve:
            return {}
        
        equity_df = pd.DataFrame(self.equity_curve)
        
        # Basic metrics
        final_value = equity_df['equity'].iloc[-1]
        total_return = (final_value / self.initial_capital) - 1
        
        # Annualized return
        days = len(equity_df)
        symbols_for_annual = self._data_symbols or list(self.positions.keys()) + [t['symbol'] for t in self.trades]
        ann_factor = self._annualization_factor(symbols_for_annual)
        years = days / ann_factor
        annual_return = (1 + total_return) ** (1 / years) - 1 if years > 0 else 0
        
        # Volatility and Sharpe — NaN-safe: degenerate samples (too few
        # points or zero variance) return NaN so walk-forward aggregation
        # can exclude them instead of silently collapsing to 0.
        daily_returns_arr = np.asarray(self.daily_returns, dtype=float)
        if daily_returns_arr.size < 5:
            annual_vol = 0.0
            sharpe_ratio = float("nan")
            sortino_ratio = float("nan")
        else:
            daily_vol = float(np.std(daily_returns_arr, ddof=0))
            annual_vol = daily_vol * float(np.sqrt(ann_factor))
            if daily_vol < 1e-8 or not np.isfinite(daily_vol):
                sharpe_ratio = float("nan")
            else:
                sharpe_ratio = float(
                    (float(np.mean(daily_returns_arr)) / daily_vol)
                    * float(np.sqrt(ann_factor))
                )

            downside = daily_returns_arr[daily_returns_arr < 0]
            if downside.size < 5:
                sortino_ratio = float("nan")
            else:
                downside_vol = float(np.std(downside, ddof=0)) * float(np.sqrt(ann_factor))
                sortino_ratio = (
                    float((annual_return - 0.02) / downside_vol)
                    if downside_vol > 0 else float("nan")
                )
        
        # Max Drawdown
        peak = equity_df['equity'].expanding().max()
        drawdown = (equity_df['equity'] - peak) / peak
        max_drawdown = drawdown.min()
        
        # Calmar Ratio
        calmar_ratio = annual_return / abs(max_drawdown) if abs(max_drawdown) > 0 else 0
        
        # Trade statistics
        trades_df = pd.DataFrame(self.trades)
        
        if not trades_df.empty:
            total_trades = len(trades_df)
            total_commissions = trades_df['commission'].sum()
            total_slippage = trades_df['slippage'].sum()
            
            # Calculate P&L per trade (simplified)
            winning_trades = 0
            total_pnl = 0
            gross_profit = 0
            gross_loss = 0
            
            # Match buys and sells for both long and short trades
            long_entries = defaultdict(list)   # BUY entries waiting for SELL exit
            short_entries = defaultdict(list)  # SELL entries waiting for BUY exit
            completed_trades = 0

            for _, trade in trades_df.iterrows():
                symbol = trade['symbol']

                if trade['side'] == 'BUY':
                    # Check if closing a short position first
                    if short_entries[symbol]:
                        entry = short_entries[symbol].pop(0)
                        pnl = (entry['price'] - trade['execution_price']) * trade['quantity']
                        pnl -= (entry['commission'] + trade['commission'])

                        total_pnl += pnl
                        completed_trades += 1
                        if pnl > 0:
                            winning_trades += 1
                            gross_profit += pnl
                        else:
                            gross_loss += abs(pnl)
                    else:
                        # Long entry
                        long_entries[symbol].append({
                            'qty': trade['quantity'],
                            'price': trade['execution_price'],
                            'commission': trade['commission']
                        })

                elif trade['side'] == 'SELL':
                    # Check if closing a long position first
                    if long_entries[symbol]:
                        entry = long_entries[symbol].pop(0)
                        pnl = (trade['execution_price'] - entry['price']) * trade['quantity']
                        pnl -= (entry['commission'] + trade['commission'])

                        total_pnl += pnl
                        completed_trades += 1
                        if pnl > 0:
                            winning_trades += 1
                            gross_profit += pnl
                        else:
                            gross_loss += abs(pnl)
                    else:
                        # Short entry
                        short_entries[symbol].append({
                            'qty': trade['quantity'],
                            'price': trade['execution_price'],
                            'commission': trade['commission']
                        })

            win_rate = winning_trades / completed_trades if completed_trades > 0 else 0
            avg_trade_pnl = total_pnl / completed_trades if completed_trades > 0 else 0
            profit_factor = gross_profit / gross_loss if gross_loss > 0 else (float('inf') if gross_profit > 0 else 0)
        
        else:
            total_trades = 0
            total_commissions = 0
            total_slippage = 0
            win_rate = 0
            avg_trade_pnl = 0
            profit_factor = 0
        
        # Probabilistic Sharpe Ratio (PSR) and Deflated Sharpe Ratio (DSR)
        psr = 0.0
        dsr = 0.0
        n_obs = len(self.daily_returns)
        if n_obs >= 10 and sharpe_ratio > 0:
            returns_arr = np.array(self.daily_returns)
            skew = float(pd.Series(returns_arr).skew())
            kurt = float(pd.Series(returns_arr).kurtosis()) + 3  # convert excess to raw

            # Standard error of Sharpe accounting for non-normality (Lo 2002)
            sr_ann = sharpe_ratio  # already annualized
            sr_per = sr_ann / np.sqrt(ann_factor)  # per-period Sharpe for SE formula
            se_denom = 1 - skew * sr_per + ((kurt - 1) / 4) * sr_per ** 2
            se_sr = np.sqrt(max(se_denom, 1e-10) / (n_obs - 1))

            # PSR: P(true SR > 0)
            psr = float(_norm_cdf(sr_per / se_sr)) if se_sr > 0 else 0.0

            # DSR: P(true SR > expected max SR under null)
            n_trials = self._n_sharpe_trials
            if n_trials > 1:
                e_max_sr = (
                    (1 - EULER_MASCHERONI) * _norm_ppf(1 - 1.0 / n_trials)
                    + EULER_MASCHERONI * _norm_ppf(1 - 1.0 / (n_trials * np.e))
                ) * np.sqrt(1.0 / (n_obs - 1))
                dsr = float(_norm_cdf((sr_per - e_max_sr) / se_sr)) if se_sr > 0 else 0.0
            else:
                dsr = psr  # with 1 trial, DSR = PSR

        # ── Round 11 diagnostics ──────────────────────────────────────────
        winner_r = [
            float(t.get("realised_r", 0.0))
            for t in self.trades
            if t.get("realised_r") is not None
            and float(t.get("realised_r", 0.0)) > 0.0
        ]
        avg_winner_r = float(np.mean(winner_r)) if winner_r else 0.0

        corr_samples = list(self._portfolio_corr_samples)
        avg_portfolio_correlation = (
            float(np.mean(corr_samples)) if corr_samples else 0.0
        )

        deployed_samples = list(self._deployed_capital_samples)
        avg_capital_deployed = (
            float(np.mean(deployed_samples)) if deployed_samples else 0.0
        )

        gross_deployed_samples = list(self._gross_deployed_samples)
        avg_gross_deployed = (
            float(np.mean(gross_deployed_samples))
            if gross_deployed_samples else 0.0
        )

        cost_bps_vals = [
            float(t.get("total_cost_bps", 0.0))
            for t in self.trades
            if t.get("total_cost_bps") is not None
        ]
        avg_total_cost_bps = (
            float(np.mean(cost_bps_vals)) if cost_bps_vals else 0.0
        )

        risk_on_samples = list(self._macro_regime_samples)
        risk_on_fraction = (
            float(np.mean(risk_on_samples)) if risk_on_samples else 1.0
        )
        entry_block_reasons = dict(self._entry_block_reasons)

        return {
            'initial_capital': self.initial_capital,
            'final_value': final_value,
            'total_return': total_return,
            'annual_return': annual_return,
            'annual_volatility': annual_vol,
            'sharpe_ratio': sharpe_ratio,
            'probabilistic_sharpe': psr,
            'deflated_sharpe': dsr,
            'sortino_ratio': sortino_ratio,
            'calmar_ratio': calmar_ratio,
            'max_drawdown': max_drawdown,
            'total_trades': total_trades,
            'win_rate': win_rate,
            'avg_trade_pnl': avg_trade_pnl,
            'profit_factor': profit_factor,
            'total_commissions': total_commissions,
            'total_slippage': total_slippage,
            'total_borrow_costs': self._borrow_costs_total,
            'n_sharpe_trials': self._n_sharpe_trials,
            'avg_winner_r': avg_winner_r,
            'avg_portfolio_correlation': avg_portfolio_correlation,
            'avg_capital_deployed': avg_capital_deployed,
            'avg_gross_deployed': avg_gross_deployed,
            'avg_total_cost_bps': avg_total_cost_bps,
            'risk_on_fraction': risk_on_fraction,
            'entry_block_reasons': entry_block_reasons,
            'orb_active': bool(getattr(__import__('signals.orb_signal', fromlist=['ORBSignal']).ORBSignal, 'is_active', lambda: False)()),
            'equity_curve': equity_df,
            'trades': trades_df
        }
    
    def plot_results(self, results: Dict):
        """Plot backtest results."""
        try:
            import matplotlib.pyplot as plt
            
            equity_df = results['equity_curve']
            
            fig, axes = plt.subplots(2, 1, figsize=(12, 8))
            
            # Equity curve
            axes[0].plot(equity_df['date'], equity_df['equity'], label='Portfolio Value')
            axes[0].axhline(y=self.initial_capital, color='r', linestyle='--', label='Initial Capital')
            axes[0].set_title('Equity Curve')
            axes[0].set_ylabel('Portfolio Value ($)')
            axes[0].legend()
            axes[0].grid(True, alpha=0.3)
            
            # Drawdown
            peak = equity_df['equity'].expanding().max()
            drawdown = (equity_df['equity'] - peak) / peak * 100
            axes[1].fill_between(equity_df['date'], drawdown, 0, alpha=0.3, color='red')
            axes[1].set_title('Drawdown')
            axes[1].set_ylabel('Drawdown (%)')
            axes[1].set_xlabel('Date')
            axes[1].grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig('backtest_results.png', dpi=150)
            logger.info("Results saved to backtest_results.png")
            
        except ImportError:
            logger.warning("Matplotlib not available for plotting")

    def generate_tear_sheet(self, results: Dict, output_filename: str) -> bool:
        """Generate static HTML tear sheet using quantstats."""
        try:
            import quantstats as qs
            
            equity_df = results.get('equity_curve')
            if equity_df is None or equity_df.empty:
                logger.warning("No equity curve available to generate tear sheet.")
                return False
                
            # Ensure we have a datetime index mapped to daily returns
            df_copy = equity_df.copy()
            df_copy = df_copy.set_index('date')
            
            # Calculate daily returns from equity
            returns = df_copy['equity'].pct_change().dropna()
            
            # Generate the report
            # Note: quantstats relies on matplotlib, seaborn, etc.
            # Output generates a standalone HTML file.
            qs.reports.html(returns, output=output_filename, title="Apex Trading Backtest Report")
            logger.info(f"📄 Static HTML tear sheet generated: {output_filename}")
            return True
        except ImportError:
            logger.error("quantstats is not installed. Run `pip install quantstats`.")
            return False
        except Exception as e:
            logger.error(f"Failed to generate tear sheet: {e}")
            return False


if __name__ == "__main__":
    # Test backtester
    setup_logging(level="INFO", log_file=None, json_format=False, console_output=True)
    
    # Generate sample data
    np.random.seed(42)
    dates = pd.date_range('2023-01-01', '2024-01-01', freq='D')
    
    data = {}
    for symbol in ['AAPL', 'MSFT', 'GOOGL']:
        # Generate random OHLCV
        close = 100 + np.random.randn(len(dates)).cumsum()
        data[symbol] = pd.DataFrame({
            'Open': close + np.random.randn(len(dates)) * 0.5,
            'High': close + abs(np.random.randn(len(dates))),
            'Low': close - abs(np.random.randn(len(dates))),
            'Close': close,
            'Volume': np.random.randint(1000000, 10000000, len(dates))
        }, index=dates)
    
    # Mock signal generator
    class MockSignalGenerator:
        def generate_ml_signal(self, symbol, prices):
            return {'signal': np.random.randn() * 0.5}
    
    # Run backtest
    backtester = AdvancedBacktester(initial_capital=100000)
    
    results = backtester.run_backtest(
        data=data,
        signal_generator=MockSignalGenerator(),
        start_date='2023-01-01',
        end_date='2023-12-31',
        position_size_usd=5000,
        max_positions=3
    )
    
    print("\n✅ Backtester tests complete!")

    def replay_from_journal(self, journal_path: str) -> Dict[str, Any]:
        """
        Deterministic Replay: Reconstructs the exact state of the system
        and recalculates the P&L using the Write-Ahead Log (WAL).
        """
        import json
        import hashlib
        from pathlib import Path
        
        logger.info(f"⏪ Initiating deterministic replay from journal: {journal_path}")
        
        if not Path(journal_path).exists():
            raise FileNotFoundError(f"Journal not found at {journal_path}")
            
        replay_capital = self.initial_capital
        replay_positions = {}
        trade_log = []
        
        last_hash = hashlib.sha256(b"genesis").hexdigest()
        tampered = False
        line_no = 0
        
        with open(journal_path, 'r') as f:
            for line_no, line in enumerate(f, 1):
                try:
                    event = json.loads(line)
                    
                    # Verify Tamper-Evident Chain
                    expected_raw = f"{last_hash}|{event['timestamp']}|{event['type']}|{json.dumps(event['payload'], sort_keys=True)}"
                    expected_hash = hashlib.sha256(expected_raw.encode()).hexdigest()
                    
                    if expected_hash != event.get('hash'):
                        logger.error(f"🚨 CHAIN OF CUSTODY BROKEN AT LINE {line_no}. Journal tampered with!")
                        tampered = True
                        break
                        
                    last_hash = expected_hash
                    
                    # Replay State Mutations
                    if event["type"] == "POSITION_UPDATE":
                        sym = event["payload"]["symbol"]
                        qty = event["payload"]["quantity"]
                        px = event["payload"].get("price", 0.0)
                        
                        prev_qty = replay_positions.get(sym, 0)
                        replay_positions[sym] = qty
                        
                        if prev_qty != qty:
                            trade_log.append({
                                "timestamp": event["timestamp"],
                                "symbol": sym,
                                "side": "BUY" if qty > prev_qty else "SELL",
                                "qty_change": abs(qty - prev_qty),
                                "price": px
                            })
                            
                    elif event["type"] == "CAPITAL_UPDATE":
                        replay_capital = event["payload"]["capital"]
                        
                except Exception as e:
                    logger.error(f"Replay failed parsing line {line_no}: {e}")
                    
        logger.info(f"✅ Replay Complete. Final Capital: ${replay_capital:,.2f}")
        return {
            "tampered": tampered,
            "final_capital": replay_capital,
            "final_positions": replay_positions,
            "total_events_processed": line_no,
            "trade_log": trade_log
        }
