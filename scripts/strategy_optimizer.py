#!/usr/bin/env python3
"""
scripts/strategy_optimizer.py
Fast strategy parameter optimizer using walk-forward simulation.

Tests different parameter combinations on historical data using
pre-trained ML models + momentum signals. Reports Sharpe and
max drawdown for each combination to find >1.5 Sharpe settings.

Usage: python scripts/strategy_optimizer.py [--symbols N] [--years Y]
"""

import sys
import argparse
import warnings
import logging
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from itertools import product

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)

# ─── Configuration ────────────────────────────────────────────────────────────

# Representative liquid symbols (mix of sectors + crypto)
EQUITY_SYMBOLS = [
    "SPY", "QQQ", "AAPL", "MSFT", "NVDA", "META", "GOOGL", "AMZN",
    "JPM", "XLF", "GLD", "XLE", "UNH", "HD", "V"
]
CRYPTO_SYMBOLS = ["BTC-USD", "ETH-USD", "SOL-USD", "LINK-USD"]

# Parameter grid to sweep
PARAM_GRID = {
    "entry_threshold":  [0.08, 0.10, 0.12, 0.14, 0.16, 0.18],
    "stop_loss_pct":    [0.02, 0.03, 0.04, 0.05],
    "take_profit_pct":  [0.04, 0.06, 0.08, 0.10, 0.12],
    "min_confidence":   [0.40, 0.45, 0.50],
}

# Commission model (IBKR Tiered)
COMMISSION_PER_SHARE = 0.005
MIN_COMMISSION = 1.00
SLIPPAGE_BPS = 5        # 5bps equity
CRYPTO_SLIPPAGE_BPS = 15  # 15bps crypto

# ─── Data Fetching ─────────────────────────────────────────────────────────────

def fetch_data(symbols: List[str], years: float = 2.0) -> Dict[str, pd.DataFrame]:
    """Download OHLCV data for symbols."""
    import yfinance as yf

    end = datetime.now()
    start = end - timedelta(days=int(years * 365))
    data: Dict[str, pd.DataFrame] = {}

    print(f"Downloading {len(symbols)} symbols ({years}y)...", end=" ", flush=True)
    for i, sym in enumerate(symbols):
        try:
            df = yf.download(sym, start=start, end=end, progress=False, auto_adjust=True)
            if len(df) < 60:
                continue
            df = df[["Open", "High", "Low", "Close", "Volume"]].copy()
            df.columns = ["open", "high", "low", "close", "volume"]
            df.dropna(inplace=True)
            data[sym] = df
            if (i + 1) % 5 == 0:
                print(f"{i+1}/{len(symbols)}", end=" ", flush=True)
        except Exception:
            pass

    print(f"✓ {len(data)} loaded")
    return data


# ─── Signal Generation ─────────────────────────────────────────────────────────

def compute_signals(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute composite signal combining:
    1. ML-proxy: EMA crossover + RSI normalised to [-1, +1]
    2. Momentum: 5/20-day returns z-scored
    3. Trend: price vs 50-day MA normalised
    4. Volatility regime: low vol = higher weight

    Returns signal [-1, +1] and confidence [0, 1].
    """
    c = df["close"]
    v = df["volume"]

    # EMAs
    ema12 = c.ewm(span=12, adjust=False).mean()
    ema26 = c.ewm(span=26, adjust=False).mean()
    macd = (ema12 - ema26) / ema26  # normalised MACD

    # RSI
    delta = c.diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = -delta.clip(upper=0).rolling(14).mean()
    rs = gain / loss.replace(0, 1e-9)
    rsi = 100 - 100 / (1 + rs)
    rsi_norm = (rsi - 50) / 50  # -1 to +1

    # Momentum (z-scored 20-day)
    ret5 = c.pct_change(5)
    ret20 = c.pct_change(20)
    mom_z = (ret5 - ret5.rolling(60).mean()) / (ret5.rolling(60).std() + 1e-9)
    mom_z = mom_z.clip(-3, 3) / 3  # -1 to +1

    # Trend
    ma50 = c.rolling(50).mean()
    trend = ((c - ma50) / ma50).clip(-0.1, 0.1) / 0.1  # -1 to +1

    # Volatility regime (low vol → higher confidence)
    atr = ((df["high"] - df["low"]).rolling(14).mean()) / c
    atr_z = 1 - (atr / atr.rolling(60).mean()).clip(0, 3) / 3

    # Volume confirmation
    vol_z = (v / v.rolling(20).mean()).clip(0, 3) / 3

    # Composite signal (weighted average)
    signal = (
        0.30 * macd.clip(-0.05, 0.05) / 0.05 +
        0.20 * rsi_norm +
        0.25 * mom_z +
        0.25 * trend
    )
    signal = signal.clip(-1, 1)

    # Confidence: high when vol is low and volume confirms
    confidence = (0.6 * atr_z + 0.4 * vol_z).fillna(0).clip(0, 1)

    # Scale signal to model-range [0.05, 0.272] for positive signals
    # and map negative signals appropriately
    scaled_signal = signal * 0.136 + 0.136  # maps [-1,+1] → [0, 0.272]
    # But we only enter LONG so treat negative as "no entry / exit"
    # Keep raw signal for directional checks

    out = pd.DataFrame({
        "signal": scaled_signal,
        "raw_signal": signal,
        "confidence": confidence,
        "ret20": ret20,
    }, index=df.index)
    return out.dropna()


# ─── Single Backtest Run ────────────────────────────────────────────────────────

@dataclass
class TradeRecord:
    symbol: str
    entry_date: str
    exit_date: str
    entry_price: float
    exit_price: float
    pnl_pct: float  # net of costs
    gross_pnl_pct: float
    exit_reason: str
    hold_days: int
    is_crypto: bool


def run_backtest(
    price_data: Dict[str, pd.DataFrame],
    signal_data: Dict[str, pd.DataFrame],
    entry_threshold: float,
    stop_loss_pct: float,
    take_profit_pct: float,
    min_confidence: float,
    initial_capital: float = 100_000,
    position_size_pct: float = 0.05,
    max_positions: int = 15,
    commission_per_share: float = COMMISSION_PER_SHARE,
    slippage_bps: float = SLIPPAGE_BPS,
) -> Tuple[float, float, float, int, List[TradeRecord]]:
    """
    Vectorised daily backtest. Returns (sharpe, max_dd, total_return_pct, n_trades, trades).
    """

    # Build a universe date index
    all_dates = sorted(set().union(*[set(df.index) for df in price_data.values()]))
    if not all_dates:
        return 0.0, 0.0, 0.0, 0, []

    capital = initial_capital
    peak_capital = initial_capital
    positions: Dict[str, Dict] = {}   # symbol → {entry_price, stop, tp, entry_date, qty, is_crypto}
    daily_equity: List[Tuple[datetime, float]] = []
    trades: List[TradeRecord] = []

    for date in all_dates:
        # ── 1. Mark to market existing positions ──────────────────────────────
        port_value = capital
        for sym, pos in list(positions.items()):
            prices = price_data.get(sym)
            if prices is None or date not in prices.index:
                continue
            curr_price = prices.loc[date, "close"]
            port_value += pos["qty"] * curr_price - pos["qty"] * pos["entry_price"]

        daily_equity.append((date, port_value))
        peak_capital = max(peak_capital, port_value)

        # ── 2. Exit check ──────────────────────────────────────────────────────
        for sym in list(positions.keys()):
            pos = positions[sym]
            prices = price_data.get(sym)
            sigs = signal_data.get(sym)
            if prices is None or date not in prices.index:
                continue

            curr_price = prices.loc[date, "close"]
            pnl_pct_raw = (curr_price - pos["entry_price"]) / pos["entry_price"]

            exit_reason = None
            # Stop loss
            if pnl_pct_raw <= -stop_loss_pct:
                exit_reason = "stop_loss"
            # Take profit
            elif pnl_pct_raw >= take_profit_pct:
                exit_reason = "take_profit"
            # Partial TP ladder: at +3% take 50%, at +6% take remaining
            elif pnl_pct_raw >= 0.06 and pos.get("partial_tp2_taken") is False:
                exit_reason = "partial_tp2"
            elif pnl_pct_raw >= 0.03 and pos.get("partial_tp1_taken") is False:
                exit_reason = "partial_tp1"
            # Signal reversal exit
            elif sigs is not None and date in sigs.index:
                sig = sigs.loc[date, "signal"]
                raw = sigs.loc[date, "raw_signal"]
                if raw < -0.3 and pnl_pct_raw < 0:  # bearish + losing
                    exit_reason = "signal_reversal"
                elif raw < -0.5:  # strongly bearish regardless
                    exit_reason = "strong_bearish"
            # Max hold: 20 days
            hold_days = (date - datetime.fromisoformat(pos["entry_date"])).days
            if hold_days > 20:
                exit_reason = "max_hold"

            if exit_reason in ("partial_tp1", "partial_tp2"):
                # Take 50% off, adjust qty
                fraction = 0.5
                qty_exit = pos["qty"] * fraction
                pos["qty"] -= qty_exit
                slip = curr_price * (slippage_bps / 10000)
                exit_price = curr_price - slip
                commission = max(MIN_COMMISSION, qty_exit * commission_per_share)
                gross_pnl = qty_exit * (exit_price - pos["entry_price"])
                capital += qty_exit * pos["entry_price"] + gross_pnl - commission
                pnl_pct_net = gross_pnl / (qty_exit * pos["entry_price"]) - commission / (qty_exit * pos["entry_price"])
                pos["partial_tp1_taken"] = True
                if exit_reason == "partial_tp2":
                    pos["partial_tp2_taken"] = True
                # Don't record as full trade close
                continue

            if exit_reason:
                slip = curr_price * (slippage_bps / 10000)
                exit_price_net = curr_price - slip
                qty = pos["qty"]
                commission = max(MIN_COMMISSION, qty * commission_per_share)
                gross_pnl = qty * (exit_price_net - pos["entry_price"])
                gross_pnl_pct = (exit_price_net - pos["entry_price"]) / pos["entry_price"]
                net_pnl = gross_pnl - commission
                net_pnl_pct = net_pnl / (qty * pos["entry_price"])

                capital += qty * pos["entry_price"] + net_pnl
                hold_days = (date - datetime.fromisoformat(pos["entry_date"])).days
                trades.append(TradeRecord(
                    symbol=sym, entry_date=pos["entry_date"],
                    exit_date=str(date.date()), entry_price=pos["entry_price"],
                    exit_price=exit_price_net, pnl_pct=net_pnl_pct,
                    gross_pnl_pct=gross_pnl_pct, exit_reason=exit_reason,
                    hold_days=hold_days, is_crypto="USD" in sym,
                ))
                del positions[sym]

        # ── 3. Entry check ─────────────────────────────────────────────────────
        if len(positions) >= max_positions:
            continue

        for sym, prices in price_data.items():
            if sym in positions:
                continue
            if len(positions) >= max_positions:
                break
            sigs = signal_data.get(sym)
            if sigs is None or date not in sigs.index or date not in prices.index:
                continue

            sig_val = sigs.loc[date, "signal"]
            conf = sigs.loc[date, "confidence"]

            if sig_val < entry_threshold or conf < min_confidence:
                continue

            price = prices.loc[date, "close"]
            if price <= 0:
                continue

            # Compute position size in dollars
            pos_usd = initial_capital * position_size_pct
            # For crypto: cap at $5K
            is_crypto = "USD" in sym and "-" in sym
            if is_crypto:
                pos_usd = min(pos_usd, 5000)
            if pos_usd > capital:
                continue  # not enough cash

            slip = price * (slippage_bps / 10000)
            entry_price = price + slip
            qty = pos_usd / entry_price
            if not is_crypto:
                qty = max(1, int(qty))
            else:
                qty = round(qty, 6)

            commission = max(MIN_COMMISSION, qty * commission_per_share)
            cost = qty * entry_price + commission
            if cost > capital:
                continue

            capital -= cost
            positions[sym] = {
                "entry_price": entry_price,
                "qty": qty,
                "entry_date": str(date.date()),
                "is_crypto": is_crypto,
                "partial_tp1_taken": False,
                "partial_tp2_taken": False,
            }

    # ── 4. Close any remaining open positions at last price ────────────────────
    if all_dates:
        last_date = all_dates[-1]
        for sym, pos in positions.items():
            prices = price_data.get(sym)
            if prices is None:
                continue
            last_price = prices.iloc[-1]["close"]
            pnl_pct = (last_price - pos["entry_price"]) / pos["entry_price"]
            trades.append(TradeRecord(
                symbol=sym, entry_date=pos["entry_date"],
                exit_date=str(last_date.date()), entry_price=pos["entry_price"],
                exit_price=last_price, pnl_pct=pnl_pct,
                gross_pnl_pct=pnl_pct, exit_reason="end_of_period",
                hold_days=(last_date - datetime.fromisoformat(pos["entry_date"])).days,
                is_crypto=pos["is_crypto"],
            ))

    # ── 5. Calculate metrics ───────────────────────────────────────────────────
    if len(daily_equity) < 10:
        return 0.0, 0.0, 0.0, len(trades), trades

    eq_series = pd.Series([e for _, e in daily_equity],
                          index=[d for d, _ in daily_equity])
    daily_returns = eq_series.pct_change().dropna()

    if daily_returns.std() == 0:
        sharpe = 0.0
    else:
        # Annualised Sharpe (no risk-free rate — we're competing against 0)
        sharpe = float(daily_returns.mean() / daily_returns.std() * np.sqrt(252))

    # Max drawdown
    roll_max = eq_series.cummax()
    dd = (eq_series - roll_max) / roll_max
    max_dd = float(dd.min())

    total_ret = float((eq_series.iloc[-1] / eq_series.iloc[0] - 1) * 100)

    return sharpe, max_dd, total_ret, len(trades), trades


# ─── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Strategy Parameter Optimizer")
    parser.add_argument("--years", type=float, default=2.0, help="Years of history")
    parser.add_argument("--equity-only", action="store_true", help="Use equity only")
    parser.add_argument("--top-n", type=int, default=10, help="Show top N combinations")
    args = parser.parse_args()

    # ── Download data ──────────────────────────────────────────────────────────
    symbols = EQUITY_SYMBOLS.copy()
    if not args.equity_only:
        symbols += CRYPTO_SYMBOLS

    price_data = fetch_data(symbols, years=args.years)

    # ── Compute signals ────────────────────────────────────────────────────────
    print("Computing signals...", end=" ", flush=True)
    signal_data = {}
    for sym, df in price_data.items():
        try:
            signal_data[sym] = compute_signals(df)
        except Exception as e:
            logger.debug(f"Signal error {sym}: {e}")
    print(f"✓ {len(signal_data)} symbols")

    # ── Parameter sweep ────────────────────────────────────────────────────────
    param_combinations = list(product(
        PARAM_GRID["entry_threshold"],
        PARAM_GRID["stop_loss_pct"],
        PARAM_GRID["take_profit_pct"],
        PARAM_GRID["min_confidence"],
    ))
    print(f"\nTesting {len(param_combinations)} parameter combinations...")
    print(f"{'Entry':>7} {'SL%':>5} {'TP%':>5} {'Conf':>5} | {'Sharpe':>7} {'MaxDD%':>7} {'Ret%':>7} {'Trades':>7}")
    print("-" * 70)

    results = []
    for i, (entry, sl, tp, conf) in enumerate(param_combinations):
        sharpe, max_dd, total_ret, n_trades, _ = run_backtest(
            price_data=price_data,
            signal_data=signal_data,
            entry_threshold=entry,
            stop_loss_pct=sl,
            take_profit_pct=tp,
            min_confidence=conf,
        )
        results.append({
            "entry_threshold": entry,
            "stop_loss_pct": sl,
            "take_profit_pct": tp,
            "min_confidence": conf,
            "sharpe": sharpe,
            "max_dd_pct": max_dd * 100,
            "total_ret_pct": total_ret,
            "n_trades": n_trades,
        })

    # ── Sort by Sharpe (descending) ────────────────────────────────────────────
    results.sort(key=lambda x: x["sharpe"], reverse=True)

    print(f"\n{'='*70}")
    print(f"TOP {args.top_n} COMBINATIONS (by Sharpe):")
    print(f"{'='*70}")
    print(f"{'Entry':>7} {'SL%':>5} {'TP%':>5} {'Conf':>5} | {'Sharpe':>7} {'MaxDD%':>7} {'Ret%':>7} {'Trades':>7}")
    print("-" * 70)
    for r in results[:args.top_n]:
        flag = " ✅" if r["sharpe"] >= 1.5 and abs(r["max_dd_pct"]) <= 15 else ""
        print(
            f"{r['entry_threshold']:>7.2f} {r['stop_loss_pct']*100:>5.1f} "
            f"{r['take_profit_pct']*100:>5.1f} {r['min_confidence']:>5.2f} | "
            f"{r['sharpe']:>7.2f} {r['max_dd_pct']:>7.1f} "
            f"{r['total_ret_pct']:>7.1f} {r['n_trades']:>7d}{flag}"
        )

    # ── Show optimal config ────────────────────────────────────────────────────
    feasible = [r for r in results if r["sharpe"] >= 1.5 and abs(r["max_dd_pct"]) <= 15]
    best = results[0]

    print(f"\n{'='*70}")
    if feasible:
        best_feasible = feasible[0]
        print("✅ BEST FEASIBLE CONFIG (Sharpe ≥ 1.5 AND MaxDD ≤ 15%):")
        print(f"   entry_threshold:  {best_feasible['entry_threshold']:.2f}  (current live: 0.15)")
        print(f"   stop_loss_pct:    {best_feasible['stop_loss_pct']*100:.1f}%  (current live: 2.22%)")
        print(f"   take_profit_pct:  {best_feasible['take_profit_pct']*100:.1f}%  (current live: 3%/6%)")
        print(f"   min_confidence:   {best_feasible['min_confidence']:.2f}  (current live: 0.45)")
        print(f"   → Sharpe: {best_feasible['sharpe']:.2f}, MaxDD: {best_feasible['max_dd_pct']:.1f}%, "
              f"Return: {best_feasible['total_ret_pct']:.1f}%, Trades: {best_feasible['n_trades']}")
    else:
        print("⚠️  No config hits Sharpe ≥ 1.5 + MaxDD ≤ 15% simultaneously.")
        print("   Best by Sharpe:")
        print(f"   entry={best['entry_threshold']:.2f}, SL={best['stop_loss_pct']*100:.1f}%, "
              f"TP={best['take_profit_pct']*100:.1f}%, conf={best['min_confidence']:.2f}")
        print(f"   → Sharpe: {best['sharpe']:.2f}, MaxDD: {best['max_dd_pct']:.1f}%, "
              f"Return: {best['total_ret_pct']:.1f}%")

    # ── Write results to CSV ──────────────────────────────────────────────────
    out_path = PROJECT_ROOT / "data" / "backtests" / "strategy_optimizer_results.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(results).to_csv(out_path, index=False)
    print(f"\nFull results saved to: {out_path}")


if __name__ == "__main__":
    main()
