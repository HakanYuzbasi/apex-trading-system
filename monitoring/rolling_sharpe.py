"""
monitoring/rolling_sharpe.py — Rolling Sharpe from closed-trade PnL
====================================================================

The existing dashboard Sharpe was a Calmar proxy (return ÷ drawdown) computed
on aggregated session metrics. It moves slowly and ignores the distribution of
trade outcomes. This module computes a proper annualised Sharpe over a
configurable rolling window of closed trades, reading the same
``performance_attribution.json`` ledger that backs the live PnL numbers.

Design notes:

- PnL stream is the realised ``pnl_pct_on_entry_notional`` of every trade that
  closed within the lookback window. When missing, fall back to
  ``net_pnl / entry_notional``.
- Annualisation factor assumes a trade-per-day frequency on average; this is
  intentionally conservative. Projects with higher trade frequency can widen
  ``ROLLING_SHARPE_DAYS`` to smooth the number.
- Returns ``0.0`` when fewer than ``ROLLING_SHARPE_MIN_TRADES`` qualifying
  closures exist — guards against noisy single-digit-sample readings.
"""
from __future__ import annotations

import json
import logging
import math
from datetime import datetime, timedelta
from pathlib import Path
from typing import Iterable, Optional

from config import ApexConfig

logger = logging.getLogger(__name__)


_ANNUAL_TRADING_DAYS: float = 252.0


def _as_float(value: object) -> Optional[float]:
    """Parse ``value`` to float, returning ``None`` on any failure."""
    try:
        out = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
    if not math.isfinite(out):
        return None
    return out


def _iter_closed_trades(raw: object) -> Iterable[dict]:
    """
    Normalise the various shapes ``performance_attribution.json`` has taken
    over the project's lifetime into a flat iterator of trade dicts.
    """
    if isinstance(raw, list):
        yield from (r for r in raw if isinstance(r, dict))
        return
    if isinstance(raw, dict):
        trades = raw.get("closed_trades")
        if isinstance(trades, list):
            yield from (r for r in trades if isinstance(r, dict))


def _extract_return_pct(trade: dict) -> Optional[float]:
    """
    Return the trade's realised % return as a decimal fraction (e.g. 0.012 for
    +1.2%). ``None`` when the trade lacks enough metadata to compute it.
    """
    pnl_bps = _as_float(trade.get("pnl_bps_on_entry_notional"))
    if pnl_bps is not None:
        return pnl_bps / 10_000.0

    net_pnl = _as_float(trade.get("net_pnl"))
    notional = _as_float(trade.get("entry_notional"))
    if notional is None:
        entry_price = _as_float(trade.get("entry_price"))
        quantity = _as_float(trade.get("quantity"))
        if entry_price is not None and quantity is not None:
            notional = entry_price * quantity
    if net_pnl is None or not notional:
        return None
    return net_pnl / notional


def compute_rolling_sharpe_from_trades(
    trades: Iterable[dict],
    *,
    lookback_days: Optional[int] = None,
    min_trades: Optional[int] = None,
    now: Optional[datetime] = None,
) -> float:
    """
    Compute an annualised Sharpe ratio from an iterable of closed trades.

    Args:
        trades: Iterable of trade dicts. Each trade must expose ``exit_time``
            (ISO-8601 string) and either ``pnl_bps_on_entry_notional`` or the
            fallback pair ``net_pnl`` + ``entry_notional`` / (``entry_price``
            × ``quantity``).
        lookback_days: Number of days to include. Defaults to
            ``ApexConfig.ROLLING_SHARPE_DAYS``.
        min_trades: Minimum trade count required before the Sharpe is
            considered stable. Defaults to ``ApexConfig.ROLLING_SHARPE_MIN_TRADES``.
        now: Reference timestamp. Defaults to ``datetime.utcnow()``.

    Returns:
        Annualised Sharpe ratio, or ``0.0`` when insufficient history.
    """
    if lookback_days is None:
        lookback_days = int(getattr(ApexConfig, "ROLLING_SHARPE_DAYS", 30))
    if min_trades is None:
        min_trades = int(getattr(ApexConfig, "ROLLING_SHARPE_MIN_TRADES", 10))
    if lookback_days <= 0:
        return 0.0

    cutoff = (now or datetime.utcnow()) - timedelta(days=lookback_days)
    returns: list[float] = []
    for trade in trades:
        exit_time_raw = trade.get("exit_time")
        if not exit_time_raw:
            continue
        try:
            exit_dt = datetime.fromisoformat(str(exit_time_raw).replace("Z", ""))
        except ValueError:
            continue
        if exit_dt < cutoff:
            continue
        ret = _extract_return_pct(trade)
        if ret is None:
            continue
        returns.append(ret)

    if len(returns) < max(2, min_trades):
        return 0.0

    mean_ret = sum(returns) / len(returns)
    variance = sum((r - mean_ret) ** 2 for r in returns) / (len(returns) - 1)
    if variance <= 0.0:
        return 0.0
    std = math.sqrt(variance)

    # Annualise using the empirical trade frequency: trades / lookback ≈
    # trades-per-day, scale by sqrt(252 × per-day).
    trades_per_day = len(returns) / max(1.0, float(lookback_days))
    annualisation = math.sqrt(_ANNUAL_TRADING_DAYS * max(trades_per_day, 1.0 / lookback_days))
    sharpe = (mean_ret / std) * annualisation
    return round(sharpe, 3)


def compute_rolling_sharpe_from_file(
    path: Path,
    *,
    lookback_days: Optional[int] = None,
    min_trades: Optional[int] = None,
    now: Optional[datetime] = None,
) -> float:
    """
    Read ``performance_attribution.json`` from disk and compute the rolling
    Sharpe. Never raises — on any parse / IO failure returns ``0.0``.

    Args:
        path: Path to the attribution JSON file.
        lookback_days: See :func:`compute_rolling_sharpe_from_trades`.
        min_trades: See :func:`compute_rolling_sharpe_from_trades`.
        now: See :func:`compute_rolling_sharpe_from_trades`.

    Returns:
        Annualised Sharpe ratio or ``0.0`` when unavailable.
    """
    if not path.exists():
        return 0.0
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as exc:
        logger.debug("rolling_sharpe: cannot read %s (%s)", path, exc)
        return 0.0
    return compute_rolling_sharpe_from_trades(
        _iter_closed_trades(raw),
        lookback_days=lookback_days,
        min_trades=min_trades,
        now=now,
    )


def default_attribution_path() -> Path:
    """Resolve the standard attribution ledger path for the admin sleeve."""
    data_dir = Path(getattr(ApexConfig, "DATA_DIR", Path("data")))
    return data_dir / "users" / "admin" / "performance_attribution.json"
