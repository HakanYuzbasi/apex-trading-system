"""
execution/cost_model.py — Unified Fee & Slippage Model

Single source of truth for transaction cost assumptions across live
connectors and the backtester. Keeps the a-priori cost estimate identical
in both paths so simulated PnL does not diverge from realised PnL.

Two pure functions:
  - ``fee_bps(asset_class, is_maker)``          → bps fee on the fill leg
  - ``slippage_bps(notional_usd, adv_usd)``     → volume-adjusted slippage

All thresholds are sourced from ``ApexConfig`` so the whole cost grid can
be retuned without editing code. Every bps is returned as a float in
basis points (1 bp = 0.01%).
"""
from __future__ import annotations

from typing import Optional

from config import ApexConfig


_EQUITY_ALIASES = frozenset({"equity", "stock", "stocks", "equities", "us_equity", "etf"})
_FX_ALIASES = frozenset({"fx", "forex", "currency", "cash_fx"})
_CRYPTO_ALIASES = frozenset({"crypto", "cryptocurrency", "coin", "digital_asset"})


def _normalize_asset_class(asset_class: str) -> str:
    """
    Normalise free-form asset_class strings to one of
    ``{"equity", "fx", "crypto"}``.

    Args:
        asset_class: Caller-supplied asset label (case-insensitive).

    Returns:
        Canonical asset-class string; ``"equity"`` when no alias matches.
    """
    if not asset_class:
        return "equity"
    key = str(asset_class).strip().lower()
    if key in _EQUITY_ALIASES:
        return "equity"
    if key in _FX_ALIASES:
        return "fx"
    if key in _CRYPTO_ALIASES:
        return "crypto"
    return "equity"


def fee_bps(asset_class: str, is_maker: bool) -> float:
    """
    Return fee in basis points for one side of a fill.

    Maker fills are taken to be limit orders that rested on the book
    (non-aggressive); taker fills are market / marketable-limit / IOC /
    aggressive limits that crossed the spread. On exchanges with
    rebate-maker pricing a negative fee value is valid.

    Args:
        asset_class: ``equity`` / ``fx`` / ``crypto`` (case-insensitive).
        is_maker: True when the order rested before fill (post-only or
            joined the touch without crossing); False otherwise.

    Returns:
        Fee as a float in basis points (not a fraction). Multiply by
        ``notional / 10_000`` to get dollar cost.

    Raises:
        TypeError: If ``asset_class`` is not a string.
    """
    if not isinstance(asset_class, str):
        raise TypeError(f"asset_class must be str, got {type(asset_class).__name__}")

    canon = _normalize_asset_class(asset_class)
    if canon == "crypto":
        cfg_key = "CRYPTO_MAKER_FEE_BPS" if is_maker else "CRYPTO_TAKER_FEE_BPS"
    elif canon == "fx":
        cfg_key = "FX_MAKER_FEE_BPS" if is_maker else "FX_TAKER_FEE_BPS"
    else:
        cfg_key = "EQUITY_MAKER_FEE_BPS" if is_maker else "EQUITY_TAKER_FEE_BPS"
    return float(getattr(ApexConfig, cfg_key))


def slippage_bps(
    notional_usd: float,
    adv_usd: Optional[float] = None,
) -> float:
    """
    Volume-adjusted a-priori slippage estimate in basis points.

    Model:
        ``bps = BASE_SLIPPAGE_BPS * (notional / adv) ** SLIPPAGE_EXPONENT``

    Small orders on liquid symbols approach the base rate; large orders
    on thin symbols scale up by the ratio's exponent. Clamped to
    ``[SLIPPAGE_MIN_BPS, SLIPPAGE_MAX_BPS]`` so extreme ratios cannot
    produce nonsensical forecasts.

    Args:
        notional_usd: Order notional (``qty × price``) in USD.
            Must be ``>= 0``; 0 returns the minimum slippage.
        adv_usd: Average daily volume in USD. When ``None`` or
            non-positive, falls back to ``SLIPPAGE_ADV_FALLBACK_USD``.

    Returns:
        Slippage as a float in basis points, clamped to the config range.

    Raises:
        ValueError: If ``notional_usd`` is negative or non-finite.
    """
    notional = float(notional_usd)
    if notional < 0 or notional != notional:
        raise ValueError(f"notional_usd must be non-negative finite, got {notional!r}")

    base = float(ApexConfig.BASE_SLIPPAGE_BPS)
    exponent = float(ApexConfig.SLIPPAGE_EXPONENT)
    min_bps = float(ApexConfig.SLIPPAGE_MIN_BPS)
    max_bps = float(ApexConfig.SLIPPAGE_MAX_BPS)

    if notional == 0.0:
        return min_bps

    if adv_usd is None or not (adv_usd == adv_usd) or float(adv_usd) <= 0.0:
        adv = float(ApexConfig.SLIPPAGE_ADV_FALLBACK_USD)
    else:
        adv = float(adv_usd)

    ratio = notional / adv
    raw = base * (ratio ** exponent)
    if raw < min_bps:
        return min_bps
    if raw > max_bps:
        return max_bps
    return float(raw)


def expected_cost_bps(
    asset_class: str,
    is_maker: bool,
    notional_usd: float,
    adv_usd: Optional[float] = None,
) -> float:
    """
    Combined a-priori cost estimate (fee + slippage) in basis points.

    Args:
        asset_class: Canonical or alias asset label.
        is_maker: True for passive limit orders, False for aggressive.
        notional_usd: Order notional in USD (``>= 0``).
        adv_usd: Symbol ADV in USD, or ``None`` for the configured fallback.

    Returns:
        Sum of fee and slippage in basis points. Aggressive (taker) orders
        pay both legs; passive (maker) fills at the touch still bear
        slippage because the midpoint crossed before the fill.

    Raises:
        ValueError: Propagated from :func:`slippage_bps`.
        TypeError: Propagated from :func:`fee_bps`.
    """
    f = fee_bps(asset_class, is_maker)
    s = slippage_bps(notional_usd, adv_usd)
    # Maker at the touch still has half-spread exposure; taker crosses.
    # We keep slippage weighted at 1.0 for taker and 0.5 for maker — the
    # maker fill captures half the spread on the *favourable* side.
    slip_weight = 0.5 if is_maker else 1.0
    return float(f + s * slip_weight)
