"""
quant_system/risk/factors.py

FactorMonitor — sector-level concentration guardrail.

Categorises every instrument_id into a Sector and tracks gross notional
exposure per sector derived from the live PortfolioLedger.  Entry signals
that would push any single sector above ``sector_concentration_limit`` of
total portfolio equity are vetoed; flatten/rebalance/cover signals are
always permitted.

Persistence
-----------
The sector-map overrides and concentration limit are the only pieces of
state that need to survive restarts (the live exposure is always
recomputed from the ledger, which is persisted separately by StateManager).
Call ``save_state(path)`` on graceful shutdown and ``load_state(path)``
immediately after construction so that any dynamic overrides or limit
changes carry forward.
"""
from __future__ import annotations

import json
import logging
import time
from enum import Enum
from pathlib import Path

from core.symbols import AssetClass, parse_symbol
from quant_system.events.signal import SignalEvent
from quant_system.portfolio.ledger import PortfolioLedger

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Sector taxonomy
# ---------------------------------------------------------------------------

class Sector(str, Enum):
    TECH = "Tech"
    FINANCE = "Finance"
    ENERGY = "Energy"
    HEALTHCARE = "Healthcare"
    CONSUMER = "Consumer"
    INDUSTRIALS = "Industrials"
    MATERIALS = "Materials"
    UTILITIES = "Utilities"
    REAL_ESTATE = "RealEstate"
    COMMUNICATIONS = "Communications"
    RETAIL = "Retail"
    CRYPTO = "Crypto"
    UNKNOWN = "Unknown"


# GICS-aligned mapping for the most liquid US equities and sector ETFs.
# This covers the typical pairs universe; extend via sector_overrides kwarg.
_DEFAULT_SECTOR_MAP: dict[str, Sector] = {
    # ── Technology ────────────────────────────────────────────────────────────
    "AAPL": Sector.TECH,  "MSFT": Sector.TECH,  "NVDA": Sector.TECH,
    "GOOGL": Sector.TECH, "GOOG": Sector.TECH,  "META": Sector.TECH,
    "AVGO": Sector.TECH,  "ORCL": Sector.TECH,  "CRM": Sector.TECH,
    "ADBE": Sector.TECH,  "AMD": Sector.TECH,   "INTC": Sector.TECH,
    "QCOM": Sector.TECH,  "TXN": Sector.TECH,   "AMAT": Sector.TECH,
    "LRCX": Sector.TECH,  "KLAC": Sector.TECH,  "MU": Sector.TECH,
    "NOW": Sector.TECH,   "PANW": Sector.TECH,  "CRWD": Sector.TECH,
    "SNOW": Sector.TECH,  "PLTR": Sector.TECH,  "UBER": Sector.TECH,
    "XLK": Sector.TECH,
    # ── Finance ───────────────────────────────────────────────────────────────
    "JPM": Sector.FINANCE,  "BAC": Sector.FINANCE, "WFC": Sector.FINANCE,
    "GS": Sector.FINANCE,   "MS": Sector.FINANCE,  "C": Sector.FINANCE,
    "BLK": Sector.FINANCE,  "SCHW": Sector.FINANCE,"AXP": Sector.FINANCE,
    "V": Sector.FINANCE,    "MA": Sector.FINANCE,  "PYPL": Sector.FINANCE,
    "COF": Sector.FINANCE,  "USB": Sector.FINANCE, "TFC": Sector.FINANCE,
    "BK": Sector.FINANCE,   "STT": Sector.FINANCE, "SPGI": Sector.FINANCE,
    "ICE": Sector.FINANCE,  "MCO": Sector.FINANCE,
    "XLF": Sector.FINANCE,
    # ── Energy ────────────────────────────────────────────────────────────────
    "XOM": Sector.ENERGY,  "CVX": Sector.ENERGY, "COP": Sector.ENERGY,
    "SLB": Sector.ENERGY,  "EOG": Sector.ENERGY, "MPC": Sector.ENERGY,
    "PSX": Sector.ENERGY,  "VLO": Sector.ENERGY, "HES": Sector.ENERGY,
    "OXY": Sector.ENERGY,  "DVN": Sector.ENERGY, "FANG": Sector.ENERGY,
    "XLE": Sector.ENERGY,
    # ── Healthcare ────────────────────────────────────────────────────────────
    "JNJ": Sector.HEALTHCARE,  "UNH": Sector.HEALTHCARE, "LLY": Sector.HEALTHCARE,
    "PFE": Sector.HEALTHCARE,  "ABBV": Sector.HEALTHCARE,"MRK": Sector.HEALTHCARE,
    "TMO": Sector.HEALTHCARE,  "ABT": Sector.HEALTHCARE, "DHR": Sector.HEALTHCARE,
    "BMY": Sector.HEALTHCARE,  "AMGN": Sector.HEALTHCARE,"GILD": Sector.HEALTHCARE,
    "ISRG": Sector.HEALTHCARE, "SYK": Sector.HEALTHCARE, "MDT": Sector.HEALTHCARE,
    "VRTX": Sector.HEALTHCARE, "BIIB": Sector.HEALTHCARE,"REGN": Sector.HEALTHCARE,
    "XLV": Sector.HEALTHCARE,
    # ── Consumer (Discretionary + Staples) ───────────────────────────────────
    "AMZN": Sector.CONSUMER, "TSLA": Sector.CONSUMER, "HD": Sector.CONSUMER,
    "MCD": Sector.CONSUMER,  "SBUX": Sector.CONSUMER, "NKE": Sector.CONSUMER,
    "LOW": Sector.CONSUMER,  "TJX": Sector.CONSUMER,  "BKNG": Sector.CONSUMER,
    "MAR": Sector.CONSUMER,  "F": Sector.CONSUMER,    "GM": Sector.CONSUMER,
    "PG": Sector.CONSUMER,   "KO": Sector.CONSUMER,   "PEP": Sector.CONSUMER,
    "PM": Sector.CONSUMER,   "MO": Sector.CONSUMER,
    "XLY": Sector.CONSUMER,  "XLP": Sector.CONSUMER,
    # ── Retail ────────────────────────────────────────────────────────────────
    "TGT": Sector.RETAIL,    "WMT": Sector.RETAIL,    "COST": Sector.RETAIL,
    # ── Industrials ───────────────────────────────────────────────────────────
    "GE": Sector.INDUSTRIALS,  "CAT": Sector.INDUSTRIALS, "BA": Sector.INDUSTRIALS,
    "HON": Sector.INDUSTRIALS, "RTX": Sector.INDUSTRIALS, "LMT": Sector.INDUSTRIALS,
    "NOC": Sector.INDUSTRIALS, "GD": Sector.INDUSTRIALS,  "UPS": Sector.INDUSTRIALS,
    "FDX": Sector.INDUSTRIALS, "DE": Sector.INDUSTRIALS,  "MMM": Sector.INDUSTRIALS,
    "EMR": Sector.INDUSTRIALS, "ETN": Sector.INDUSTRIALS, "ITW": Sector.INDUSTRIALS,
    "XLI": Sector.INDUSTRIALS,
    # ── Materials ─────────────────────────────────────────────────────────────
    "LIN": Sector.MATERIALS, "APD": Sector.MATERIALS, "ECL": Sector.MATERIALS,
    "SHW": Sector.MATERIALS, "DD": Sector.MATERIALS,  "NEM": Sector.MATERIALS,
    "FCX": Sector.MATERIALS, "NUE": Sector.MATERIALS,
    "XLB": Sector.MATERIALS,
    # ── Utilities ─────────────────────────────────────────────────────────────
    "NEE": Sector.UTILITIES, "DUK": Sector.UTILITIES, "SO": Sector.UTILITIES,
    "D": Sector.UTILITIES,   "AEP": Sector.UTILITIES, "EXC": Sector.UTILITIES,
    "XLU": Sector.UTILITIES,
    # ── Real Estate ───────────────────────────────────────────────────────────
    "AMT": Sector.REAL_ESTATE,  "PLD": Sector.REAL_ESTATE, "CCI": Sector.REAL_ESTATE,
    "EQIX": Sector.REAL_ESTATE, "PSA": Sector.REAL_ESTATE, "SPG": Sector.REAL_ESTATE,
    "XLRE": Sector.REAL_ESTATE,
    # ── Communications ────────────────────────────────────────────────────────
    "NFLX": Sector.COMMUNICATIONS, "DIS": Sector.COMMUNICATIONS,
    "CMCSA": Sector.COMMUNICATIONS,"T": Sector.COMMUNICATIONS,
    "VZ": Sector.COMMUNICATIONS,   "TMUS": Sector.COMMUNICATIONS,
    "ATVI": Sector.COMMUNICATIONS, "EA": Sector.COMMUNICATIONS,
    "XLC": Sector.COMMUNICATIONS,
    # ── Broad-market ETFs (no single-sector attribution) ──────────────────────
    "SPY": Sector.UNKNOWN, "QQQ": Sector.UNKNOWN, "IWM": Sector.UNKNOWN,
    "DIA": Sector.UNKNOWN, "VTI": Sector.UNKNOWN, "VOO": Sector.UNKNOWN,
}

# Signals with these sides reduce or close risk — always allowed through.
_RISK_REDUCING_SIDES = frozenset({"flatten", "rebalance", "cover"})


# ---------------------------------------------------------------------------
# FactorMonitor
# ---------------------------------------------------------------------------

class FactorMonitor:
    """
    Sector-level concentration guardrail.

    Parameters
    ----------
    ledger:
        The live ``PortfolioLedger`` — used read-only to query current
        positions and reference prices.
    sector_concentration_limit:
        Maximum allowed gross notional in any single sector, expressed as
        a fraction of ``ledger.total_equity()``.  Default: 0.30 (30 %).
    sector_overrides:
        Additional or replacement ``instrument_id → Sector`` mappings that
        extend the built-in GICS table.  Useful for adding thinly-traded
        names or correcting mis-classifications without touching this file.
    """

    def __init__(
        self,
        ledger: PortfolioLedger,
        *,
        sector_concentration_limit: float = 0.30,
        sector_overrides: dict[str, Sector] | None = None,
    ) -> None:
        if not 0.0 < sector_concentration_limit <= 1.0:
            raise ValueError(
                f"sector_concentration_limit must be in (0, 1], got {sector_concentration_limit}"
            )
        self._ledger = ledger
        self._limit = float(sector_concentration_limit)
        self._sector_map: dict[str, Sector] = dict(_DEFAULT_SECTOR_MAP)
        if sector_overrides:
            self._sector_map.update(sector_overrides)
        # Rate-limit VETO warnings to once per 5 min per sector (behaviour unchanged)
        self._veto_last_warned: dict[str, float] = {}
        self._veto_warn_interval = 300.0

    # ── Public accessors ────────────────────────────────────────────────────

    @property
    def sector_concentration_limit(self) -> float:
        """Maximum allowed sector gross notional as a fraction of equity."""
        return self._limit

    def sector_for(self, instrument_id: str) -> Sector:
        """
        Return the ``Sector`` for *instrument_id*.

        Crypto instruments always resolve to ``Sector.CRYPTO`` regardless
        of the sector map.  Unmapped equities return ``Sector.UNKNOWN``.
        """
        try:
            parsed = parse_symbol(instrument_id)
        except Exception:
            return Sector.UNKNOWN

        if parsed.asset_class == AssetClass.CRYPTO:
            return Sector.CRYPTO

        # Try bare ticker first, then full instrument_id string.
        sector = self._sector_map.get(parsed.base) or self._sector_map.get(instrument_id)
        if sector is not None:
            return sector

        logger.debug(
            "FactorMonitor: no sector mapping for '%s' (base='%s') — defaulting to UNKNOWN",
            instrument_id,
            parsed.base,
        )
        return Sector.UNKNOWN

    def sector_gross_notional(self, sector: Sector) -> float:
        """
        Gross notional (|qty| × price, summed over longs and shorts) for
        *sector*.  Returns 0.0 when there are no open positions in the sector
        or reference prices are unavailable.
        """
        total = 0.0
        for instrument_id, position in self._ledger.positions.items():
            if abs(position.quantity) < 1e-12:
                continue
            if self.sector_for(instrument_id) != sector:
                continue
            price = self._ledger.get_reference_price(instrument_id) or 0.0
            total += abs(position.quantity) * price
        return total

    def sector_net_notional(self, sector: Sector) -> float:
        """
        Signed net notional (long > 0, short < 0) for *sector*.
        """
        total = 0.0
        for instrument_id, position in self._ledger.positions.items():
            if abs(position.quantity) < 1e-12:
                continue
            if self.sector_for(instrument_id) != sector:
                continue
            price = self._ledger.get_reference_price(instrument_id) or 0.0
            total += position.quantity * price
        return total

    def sector_exposures(self) -> dict[str, float]:
        """
        Snapshot of gross notional per sector for all sectors with open
        positions.  Keyed by ``Sector.value`` strings for easy serialisation.
        """
        result: dict[str, float] = {}
        for sector in Sector:
            gross = self.sector_gross_notional(sector)
            if gross > 0.0:
                result[sector.value] = round(gross, 2)
        return result

    # ── Signal veto ─────────────────────────────────────────────────────────

    def check_signal(
        self,
        event: SignalEvent,
        reference_price: float,
    ) -> bool:
        """
        Return **True** if the signal is permitted, **False** if it would
        breach the sector concentration limit (and should be vetoed).

        Rules
        -----
        * ``flatten`` / ``rebalance`` / ``cover`` sides are always allowed —
          they reduce risk.
        * Instruments in ``Sector.UNKNOWN`` are always allowed — we cannot
          reliably constrain what we cannot classify.
        * For all other entry signals the projected gross notional
          (current + incoming) is checked against the limit.
        """
        if event.side in _RISK_REDUCING_SIDES:
            return True

        sector = self.sector_for(event.instrument_id)
        if sector == Sector.UNKNOWN:
            return True

        total_equity = self._ledger.total_equity()
        if total_equity <= 0.0:
            # Degenerate state — don't block; let other guards handle it.
            return True

        limit_notional = total_equity * self._limit
        current_gross = self.sector_gross_notional(sector)
        incoming_notional = self._estimate_signal_notional(event, reference_price)

        projected_gross = current_gross + incoming_notional
        if projected_gross <= limit_notional + 1e-2:
            return True

        now = time.monotonic()
        sector_key = sector.value
        if now - self._veto_last_warned.get(sector_key, 0.0) >= self._veto_warn_interval:
            logger.warning(
                "FactorMonitor VETO | sector=%s at %.1f%% of equity=$%.0f "
                "(limit=%.0f%%) — suppressing per-signal logs for 5 min",
                sector.value,
                100.0 * projected_gross / total_equity,
                total_equity,
                self._limit * 100.0,
            )
            self._veto_last_warned[sector_key] = now
        else:
            logger.debug(
                "FactorMonitor VETO | instrument=%s sector=%s projected=$%.0f limit=$%.0f",
                event.instrument_id, sector.value, projected_gross, limit_notional,
            )
        return False

    # ── Persistence ─────────────────────────────────────────────────────────

    def save_state(self, path: Path | str) -> None:
        """
        Persist the concentration limit and any non-default sector overrides
        to *path* as JSON.  Call on graceful shutdown or after dynamic
        adjustments so the next startup restores the same configuration.
        """
        overrides = {
            ticker: sector.value
            for ticker, sector in self._sector_map.items()
            if _DEFAULT_SECTOR_MAP.get(ticker) != sector
        }
        payload = {
            "sector_concentration_limit": self._limit,
            "sector_overrides": overrides,
        }
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(json.dumps(payload, indent=2), encoding="utf-8")
        logger.debug("FactorMonitor state saved to %s", path)

    def load_state(self, path: Path | str) -> None:
        """
        Restore the concentration limit and sector overrides from *path*.
        Silently no-ops if the file does not exist.
        """
        p = Path(path)
        if not p.exists():
            return
        try:
            payload = json.loads(p.read_text(encoding="utf-8"))
            new_limit = float(
                payload.get("sector_concentration_limit", self._limit)
            )
            if 0.0 < new_limit <= 1.0:
                self._limit = new_limit
            for ticker, sector_value in payload.get("sector_overrides", {}).items():
                try:
                    self._sector_map[str(ticker)] = Sector(sector_value)
                except ValueError:
                    logger.warning(
                        "FactorMonitor: unrecognised sector value '%s' for '%s'; skipping",
                        sector_value,
                        ticker,
                    )
            logger.info(
                "FactorMonitor state restored from %s (limit=%.0f%%)",
                path,
                self._limit * 100.0,
            )
        except Exception:
            logger.exception("FactorMonitor: failed to load state from %s", path)

    # ── Helpers ─────────────────────────────────────────────────────────────

    @staticmethod
    def _estimate_signal_notional(event: SignalEvent, reference_price: float) -> float:
        """Dollar notional of the incoming signal."""
        tv = abs(float(event.target_value))
        if event.target_type == "notional":
            return tv
        if event.target_type == "units" and reference_price > 0:
            return tv * reference_price
        # "weight" signals are a fraction of cash — cash is not available here;
        # return 0 so weight-type signals are never vetoed by the factor monitor
        # (they are already bounded by the RiskManager's margin check).
        return 0.0
