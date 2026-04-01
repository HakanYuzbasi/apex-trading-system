"""
risk/liquidity_sizer.py

Real-time liquidity-adjusted position sizing and portfolio concentration heat.

Three components
----------------
1. **Spread penalty multiplier** — if bid/ask spread for a symbol exceeds its
   baseline (rolling 50-cycle median), position size is reduced.
   Formula:  mult = 1 / (1 + k × max(0, spread_ratio - 1))
   where spread_ratio = current_spread / baseline_spread, k = 1.5 (tunable).

2. **Concentration heat** — fraction of total notional in the top-N symbols,
   and per-symbol weight.  Alerts when top-5 > concentration_cap.

3. **Correlated stress exposure** — given a scenario drop %, what is the
   expected simultaneous loss across correlated positions?
   Approximated as: sum(notional_i × drop_pct) for correlated subsets.

Usage in execution_loop::

    # init
    self._liquidity_sizer = LiquiditySizer(data_dir=ApexConfig.DATA_DIR)

    # each cycle, after fetching price/spread
    self._liquidity_sizer.record_spread(symbol, spread_bps)

    # at sizing step, multiply final size
    liq_mult = self._liquidity_sizer.get_liquidity_multiplier(symbol, current_spread_bps)
    final_qty = base_qty * liq_mult

    # for portfolio heat dashboard
    heat = self._liquidity_sizer.get_concentration_heat(positions)
"""
from __future__ import annotations

import json
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np


_DEFAULT_SPREAD_WINDOW = 50       # rolling cycles for baseline
_DEFAULT_PENALTY_K = 1.5          # spread penalty coefficient
_DEFAULT_MULT_FLOOR = 0.30        # minimum liquidity multiplier
_DEFAULT_CONCENTRATION_CAP = 0.65 # top-5 share threshold
_DEFAULT_STRESS_DROP = 0.05       # 5% scenario drop


@dataclass
class ConcentrationHeat:
    """Portfolio concentration snapshot."""
    top5_share: float              # fraction of total in top-5 symbols
    top1_share: float              # fraction in single largest position
    top5_symbols: List[str]
    concentration_breached: bool   # top5_share > cap
    by_symbol: Dict[str, float]    # symbol → weight

    def to_dict(self) -> dict:
        return {
            "top5_share": round(self.top5_share, 4),
            "top1_share": round(self.top1_share, 4),
            "top5_symbols": self.top5_symbols,
            "concentration_breached": self.concentration_breached,
            "by_symbol": {k: round(v, 4) for k, v in self.by_symbol.items()},
        }


class LiquiditySizer:
    """Real-time liquidity sizing and concentration monitor.

    Parameters
    ----------
    data_dir : Path | None
        Persistence location for spread history.
    spread_window : int
        Number of cycles kept for per-symbol spread baseline.
    penalty_k : float
        Penalty coefficient: larger → harsher size reduction on wide spreads.
    mult_floor : float
        Minimum multiplier applied (never reduces size below this fraction).
    concentration_cap : float
        top-5 share threshold that triggers a ``concentration_breached`` flag.
    """

    def __init__(
        self,
        data_dir: Optional[Path] = None,
        spread_window: int = _DEFAULT_SPREAD_WINDOW,
        penalty_k: float = _DEFAULT_PENALTY_K,
        mult_floor: float = _DEFAULT_MULT_FLOOR,
        concentration_cap: float = _DEFAULT_CONCENTRATION_CAP,
    ) -> None:
        self._data_dir = Path(data_dir) if data_dir else None
        self._spread_window = spread_window
        self._penalty_k = penalty_k
        self._mult_floor = mult_floor
        self._concentration_cap = concentration_cap

        # symbol → rolling spread deque
        self._spreads: Dict[str, deque] = defaultdict(
            lambda: deque(maxlen=self._spread_window)
        )

        if self._data_dir:
            self._load()

    # ── Spread API ────────────────────────────────────────────────────────────

    def record_spread(self, symbol: str, spread_bps: float) -> None:
        """Record a new spread observation for a symbol."""
        if spread_bps > 0:
            self._spreads[symbol].append(float(spread_bps))

    def get_liquidity_multiplier(
        self,
        symbol: str,
        current_spread_bps: float,
    ) -> float:
        """Return [mult_floor, 1.0] sizing multiplier based on spread vs baseline.

        If fewer than 5 observations exist, returns 1.0 (no adjustment).
        """
        hist = list(self._spreads.get(symbol, []))
        if len(hist) < 5 or current_spread_bps <= 0:
            return 1.0

        baseline = float(np.median(hist))
        if baseline < 0.01:
            return 1.0

        spread_ratio = current_spread_bps / baseline
        if spread_ratio <= 1.0:
            return 1.0

        mult = 1.0 / (1.0 + self._penalty_k * (spread_ratio - 1.0))
        return float(max(self._mult_floor, min(1.0, mult)))

    def get_baseline_spread(self, symbol: str) -> Optional[float]:
        hist = list(self._spreads.get(symbol, []))
        return float(np.median(hist)) if hist else None

    # ── Concentration API ─────────────────────────────────────────────────────

    def get_concentration_heat(
        self,
        positions: Dict[str, float],  # symbol → notional value (USD)
    ) -> ConcentrationHeat:
        """Compute portfolio concentration metrics."""
        if not positions:
            return ConcentrationHeat(
                top5_share=0.0,
                top1_share=0.0,
                top5_symbols=[],
                concentration_breached=False,
                by_symbol={},
            )
        total = sum(abs(v) for v in positions.values())
        if total < 0.01:
            return ConcentrationHeat(
                top5_share=0.0, top1_share=0.0,
                top5_symbols=[], concentration_breached=False, by_symbol={},
            )

        by_sym = {sym: abs(v) / total for sym, v in positions.items()}
        sorted_syms = sorted(by_sym.items(), key=lambda x: x[1], reverse=True)
        top5 = sorted_syms[:5]
        top5_share = sum(v for _, v in top5)
        top1_share = sorted_syms[0][1] if sorted_syms else 0.0

        return ConcentrationHeat(
            top5_share=round(top5_share, 4),
            top1_share=round(top1_share, 4),
            top5_symbols=[s for s, _ in top5],
            concentration_breached=top5_share > self._concentration_cap,
            by_symbol=dict(sorted_syms),
        )

    # ── Stress exposure API ───────────────────────────────────────────────────

    def get_stress_exposure(
        self,
        positions: Dict[str, float],  # symbol → notional (positive = long)
        scenario_drop_pct: float = _DEFAULT_STRESS_DROP,
        crypto_symbols: Optional[List[str]] = None,
    ) -> dict:
        """Estimate simultaneous loss if assets drop by scenario_drop_pct.

        Treats equities and crypto as separate correlated baskets.
        """
        if not positions:
            return {"equity_loss": 0.0, "crypto_loss": 0.0, "total_loss": 0.0}

        crypto_set = set(crypto_symbols or [])
        equity_long = sum(
            v for sym, v in positions.items()
            if sym not in crypto_set and v > 0
        )
        crypto_long = sum(
            v for sym, v in positions.items()
            if sym in crypto_set and v > 0
        )

        drop = abs(float(scenario_drop_pct))
        equity_loss = equity_long * drop
        crypto_loss = crypto_long * drop

        return {
            "scenario_drop_pct": round(drop, 4),
            "equity_long_notional": round(equity_long, 2),
            "crypto_long_notional": round(crypto_long, 2),
            "equity_loss": round(equity_loss, 2),
            "crypto_loss": round(crypto_loss, 2),
            "total_loss": round(equity_loss + crypto_loss, 2),
        }

    def get_report(self, positions: Dict[str, float]) -> dict:
        heat = self.get_concentration_heat(positions)
        stress = self.get_stress_exposure(positions)
        spread_coverage = sum(
            1 for sym in positions if len(self._spreads.get(sym, [])) >= 5
        )
        return {
            "concentration": heat.to_dict(),
            "stress_5pct": stress,
            "spread_baseline_coverage": spread_coverage,
            "total_positions": len(positions),
        }

    # ── Persistence ───────────────────────────────────────────────────────────

    def _path(self) -> Path:
        assert self._data_dir is not None
        return self._data_dir / "liquidity_sizer_state.json"

    def _load(self) -> None:
        try:
            p = self._path()
            if not p.exists():
                return
            raw = json.loads(p.read_text(encoding="utf-8"))
            for sym, values in raw.get("spreads", {}).items():
                dq = deque(maxlen=self._spread_window)
                dq.extend(float(v) for v in values)
                self._spreads[sym] = dq
        except Exception:
            pass

    def save(self) -> None:
        """Persist spread history (call periodically from execution loop)."""
        if self._data_dir is None:
            return
        try:
            self._data_dir.mkdir(parents=True, exist_ok=True)
            state = {
                "spreads": {sym: list(dq) for sym, dq in self._spreads.items()},
                "saved_at": time.time(),
            }
            tmp = self._path().with_suffix(".json.tmp")
            tmp.write_text(json.dumps(state), encoding="utf-8")
            tmp.replace(self._path())
        except Exception:
            pass
