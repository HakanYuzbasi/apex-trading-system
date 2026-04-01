"""
monitoring/adaptive_weight_manager.py — Adaptive Signal Weight Manager

Reads Factor IC scores from FactorICTracker and automatically adjusts
blend weights for each signal. Signals proving predictive get upweighted;
signals with near-zero or negative IC get downweighted.

Algorithm:
    1. Get IC scores per signal from FactorICTracker.
    2. Scale: adjusted_weight = base_weight × sigmoid(k × IC)
       where k=4.0 maps IC=0.10 → ×1.20, IC=0.25 → ×1.54, IC=-0.10 → ×0.83
    3. Apply EMA smoothing (α=0.25) to prevent oscillation.
    4. Clamp to [base × MIN_MULT, base × MAX_MULT].
    5. Persist to JSON. Update every INTERVAL_CYCLES cycles.

Wire up:
    weight = self._adaptive_weights.get_weight("god_level",
                  getattr(ApexConfig, "GOD_LEVEL_BLEND_WEIGHT", 0.12))

Config keys:
    ADAPTIVE_WEIGHTS_ENABLED        = True
    ADAPTIVE_WEIGHTS_MIN_MULT       = 0.30   # floor: 30% of base weight
    ADAPTIVE_WEIGHTS_MAX_MULT       = 2.50   # cap:   250% of base weight
    ADAPTIVE_WEIGHTS_EMA_ALPHA      = 0.25   # smoothing (0=frozen, 1=instant)
    ADAPTIVE_WEIGHTS_IC_K           = 4.0    # sigmoid sensitivity to IC
    ADAPTIVE_WEIGHTS_INTERVAL       = 100    # update every N cycles
    ADAPTIVE_WEIGHTS_PERSIST_PATH   = "data/adaptive_weights.json"
"""
from __future__ import annotations

import json
import logging
import math
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Optional

logger = logging.getLogger(__name__)

_DEF = {
    "ADAPTIVE_WEIGHTS_ENABLED":       True,
    "ADAPTIVE_WEIGHTS_MIN_MULT":      0.30,
    "ADAPTIVE_WEIGHTS_MAX_MULT":      2.50,
    "ADAPTIVE_WEIGHTS_EMA_ALPHA":     0.25,
    "ADAPTIVE_WEIGHTS_IC_K":          4.0,
    "ADAPTIVE_WEIGHTS_INTERVAL":      100,
    "ADAPTIVE_WEIGHTS_PERSIST_PATH":  "data/adaptive_weights.json",
}

# Base (config-default) weight for each tracked signal
_BASE_WEIGHTS: Dict[str, float] = {
    "god_level":         0.12,
    "mean_reversion":    0.10,
    "sector_rotation":   0.08,
    "intraday_mr":       0.12,
    "earnings_catalyst": 0.10,
    "primary_signal":    1.00,   # tracked for IC reference but not adjusted
}


def _cfg(key: str):
    try:
        from config import ApexConfig
        v = getattr(ApexConfig, key, None)
        return v if v is not None else _DEF[key]
    except Exception:
        return _DEF[key]


def _sigmoid_scale(ic: float, k: float) -> float:
    """
    Map IC in [-1, 1] → multiplier using sigmoid centred at 0.
    k controls sensitivity:  IC=0 → 1.0,  IC>0 → >1.0,  IC<0 → <1.0
    """
    return 1.0 / (1.0 + math.exp(-k * ic))   # sigmoid(k·IC) in (0,1)
    # Rescale so sigmoid(0)=1.0:  multiply by 2
    # Actually we want: mult = 2 × sigmoid(k × IC)  →  0 to 2, with 0.0→1.0


def _ic_to_multiplier(ic: float, k: float) -> float:
    """IC → weight multiplier.  IC=0 → 1.0, IC=+1 → ~2.0, IC=-1 → ~0.0."""
    return 2.0 / (1.0 + math.exp(-k * ic))


class AdaptiveWeightManager:
    """
    Dynamically adjusts signal blend weights based on rolling Factor IC.
    """

    def __init__(self, persist_path: Optional[str] = None):
        self._persist_path = persist_path or str(_cfg("ADAPTIVE_WEIGHTS_PERSIST_PATH"))
        # Current weights (signal_name → float)
        self._weights: Dict[str, float] = dict(_BASE_WEIGHTS)
        self._cycle_count: int = 0
        self._last_update_ts: str = ""
        self._load()

    # ── Public API ────────────────────────────────────────────────────────────

    def get_weight(self, signal_name: str, default: float) -> float:
        """
        Return the current runtime weight for a signal.
        Falls back to `default` if adaptive weights are disabled or unavailable.
        """
        if not _cfg("ADAPTIVE_WEIGHTS_ENABLED"):
            return default
        return self._weights.get(signal_name, default)

    def maybe_update(self, ic_tracker, cycle: int) -> bool:
        """
        Called every main loop cycle. Updates weights when interval is reached.

        Args:
            ic_tracker: FactorICTracker instance
            cycle: current loop cycle number

        Returns True if weights were updated this call.
        """
        if not _cfg("ADAPTIVE_WEIGHTS_ENABLED"):
            return False
        interval = int(_cfg("ADAPTIVE_WEIGHTS_INTERVAL"))
        if cycle % interval != 0:
            return False
        return self.update_from_tracker(ic_tracker)

    def update_from_tracker(self, ic_tracker) -> bool:
        """
        Force an immediate weight update from a FactorICTracker.
        Returns True on success.
        """
        if ic_tracker is None:
            return False
        try:
            report = ic_tracker.get_report()
        except Exception as e:
            logger.debug("AdaptiveWeights: could not get IC report — %s", e)
            return False

        if not report.signals:
            return False

        # Build IC lookup
        ic_map = {r.signal_name: r.ic for r in report.signals if r.is_reliable}
        if not ic_map:
            return False

        k       = float(_cfg("ADAPTIVE_WEIGHTS_IC_K"))
        alpha   = float(_cfg("ADAPTIVE_WEIGHTS_EMA_ALPHA"))
        min_m   = float(_cfg("ADAPTIVE_WEIGHTS_MIN_MULT"))
        max_m   = float(_cfg("ADAPTIVE_WEIGHTS_MAX_MULT"))

        updated: Dict[str, float] = {}
        for name, base in _BASE_WEIGHTS.items():
            if name == "primary_signal":
                continue  # don't adjust base signal weight
            ic = ic_map.get(name)
            if ic is None:
                continue  # no reliable IC yet — keep current weight

            mult = _ic_to_multiplier(ic, k)
            mult = max(min_m, min(max_m, mult))
            target = base * mult

            # EMA smoothing
            current = self._weights.get(name, base)
            new_w = alpha * target + (1.0 - alpha) * current
            new_w = max(base * min_m, min(base * max_m, new_w))
            updated[name] = round(new_w, 5)

        if updated:
            self._weights.update(updated)
            self._last_update_ts = datetime.now(timezone.utc).isoformat()
            self._save()
            logger.info(
                "AdaptiveWeights updated: %s",
                " | ".join(f"{k}={v:.4f}" for k, v in sorted(updated.items())),
            )
            return True
        return False

    def get_report(self) -> dict:
        """Serialisable state for API/dashboard."""
        base = dict(_BASE_WEIGHTS)
        return {
            "weights": {
                name: {
                    "current": round(self._weights.get(name, base.get(name, 0.0)), 5),
                    "base":    round(base.get(name, 0.0), 5),
                    "mult":    round(
                        self._weights.get(name, base.get(name, 1.0)) /
                        max(base.get(name, 1.0), 1e-9), 4
                    ),
                }
                for name in _BASE_WEIGHTS
                if name != "primary_signal"
            },
            "last_updated": self._last_update_ts,
            "enabled": bool(_cfg("ADAPTIVE_WEIGHTS_ENABLED")),
        }

    # ── Persistence ───────────────────────────────────────────────────────────

    def _load(self) -> None:
        try:
            path = Path(self._persist_path)
            if not path.exists():
                return
            with open(path) as f:
                state = json.load(f)
            loaded = state.get("weights", {})
            for name, val in loaded.items():
                if name in _BASE_WEIGHTS:
                    self._weights[name] = float(val)
            self._last_update_ts = state.get("last_updated", "")
        except Exception as e:
            logger.debug("AdaptiveWeights: could not load state — %s", e)

    def _save(self) -> None:
        try:
            path = Path(self._persist_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            payload = {
                "weights": {k: v for k, v in self._weights.items()},
                "last_updated": self._last_update_ts,
            }
            tmp = path.with_suffix(".tmp")
            with open(tmp, "w") as f:
                json.dump(payload, f, indent=2)
            os.replace(tmp, path)
        except Exception as e:
            logger.debug("AdaptiveWeights: could not save state — %s", e)
