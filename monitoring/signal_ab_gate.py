"""
monitoring/signal_ab_gate.py

Thompson-Sampling A/B gate for signal weight variants.

Prevents bad model retrains from reaching production without statistical
confirmation. Workflow:

1. Register a challenger variant with ``register_challenger(weights, name)``.
2. Each trade call ``should_use_challenger()`` — Thompson Sampling picks the
   better posterior at random.
3. After the trade closes, call ``record_outcome(used_challenger, win)``.
4. Periodically call ``maybe_promote()`` — returns the new weights if the
   challenger has passed significance and 48-hour shadow period, else None.
5. If challenger IC drops, ``maybe_rollback()`` reverts to control.

The Beta(α, β) posterior represents the win-rate distribution.
Starting prior is Beta(1, 1) = uniform.
"""
from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional
import numpy as np


_DEFAULT_MIN_TRADES: int = 30
_DEFAULT_PROMOTION_PROB: float = 0.95   # P(challenger > control) threshold
_DEFAULT_ROLLBACK_DROP: float = 0.05    # absolute win-rate drop triggers rollback
_DEFAULT_SHADOW_HOURS: float = 48.0


# ── Variant state ─────────────────────────────────────────────────────────────

@dataclass
class VariantState:
    name: str
    weights: Dict[str, float]       # signal component weights
    alpha: float = 1.0              # Beta posterior alpha (wins + prior)
    beta_: float = 1.0              # Beta posterior beta (losses + prior)
    n_trades: int = 0
    created_at: float = field(default_factory=time.time)
    promoted_at: Optional[float] = None

    @property
    def win_rate_mean(self) -> float:
        return self.alpha / (self.alpha + self.beta_)

    def sample(self) -> float:
        """Draw one sample from the Beta posterior."""
        return float(np.random.beta(
            max(0.01, self.alpha),
            max(0.01, self.beta_),
        ))

    def record(self, win: bool) -> None:
        if win:
            self.alpha += 1.0
        else:
            self.beta_ += 1.0
        self.n_trades += 1

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "weights": self.weights,
            "alpha": self.alpha,
            "beta_": self.beta_,
            "n_trades": self.n_trades,
            "created_at": self.created_at,
            "promoted_at": self.promoted_at,
            "win_rate_mean": round(self.win_rate_mean, 4),
        }


# ── Gate ──────────────────────────────────────────────────────────────────────

class SignalABGate:
    """Thompson-Sampling A/B gate for signal component weight variants.

    Parameters
    ----------
    min_trades : int
        Minimum trades on challenger before promotion is considered.
    promotion_prob : float
        P(challenger > control) required for promotion.
    rollback_ic_drop : float
        If challenger win rate drops this much below control, auto-rollback.
    shadow_hours : float
        Challenger must have been running for this many hours before promotion.
    data_dir : Path | None
        If set, state is persisted to ``signal_ab_gate.json``.
    """

    def __init__(
        self,
        min_trades: int = _DEFAULT_MIN_TRADES,
        promotion_prob: float = _DEFAULT_PROMOTION_PROB,
        rollback_ic_drop: float = _DEFAULT_ROLLBACK_DROP,
        shadow_hours: float = _DEFAULT_SHADOW_HOURS,
        data_dir: Optional[Path] = None,
    ) -> None:
        self._min_trades = min_trades
        self._promotion_prob = promotion_prob
        self._rollback_drop = rollback_ic_drop
        self._shadow_hours = shadow_hours
        self._data_dir = Path(data_dir) if data_dir else None

        self._control: Optional[VariantState] = None
        self._challenger: Optional[VariantState] = None
        self._promotion_history: List[dict] = []

        if self._data_dir:
            self._load()

    # ── Public API ────────────────────────────────────────────────────────────

    def register_challenger(
        self,
        weights: Dict[str, float],
        name: str = "challenger",
    ) -> None:
        """Register a new challenger variant to A/B test against current control."""
        if self._control is None:
            # First registration — becomes control, no challenger yet
            self._control = VariantState(name="control", weights=dict(weights))
            self._persist()
            return
        self._challenger = VariantState(name=name, weights=dict(weights))
        self._persist()

    def set_control_weights(self, weights: Dict[str, float]) -> None:
        """Overwrite the control weights (e.g. on initial startup)."""
        if self._control is None:
            self._control = VariantState(name="control", weights=dict(weights))
        else:
            self._control.weights = dict(weights)
        self._persist()

    def should_use_challenger(self) -> bool:
        """Thompson Sampling: True if challenger draws a higher sample."""
        if self._challenger is None or self._control is None:
            return False
        return self._challenger.sample() > self._control.sample()

    def record_outcome(self, used_challenger: bool, win: bool) -> None:
        """Record a trade outcome for the variant that was used."""
        if used_challenger and self._challenger is not None:
            self._challenger.record(win)
        elif not used_challenger and self._control is not None:
            self._control.record(win)
        self._persist()

    def maybe_promote(self) -> Optional[Dict[str, float]]:
        """Promote challenger to control if it passes all gates.

        Returns the new control weights if promoted, else None.
        """
        if self._challenger is None or self._control is None:
            return None
        if self._challenger.n_trades < self._min_trades:
            return None
        shadow_elapsed = (time.time() - self._challenger.created_at) / 3600.0
        if shadow_elapsed < self._shadow_hours:
            return None
        if self._p_challenger_better() < self._promotion_prob:
            return None

        # Promote
        new_weights = dict(self._challenger.weights)
        self._promotion_history.append({
            "ts": time.time(),
            "old_weights": self._control.weights,
            "new_weights": new_weights,
            "challenger_win_rate": round(self._challenger.win_rate_mean, 4),
            "control_win_rate": round(self._control.win_rate_mean, 4),
            "n_challenger_trades": self._challenger.n_trades,
            "p_better": round(self._p_challenger_better(), 4),
        })
        self._control = VariantState(
            name="control",
            weights=new_weights,
            alpha=self._challenger.alpha,
            beta_=self._challenger.beta_,
            n_trades=self._challenger.n_trades,
            promoted_at=time.time(),
        )
        self._challenger = None
        self._persist()
        return new_weights

    def maybe_rollback(self) -> bool:
        """Rollback challenger if its win rate has dropped too far below control.

        Returns True if rollback occurred.
        """
        if self._challenger is None or self._control is None:
            return False
        if self._challenger.n_trades < max(10, self._min_trades // 3):
            return False
        drop = self._control.win_rate_mean - self._challenger.win_rate_mean
        if drop >= self._rollback_drop:
            self._challenger = None
            self._persist()
            return True
        return False

    @property
    def active_weights(self) -> Optional[Dict[str, float]]:
        """Current control weights, or None if not set."""
        return self._control.weights if self._control else None

    def get_status(self) -> dict:
        return {
            "control": self._control.to_dict() if self._control else None,
            "challenger": self._challenger.to_dict() if self._challenger else None,
            "p_challenger_better": round(self._p_challenger_better(), 4)
            if self._challenger and self._control else None,
            "promotions": len(self._promotion_history),
            "promotion_history": self._promotion_history[-5:],
            "thresholds": {
                "min_trades": self._min_trades,
                "promotion_prob": self._promotion_prob,
                "rollback_drop": self._rollback_drop,
                "shadow_hours": self._shadow_hours,
            },
        }

    # ── Internals ─────────────────────────────────────────────────────────────

    def _p_challenger_better(self) -> float:
        """Monte Carlo estimate of P(challenger > control) using 4 000 samples."""
        if self._challenger is None or self._control is None:
            return 0.0
        n = 4_000
        ctrl = np.random.beta(
            max(0.01, self._control.alpha),
            max(0.01, self._control.beta_),
            size=n,
        )
        chall = np.random.beta(
            max(0.01, self._challenger.alpha),
            max(0.01, self._challenger.beta_),
            size=n,
        )
        return float(np.mean(chall > ctrl))

    # ── Persistence ───────────────────────────────────────────────────────────

    def _path(self) -> Path:
        assert self._data_dir is not None
        return self._data_dir / "signal_ab_gate.json"

    def _persist(self) -> None:
        if self._data_dir is None:
            return
        try:
            self._data_dir.mkdir(parents=True, exist_ok=True)
            state = {
                "control": self._control.to_dict() if self._control else None,
                "challenger": self._challenger.to_dict() if self._challenger else None,
                "promotion_history": self._promotion_history[-20:],
            }
            tmp = self._path().with_suffix(".json.tmp")
            tmp.write_text(json.dumps(state, indent=2), encoding="utf-8")
            tmp.replace(self._path())
        except Exception:
            pass

    def _load(self) -> None:
        try:
            p = self._path()
            if not p.exists():
                return
            raw = json.loads(p.read_text(encoding="utf-8"))
            if raw.get("control"):
                c = raw["control"]
                self._control = VariantState(
                    name=c.get("name", "control"),
                    weights=c.get("weights", {}),
                    alpha=float(c.get("alpha", 1.0)),
                    beta_=float(c.get("beta_", 1.0)),
                    n_trades=int(c.get("n_trades", 0)),
                    created_at=float(c.get("created_at", time.time())),
                    promoted_at=c.get("promoted_at"),
                )
            if raw.get("challenger"):
                c = raw["challenger"]
                self._challenger = VariantState(
                    name=c.get("name", "challenger"),
                    weights=c.get("weights", {}),
                    alpha=float(c.get("alpha", 1.0)),
                    beta_=float(c.get("beta_", 1.0)),
                    n_trades=int(c.get("n_trades", 0)),
                    created_at=float(c.get("created_at", time.time())),
                    promoted_at=c.get("promoted_at"),
                )
            self._promotion_history = list(raw.get("promotion_history", []))
        except Exception:
            pass
