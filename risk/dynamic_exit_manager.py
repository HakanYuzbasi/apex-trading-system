"""
risk/dynamic_exit_manager.py - Dynamic Exit Strategy Manager

Adapts exit thresholds based on:
- Market regime (bull/bear/volatile/neutral)
- VIX level
- Signal strength at entry
- Position P&L trajectory
- Holding time
- Momentum decay

Exit philosophy:
- Cut losers fast, let winners run
- Tighter stops in volatile markets
- Wider targets in trending markets
- Time decay on stale positions
"""

import json
import os
import numpy as np
from collections import deque
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timezone
from dataclasses import dataclass
from enum import Enum
import logging

from config import ApexConfig

logger = logging.getLogger(__name__)


# Regime-name alias map — the broader risk stack uses multiple vocabularies
# for the volatility regime (``high_volatility`` in this module, ``volatile``
# in adaptive_atr_stops). Normalise before lookup so regime routing is
# deterministic regardless of which producer labelled the regime.
_REGIME_ALIASES: Dict[str, str] = {
    "volatile": "high_volatility",
    "high_vol": "high_volatility",
    "highvol": "high_volatility",
    "crisis": "high_volatility",
    "risk_off": "bear",
    "risk_on": "bull",
}


def _normalize_regime(regime: Optional[str]) -> str:
    """Return the canonical regime key, falling back to ``'neutral'``."""
    if not regime:
        return "neutral"
    key = str(regime).strip().lower()
    return _REGIME_ALIASES.get(key, key)


def _now_tzaware(ref: Optional[datetime] = None) -> datetime:
    """
    Return a timezone-aware ``datetime.now()`` that is *comparison-safe*
    with ``ref`` (matching naive/aware-ness). Entry times recorded by the
    execution loop can be either naive UTC or aware UTC depending on
    code path — comparing the two raises ``TypeError``.

    Args:
        ref: Reference datetime whose tzinfo drives the result.

    Returns:
        A ``datetime`` whose ``tzinfo`` matches ``ref`` (naive if ``ref``
        is naive, UTC-aware if ``ref`` is aware or ``None``).
    """
    if ref is not None and ref.tzinfo is None:
        # Match naive behaviour — strip tz from now()
        return datetime.utcnow()
    return datetime.now(timezone.utc)


def _safe_elapsed_days(entry_time: datetime) -> int:
    """Holding-period in whole days, tz-safe; floors negatives at 0."""
    now = _now_tzaware(entry_time)
    delta = now - entry_time
    return max(0, int(delta.days))


def _safe_elapsed_hours(entry_time: datetime) -> float:
    """Holding-period in hours, tz-safe; floors negatives at 0."""
    now = _now_tzaware(entry_time)
    delta = now - entry_time
    return max(0.0, float(delta.total_seconds()) / 3600.0)


# ── Regime-multiplier parsing (config-driven) ────────────────────────────────
# Each regime string is "stop,target,hold,signal" (four floats). Invalid or
# missing strings fall back to the hard-coded defaults below — so the system
# is always safe even when config is misconfigured.
_REGIME_DEFAULTS: Dict[str, Dict[str, float]] = {
    "strong_bull":     {"stop_mult": 1.2, "target_mult": 1.5, "hold_mult": 1.5, "signal_mult": 0.8},
    "bull":            {"stop_mult": 1.1, "target_mult": 1.3, "hold_mult": 1.2, "signal_mult": 0.9},
    "neutral":         {"stop_mult": 0.9, "target_mult": 0.8, "hold_mult": 0.7, "signal_mult": 1.2},
    "bear":            {"stop_mult": 0.8, "target_mult": 1.2, "hold_mult": 0.8, "signal_mult": 1.1},
    "strong_bear":     {"stop_mult": 0.7, "target_mult": 1.4, "hold_mult": 0.6, "signal_mult": 1.3},
    "high_volatility": {"stop_mult": 0.6, "target_mult": 0.7, "hold_mult": 0.5, "signal_mult": 1.5},
}

# Map canonical regime → matching ApexConfig attribute suffix.
_REGIME_CONFIG_KEYS: Dict[str, str] = {
    "strong_bull":     "EXIT_REGIME_STRONG_BULL",
    "bull":            "EXIT_REGIME_BULL",
    "neutral":         "EXIT_REGIME_NEUTRAL",
    "bear":            "EXIT_REGIME_BEAR",
    "strong_bear":     "EXIT_REGIME_STRONG_BEAR",
    "high_volatility": "EXIT_REGIME_HIGH_VOLATILITY",
}


def _parse_regime_quad(raw: Optional[str]) -> Optional[Dict[str, float]]:
    """
    Parse a ``"stop,target,hold,signal"`` string into a regime-multiplier
    dict. Returns ``None`` if parsing fails so the caller can fall back.

    Args:
        raw: Comma-separated four-value string sourced from
            ``ApexConfig.EXIT_REGIME_*``.

    Returns:
        ``{"stop_mult", "target_mult", "hold_mult", "signal_mult"}``
        or ``None``.
    """
    if not raw:
        return None
    try:
        parts = [p.strip() for p in str(raw).split(",")]
        if len(parts) != 4:
            return None
        stop, target, hold, signal = (float(p) for p in parts)
        if any(not np.isfinite(v) for v in (stop, target, hold, signal)):
            return None
        if stop <= 0 or target <= 0 or hold <= 0 or signal <= 0:
            return None
        return {
            "stop_mult": stop,
            "target_mult": target,
            "hold_mult": hold,
            "signal_mult": signal,
        }
    except Exception:
        return None


def _build_regime_adjustments() -> Dict[str, Dict[str, float]]:
    """
    Assemble the full ``REGIME_ADJUSTMENTS`` mapping from :class:`ApexConfig`,
    falling back to :data:`_REGIME_DEFAULTS` for any regime whose config
    string is missing or malformed.
    """
    out: Dict[str, Dict[str, float]] = {}
    for regime, cfg_attr in _REGIME_CONFIG_KEYS.items():
        raw = getattr(ApexConfig, cfg_attr, None)
        parsed = _parse_regime_quad(raw)
        out[regime] = parsed if parsed is not None else dict(_REGIME_DEFAULTS[regime])
    return out


class ExitUrgency(Enum):
    """Exit urgency levels."""
    IMMEDIATE = "immediate"      # Exit now at market
    HIGH = "high"                # Exit within minutes
    MODERATE = "moderate"        # Exit within hours
    LOW = "low"                  # Monitor, no rush
    HOLD = "hold"                # Keep position


def urgency_to_order_mode(urgency: "ExitUrgency") -> Dict[str, bool]:
    """
    Translate an :class:`ExitUrgency` value into connector order-mode kwargs.

    All connectors share the ``force_market`` (taker market order) and
    ``is_maker`` (rest-on-book limit) conventions. Instead of every caller
    re-deriving the mapping, this helper returns a dict that can be
    spread directly into ``connector.execute_order(**mode, ...)``.

        IMMEDIATE → force_market=True,  is_maker=False (cross the book now)
        HIGH      → force_market=False, is_maker=False (aggressive limit at touch)
        MODERATE  → force_market=False, is_maker=False (limit at touch, no rush)
        LOW       → force_market=False, is_maker=True  (passive post-only at mid)
        HOLD      → force_market=False, is_maker=True  (no order; kept for parity)

    Args:
        urgency: Exit urgency level from :meth:`DynamicExitManager.should_exit`.

    Returns:
        Dict with keys ``force_market`` (bool) and ``is_maker`` (bool).

    Raises:
        TypeError: If ``urgency`` is not an :class:`ExitUrgency`.
    """
    if not isinstance(urgency, ExitUrgency):
        raise TypeError(
            f"urgency must be ExitUrgency, got {type(urgency).__name__}"
        )
    if urgency is ExitUrgency.IMMEDIATE:
        return {"force_market": True, "is_maker": False}
    if urgency is ExitUrgency.LOW or urgency is ExitUrgency.HOLD:
        return {"force_market": False, "is_maker": True}
    return {"force_market": False, "is_maker": False}


@dataclass
class DynamicExitLevels:
    """Dynamic exit levels for a position."""
    stop_loss_pct: float         # Stop loss as % from entry
    take_profit_pct: float       # Take profit as % from entry
    trailing_activation_pct: float  # When to activate trailing stop
    trailing_distance_pct: float    # Trailing stop distance
    max_hold_days: int           # Maximum holding period
    signal_exit_threshold: float # Signal level to trigger exit
    urgency: ExitUrgency         # Current exit urgency
    reason: str                  # Explanation


class DynamicExitManager:
    """
    Manages dynamic exit levels based on market conditions.

    Key principles:
    1. Volatile markets = tighter stops, faster exits
    2. Trending markets = wider targets, let profits run
    3. Losing positions = faster exits, lower thresholds
    4. Winning positions = trailing stops, protect gains
    5. Stale positions = time decay, lower thresholds
    """

    # Base parameters — sourced from ApexConfig so all thresholds are tunable
    # via env without code changes. Preserved as class attributes for
    # backwards-compatibility with callers that read ``cls.BASE_*`` directly.
    BASE_STOP_LOSS_PCT = float(ApexConfig.EXIT_BASE_STOP_LOSS_PCT)
    BASE_TAKE_PROFIT_PCT = float(ApexConfig.EXIT_BASE_TAKE_PROFIT_PCT)
    BASE_TRAILING_ACTIVATION = float(ApexConfig.EXIT_BASE_TRAILING_ACTIVATION)
    BASE_TRAILING_DISTANCE = float(ApexConfig.EXIT_BASE_TRAILING_DISTANCE)
    BASE_MAX_HOLD_DAYS = int(ApexConfig.EXIT_BASE_MAX_HOLD_DAYS)
    BASE_SIGNAL_EXIT = float(ApexConfig.SIGNAL_EXIT_BASE)

    # Regime multipliers — built once from ApexConfig.EXIT_REGIME_* strings,
    # with fallback to :data:`_REGIME_DEFAULTS` per regime on parse failure.
    # Exposed as a class attribute for backwards compatibility with callers
    # that iterate the dict directly.
    REGIME_ADJUSTMENTS: Dict[str, Dict[str, float]] = _build_regime_adjustments()

    # Adaptive multiplier learning: slow learning rate, bounded changes.
    # Nudges regime multipliers toward better values based on real trade outcomes.
    _ADAPT_LEARNING_RATE: float = 0.015      # per-trade nudge size
    _ADAPT_MIN_TRADES: int = 8               # need ≥N outcomes before adapting
    _ADAPT_STOP_MULT_BOUNDS: tuple = (0.50, 1.80)
    _ADAPT_TARGET_MULT_BOUNDS: tuple = (0.60, 2.50)
    _ADAPT_HOLD_MULT_BOUNDS: tuple = (0.30, 2.00)
    _ADAPT_SIGNAL_MULT_BOUNDS: tuple = (0.50, 2.00)
    _PERSIST_PATH: str = str(getattr(ApexConfig, "DATA_DIR", "data")) + "/exit_multipliers.json"

    def __init__(self):
        self.position_history: Dict[str, list] = {}  # Track P&L trajectory
        self.base_signal_exit = getattr(ApexConfig, "SIGNAL_EXIT_BASE", self.BASE_SIGNAL_EXIT)

        # Adaptive multipliers — start from class-level REGIME_ADJUSTMENTS defaults,
        # then drift toward better values based on observed trade outcomes.
        import copy
        self._adapted_multipliers: Dict[str, Dict[str, float]] = copy.deepcopy(self.REGIME_ADJUSTMENTS)

        # Per-regime outcome buffer: list of dicts with exit_reason, pnl_pct, exit_signal
        self._outcome_buffer: Dict[str, List[dict]] = {r: [] for r in self.REGIME_ADJUSTMENTS}

        # Load previously learned multipliers from disk (persists across restarts)
        self._load_multipliers()
        logger.info("DynamicExitManager initialized (adaptive learning enabled)")

    # ── Persistence ──────────────────────────────────────────────────────────

    def _load_multipliers(self) -> None:
        """Load previously adapted multipliers from disk."""
        try:
            if os.path.exists(self._PERSIST_PATH):
                with open(self._PERSIST_PATH) as fh:
                    saved = json.load(fh)
                for regime, mults in saved.items():
                    if regime in self._adapted_multipliers:
                        self._adapted_multipliers[regime].update(mults)
                logger.info("Loaded adapted exit multipliers from %s", self._PERSIST_PATH)
        except Exception as e:
            logger.debug("Could not load exit multipliers (non-fatal): %s", e)

    def _save_multipliers(self) -> None:
        """Persist adapted multipliers to disk."""
        try:
            os.makedirs(os.path.dirname(self._PERSIST_PATH), exist_ok=True)
            with open(self._PERSIST_PATH, "w") as fh:
                json.dump(self._adapted_multipliers, fh, indent=2)
        except Exception as e:
            logger.debug("Could not save exit multipliers (non-fatal): %s", e)

    # ── Learning ─────────────────────────────────────────────────────────────

    def record_closed_trade(
        self,
        regime: str,
        exit_reason: str,
        pnl_pct: float,
        exit_signal: float = 0.0,
    ) -> None:
        """
        Record a closed trade outcome and adapt regime multipliers.

        Called once per position close from execution_loop. The learning logic:
        - Stop-loss exits that are worse than -2× the base stop → stop was too tight
          (noise triggered it) → nudge stop_mult UP.
        - Stop-loss exits near exactly -base_stop → stop was appropriate.
        - Take-profit exits → target was reachable → nudge target_mult slightly UP
          (we can afford to aim a bit higher next time).
        - Signal-decay exits that resulted in big losses → signal exit was too slow
          → nudge signal_mult UP (be more sensitive next time).
        - Long hold exits (>2× base hold_days) that were losers → hold too long
          → nudge hold_mult DOWN.
        """
        _regime = regime.lower() if regime else "neutral"
        if _regime not in self._outcome_buffer:
            _regime = "neutral"

        self._outcome_buffer[_regime].append({
            "exit_reason": exit_reason,
            "pnl_pct": float(pnl_pct),
            "exit_signal": float(exit_signal),
        })

        # Adapt once we have enough data for this regime
        buf = self._outcome_buffer[_regime]
        if len(buf) >= self._ADAPT_MIN_TRADES:
            self._adapt_multipliers(_regime, buf[-self._ADAPT_MIN_TRADES:])
            # Keep rolling: don't accumulate forever
            if len(buf) > self._ADAPT_MIN_TRADES * 3:
                self._outcome_buffer[_regime] = buf[-self._ADAPT_MIN_TRADES:]

    def _adapt_multipliers(self, regime: str, outcomes: List[dict]) -> None:
        """
        Nudge regime multipliers based on recent trade outcomes.
        Learning rate is intentionally small to avoid wild oscillations.
        """
        lr = self._ADAPT_LEARNING_RATE
        mults = self._adapted_multipliers[regime]

        stop_exits    = [o for o in outcomes if "stop" in o["exit_reason"].lower()]
        tp_exits      = [o for o in outcomes if "profit" in o["exit_reason"].lower()
                                                or "take" in o["exit_reason"].lower()]
        signal_exits  = [o for o in outcomes if "signal" in o["exit_reason"].lower()
                                                or "decay" in o["exit_reason"].lower()]

        # ── Stop-loss adaptation ──────────────────────────────────────────────
        if stop_exits:
            avg_stop_pnl = float(np.mean([o["pnl_pct"] for o in stop_exits]))
            base_stop = self.BASE_STOP_LOSS_PCT * mults.get("stop_mult", 1.0)
            # If average exit is much shallower than expected stop depth → stops too tight
            if avg_stop_pnl > -base_stop * 0.60:
                mults["stop_mult"] = float(np.clip(
                    mults["stop_mult"] + lr, *self._ADAPT_STOP_MULT_BOUNDS
                ))
            # If average stop exit is extremely deep → stops firing too late
            elif avg_stop_pnl < -base_stop * 1.50:
                mults["stop_mult"] = float(np.clip(
                    mults["stop_mult"] - lr, *self._ADAPT_STOP_MULT_BOUNDS
                ))

        # ── Take-profit adaptation ────────────────────────────────────────────
        if tp_exits:
            # If we're consistently hitting targets → room to aim higher
            mults["target_mult"] = float(np.clip(
                mults["target_mult"] + lr * 0.5, *self._ADAPT_TARGET_MULT_BOUNDS
            ))

        # ── Signal-exit adaptation ────────────────────────────────────────────
        if signal_exits:
            avg_sig_pnl = float(np.mean([o["pnl_pct"] for o in signal_exits]))
            # If signal exits are ending in large losses → exiting too late → be more sensitive
            if avg_sig_pnl < -0.015:
                mults["signal_mult"] = float(np.clip(
                    mults["signal_mult"] + lr, *self._ADAPT_SIGNAL_MULT_BOUNDS
                ))
            # If signal exits are small losses / near breakeven → exits are timely
            elif avg_sig_pnl > -0.005:
                mults["signal_mult"] = float(np.clip(
                    mults["signal_mult"] - lr * 0.5, *self._ADAPT_SIGNAL_MULT_BOUNDS
                ))

        # ── Hold-time adaptation ──────────────────────────────────────────────
        loser_exits = [o for o in outcomes if o["pnl_pct"] < -0.01]
        if len(loser_exits) > len(outcomes) * 0.6:
            # >60% losers in this regime → holding too long → shorten hold
            mults["hold_mult"] = float(np.clip(
                mults["hold_mult"] - lr, *self._ADAPT_HOLD_MULT_BOUNDS
            ))
        elif len(loser_exits) < len(outcomes) * 0.3:
            # <30% losers → doing well → can hold slightly longer
            mults["hold_mult"] = float(np.clip(
                mults["hold_mult"] + lr * 0.5, *self._ADAPT_HOLD_MULT_BOUNDS
            ))

        self._adapted_multipliers[regime] = mults
        self._save_multipliers()
        logger.info(
            "📈 ExitManager adapted [%s]: stop=%.2f target=%.2f hold=%.2f signal=%.2f",
            regime,
            mults.get("stop_mult", 1.0), mults.get("target_mult", 1.0),
            mults.get("hold_mult", 1.0), mults.get("signal_mult", 1.0),
        )

    def calculate_exit_levels(
        self,
        symbol: str,
        entry_price: float,
        current_price: float,
        side: str,  # 'LONG' or 'SHORT'
        entry_signal: float,
        current_signal: float,
        confidence: float,
        regime: str,
        vix_level: Optional[float],
        atr: Optional[float],
        entry_time: datetime
    ) -> DynamicExitLevels:
        """
        Calculate dynamic exit levels for a position.

        Args:
            symbol: Stock ticker
            entry_price: Entry price
            current_price: Current price
            side: LONG or SHORT
            entry_signal: Signal strength at entry
            current_signal: Current signal strength
            confidence: Current confidence
            regime: Market regime
            vix_level: Current VIX (optional)
            atr: Current ATR (optional)
            entry_time: When position was opened

        Returns:
            DynamicExitLevels with adjusted thresholds
        """
        # Start with base values
        stop_pct = self.BASE_STOP_LOSS_PCT
        target_pct = self.BASE_TAKE_PROFIT_PCT
        trail_activation = self.BASE_TRAILING_ACTIVATION
        trail_distance = self.BASE_TRAILING_DISTANCE
        max_hold = self.BASE_MAX_HOLD_DAYS
        signal_exit = self.base_signal_exit

        # Calculate current P&L (entry_price must be positive — validated below)
        if entry_price <= 0 or not np.isfinite(entry_price):
            raise ValueError(f"entry_price must be positive finite, got {entry_price!r}")
        if current_price <= 0 or not np.isfinite(current_price):
            raise ValueError(f"current_price must be positive finite, got {current_price!r}")

        if side == 'LONG':
            pnl_pct = (current_price / entry_price - 1)
        else:
            pnl_pct = (entry_price / current_price - 1)

        # Holding-period — tz-safe. The previous line
        #   ``(datetime.now() - entry_time).total_seconds() / 3600``
        # computed a value that was then *discarded* (orphaned expression,
        # not assigned), and crashed when entry_time was tz-aware while
        # datetime.now() was naive.
        holding_days = _safe_elapsed_days(entry_time)
        holding_hours = _safe_elapsed_hours(entry_time)

        # === 1. REGIME ADJUSTMENTS (adaptive — drifts toward better values per trade) ===
        canonical_regime = _normalize_regime(regime)
        regime_adj = self._adapted_multipliers.get(
            canonical_regime,
            self._adapted_multipliers.get('neutral', self.REGIME_ADJUSTMENTS['neutral']),
        )

        # Adjust for side in bearish/bullish regimes (counter-trend tighter stops)
        if side == 'LONG' and canonical_regime in ('bear', 'strong_bear'):
            stop_pct *= regime_adj['stop_mult'] * 0.8
            target_pct *= regime_adj['target_mult'] * 0.7
        elif side == 'SHORT' and canonical_regime in ('bull', 'strong_bull'):
            stop_pct *= regime_adj['stop_mult'] * 0.8
            target_pct *= regime_adj['target_mult'] * 0.7
        else:
            stop_pct *= regime_adj['stop_mult']
            target_pct *= regime_adj['target_mult']

        max_hold = int(max_hold * regime_adj['hold_mult'])
        signal_exit *= regime_adj['signal_mult']

        # === 2. VIX ADJUSTMENTS (all thresholds sourced from ApexConfig) ===
        if vix_level is not None and np.isfinite(vix_level):
            vix_val = float(vix_level)
            if vix_val > ApexConfig.EXIT_VIX_EXTREME:
                stop_pct *= float(ApexConfig.EXIT_VIX_EXTREME_STOP_MULT)
                target_pct *= float(ApexConfig.EXIT_VIX_EXTREME_STOP_MULT)
                max_hold = min(max_hold, int(ApexConfig.EXIT_VIX_EXTREME_MAX_HOLD))
                signal_exit *= float(ApexConfig.EXIT_VIX_EXTREME_SIGNAL_MULT)
            elif vix_val > ApexConfig.EXIT_VIX_HIGH:
                stop_pct *= float(ApexConfig.EXIT_VIX_HIGH_STOP_MULT)
                target_pct *= float(ApexConfig.EXIT_VIX_HIGH_STOP_MULT)
                max_hold = min(max_hold, int(ApexConfig.EXIT_VIX_HIGH_MAX_HOLD))
                signal_exit *= float(ApexConfig.EXIT_VIX_HIGH_SIGNAL_MULT)
            elif vix_val > ApexConfig.EXIT_VIX_ELEVATED:
                stop_pct *= float(ApexConfig.EXIT_VIX_ELEVATED_STOP_MULT)
                target_pct *= float(ApexConfig.EXIT_VIX_ELEVATED_STOP_MULT)
                signal_exit *= float(ApexConfig.EXIT_VIX_ELEVATED_SIGNAL_MULT)
            elif vix_val < ApexConfig.EXIT_VIX_COMPLACENCY:
                stop_pct *= float(ApexConfig.EXIT_VIX_COMPLACENCY_STOP_MULT)
                target_pct *= float(ApexConfig.EXIT_VIX_COMPLACENCY_TARGET_MULT)
                max_hold = int(max_hold * float(ApexConfig.EXIT_VIX_COMPLACENCY_HOLD_MULT))

        # === 3. ATR-BASED ADJUSTMENTS ===
        if atr is not None and np.isfinite(atr) and atr > 0 and entry_price > 0:
            atr_pct = atr / entry_price
            stop_pct = max(stop_pct, atr_pct * float(ApexConfig.EXIT_ATR_STOP_MULT))
            target_pct = max(target_pct, atr_pct * float(ApexConfig.EXIT_ATR_TARGET_MULT))
            trail_distance = max(trail_distance, atr_pct * float(ApexConfig.EXIT_ATR_TRAIL_MULT))

        # === 4. SIGNAL STRENGTH ADJUSTMENTS ===
        # Strong entry signal = more conviction = can hold longer
        if abs(entry_signal) > 0.7:
            stop_pct *= 1.15
            target_pct *= 1.2
            max_hold = int(max_hold * 1.2)
        elif abs(entry_signal) < 0.5:
            stop_pct *= 0.9
            target_pct *= 0.85
            max_hold = int(max_hold * 0.8)

        # === 5. P&L TRAJECTORY ADJUSTMENTS (v3 — config-driven tiers) ===
        # All thresholds and multipliers sourced from ApexConfig.EXIT_PNL_*
        # so bands can be tuned without code changes.
        pnl_big_winner = float(ApexConfig.EXIT_PNL_BIG_WINNER_PCT)
        pnl_winner = float(ApexConfig.EXIT_PNL_WINNER_PCT)
        pnl_small_winner = float(ApexConfig.EXIT_PNL_SMALL_WINNER_PCT)
        pnl_loser = float(ApexConfig.EXIT_PNL_LOSER_PCT)
        trail_activation_pnl = float(ApexConfig.EXIT_PNL_TRAIL_ACTIVATION_PCT)
        big_win_trail_mult = float(ApexConfig.EXIT_BIG_WINNER_TRAIL_MULT)
        win_trail_mult = float(ApexConfig.EXIT_WINNER_TRAIL_MULT)
        win_signal_mult = float(ApexConfig.EXIT_WINNER_SIGNAL_MULT)
        small_win_signal_mult = float(ApexConfig.EXIT_SMALL_WINNER_SIGNAL_MULT)
        loser_signal_mult = float(ApexConfig.EXIT_LOSER_SIGNAL_MULT)
        loser_max_hold = int(ApexConfig.EXIT_LOSER_MAX_HOLD_DAYS)
        big_win_mode = str(ApexConfig.EXIT_BIG_WINNER_SIGNAL_MODE).lower()
        big_win_chop_mult = float(ApexConfig.EXIT_BIG_WINNER_CHOP_SIGNAL_MULT)

        if pnl_pct > pnl_big_winner:
            # Big winner — tight trail + disciplined signal-exit policy.
            # Legacy behaviour (mode="disable") zeroed signal_exit outright,
            # which causes winners to ride through a full regime flip in
            # choppy / high-volatility markets. Mode="regime" (default) keeps
            # signal_exit active in chop/high_vol and only fully disables it
            # in trending regimes where signal reversals are less frequent.
            trail_activation = trail_activation_pnl
            trail_distance *= big_win_trail_mult
            if big_win_mode == "disable":
                signal_exit = 0.0
            elif canonical_regime in ("high_volatility", "neutral"):
                signal_exit *= big_win_chop_mult
            else:
                signal_exit = 0.0
        elif pnl_pct > pnl_winner:
            # Winning — activate trailing with breathing room.
            trail_activation = trail_activation_pnl
            trail_distance *= win_trail_mult
            signal_exit *= win_signal_mult
        elif pnl_pct > pnl_small_winner:
            # Small winner — be patient, let it grow.
            signal_exit *= small_win_signal_mult
        elif pnl_pct < pnl_loser:
            # Losing beyond the loser band — tighten and cap hold.
            signal_exit *= loser_signal_mult
            max_hold = min(max_hold, loser_max_hold)

        # === 6. TIME DECAY ADJUSTMENTS ===
        # Intra-day adverse tightening — a position underwater by
        # EXIT_INTRADAY_ADVERSE_FRAC × base_stop within the first 24h is almost
        # certainly wrong. Pull the stop in and raise signal-exit sensitivity.
        base_stop_ref = float(ApexConfig.EXIT_BASE_STOP_LOSS_PCT)
        adverse_frac = float(ApexConfig.EXIT_INTRADAY_ADVERSE_FRAC)
        intraday_stop_mult = float(ApexConfig.EXIT_INTRADAY_STOP_MULT)
        intraday_signal_mult = float(ApexConfig.EXIT_INTRADAY_SIGNAL_MULT)
        if holding_hours < 24.0 and pnl_pct < -(base_stop_ref * adverse_frac):
            stop_pct *= intraday_stop_mult
            signal_exit *= intraday_signal_mult
            logger.debug(
                "Intra-day adverse tightening: holding_hours=%.1f pnl=%.2f%%",
                holding_hours, pnl_pct * 100.0,
            )

        # Only apply time decay to LOSING positions
        if holding_days >= 7 and pnl_pct < -0.01:
            # Stale losing position - start tightening
            decay_factor = max(0.6, 1 - (holding_days - 7) / max_hold * 0.3)
            signal_exit *= (1 / decay_factor)

        if holding_days >= 10 and pnl_pct < -0.02:
            # 10 days with significant loss - lower expectations
            target_pct *= 0.8

        if holding_days >= max_hold * 0.9:
            # Approaching max hold
            signal_exit *= 0.8

        # === 7. SIGNAL REVERSAL CHECK ===
        # Only care about reversals on losing positions
        signal_reversed = (entry_signal > 0 and current_signal < -0.3) or \
                         (entry_signal < 0 and current_signal > 0.3)

        if signal_reversed and pnl_pct < 0.005:  # Trigger earlier than 0%
            signal_exit *= 0.5  # Much more sensitive when signal flips AND losing or barely winning

        # === DETERMINE EXIT URGENCY ===
        urgency = ExitUrgency.HOLD
        reason = "Position healthy"

        # Check stop loss
        if pnl_pct <= -stop_pct:
            urgency = ExitUrgency.IMMEDIATE
            reason = f"Stop loss hit ({pnl_pct*100:+.1f}%)"

        # Check take profit
        elif pnl_pct >= target_pct:
            urgency = ExitUrgency.HIGH
            reason = f"Take profit reached ({pnl_pct*100:+.1f}%)"

        # Check signal reversal
        elif signal_reversed and pnl_pct < 0:
            urgency = ExitUrgency.HIGH
            reason = f"Signal reversed while losing ({current_signal:+.2f})"

        # Check holding time
        elif holding_days >= max_hold:
            urgency = ExitUrgency.HIGH
            reason = f"Max holding period ({holding_days}d)"

        # Check signal decay
        elif abs(current_signal) < signal_exit and holding_days >= 3:
            if pnl_pct < 0:
                urgency = ExitUrgency.HIGH
                reason = f"Signal weak on loser ({current_signal:+.2f})"
            else:
                urgency = ExitUrgency.MODERATE
                reason = f"Signal decayed ({current_signal:+.2f})"

        # Approaching limits
        elif holding_days >= max_hold * 0.8:
            urgency = ExitUrgency.MODERATE
            reason = f"Approaching max hold ({holding_days}/{max_hold}d)"

        elif pnl_pct <= -stop_pct * 0.7:
            urgency = ExitUrgency.MODERATE
            reason = f"Approaching stop ({pnl_pct*100:+.1f}%)"

        return DynamicExitLevels(
            stop_loss_pct=float(np.clip(
                stop_pct,
                ApexConfig.EXIT_STOP_CLAMP_MIN,
                ApexConfig.EXIT_STOP_CLAMP_MAX,
            )),
            take_profit_pct=float(np.clip(
                target_pct,
                ApexConfig.EXIT_TARGET_CLAMP_MIN,
                ApexConfig.EXIT_TARGET_CLAMP_MAX,
            )),
            trailing_activation_pct=float(np.clip(
                trail_activation,
                ApexConfig.EXIT_STOP_CLAMP_MIN / 2.0,
                ApexConfig.EXIT_STOP_CLAMP_MAX,
            )),
            trailing_distance_pct=float(np.clip(
                trail_distance,
                ApexConfig.EXIT_STOP_CLAMP_MIN / 2.0,
                ApexConfig.EXIT_STOP_CLAMP_MAX,
            )),
            max_hold_days=int(np.clip(
                max_hold,
                int(ApexConfig.EXIT_MAX_HOLD_CLAMP_MIN),
                int(ApexConfig.EXIT_MAX_HOLD_CLAMP_MAX),
            )),
            signal_exit_threshold=float(np.clip(
                signal_exit,
                ApexConfig.EXIT_SIGNAL_EXIT_CLAMP_MIN,
                ApexConfig.EXIT_SIGNAL_EXIT_CLAMP_MAX,
            )),
            urgency=urgency,
            reason=reason,
        )

    def should_exit(
        self,
        symbol: str,
        entry_price: float,
        current_price: float,
        side: str,
        entry_signal: float,
        current_signal: float,
        confidence: float,
        regime: str,
        vix_level: Optional[float],
        atr: Optional[float],
        entry_time: datetime,
        peak_price: Optional[float] = None
    ) -> Tuple[bool, str, ExitUrgency]:
        """
        Determine if position should be exited.

        Returns:
            (should_exit, reason, urgency)
        """
        levels = self.calculate_exit_levels(
            symbol, entry_price, current_price, side,
            entry_signal, current_signal, confidence,
            regime, vix_level, atr, entry_time
        )

        # Calculate P&L
        if side == 'LONG':
            pnl_pct = (current_price / entry_price - 1)
        else:
            pnl_pct = (entry_price / current_price - 1)

        holding_days = _safe_elapsed_days(entry_time)

        # === HARD EXITS ===

        # Stop loss
        if pnl_pct <= -levels.stop_loss_pct:
            return True, f"Stop loss: {pnl_pct*100:+.1f}% <= -{levels.stop_loss_pct*100:.1f}%", ExitUrgency.IMMEDIATE

        # Take profit
        if pnl_pct >= levels.take_profit_pct:
            return True, f"Take profit: {pnl_pct*100:+.1f}% >= +{levels.take_profit_pct*100:.1f}%", ExitUrgency.HIGH

        # Max holding period
        if holding_days >= levels.max_hold_days:
            return True, f"Max hold: {holding_days}d >= {levels.max_hold_days}d", ExitUrgency.HIGH

        # === TRAILING STOP ===
        if peak_price is not None and pnl_pct > levels.trailing_activation_pct:
            if side == 'LONG':
                trailing_stop = peak_price * (1 - levels.trailing_distance_pct)
                if current_price <= trailing_stop:
                    return True, f"Trailing stop: ${current_price:.2f} <= ${trailing_stop:.2f}", ExitUrgency.IMMEDIATE
            else:
                trailing_stop = peak_price * (1 + levels.trailing_distance_pct)
                if current_price >= trailing_stop:
                    return True, f"Trailing stop: ${current_price:.2f} >= ${trailing_stop:.2f}", ExitUrgency.IMMEDIATE

        # === SIGNAL-BASED EXITS ===

        # Strong reversal signal - only exit if NOT winning big
        if side == 'LONG' and current_signal < -0.50 and confidence > 0.40 and pnl_pct < 0.03:
            return True, f"Strong bearish reversal: {current_signal:+.2f}", ExitUrgency.HIGH

        if side == 'SHORT' and current_signal > 0.50 and confidence > 0.40 and pnl_pct < 0.03:
            return True, f"Strong bullish reversal: {current_signal:+.2f}", ExitUrgency.HIGH

        # Moderate reversal + losing - cut losses fast
        if side == 'LONG' and current_signal < -0.30 and pnl_pct < 0.0:
            return True, f"Bearish on loser: {current_signal:+.2f}, P&L={pnl_pct*100:+.1f}%", ExitUrgency.HIGH

        if side == 'SHORT' and current_signal > 0.30 and pnl_pct < 0.0:
            return True, f"Bullish on loser: {current_signal:+.2f}, P&L={pnl_pct*100:+.1f}%", ExitUrgency.HIGH

        # Signal decay - ONLY on losing positions after extended hold time
        if holding_days >= 5 and pnl_pct < -0.01:
            if side == 'LONG' and current_signal < levels.signal_exit_threshold:
                return True, f"Signal decay on loser: {current_signal:+.2f} after {holding_days}d", ExitUrgency.MODERATE

            if side == 'SHORT' and current_signal > -levels.signal_exit_threshold:
                return True, f"Signal decay on loser: {current_signal:+.2f} after {holding_days}d", ExitUrgency.MODERATE

        # WINNERS: Only exit via trailing stop or take profit (handled above)
        # Don't exit winners on signal decay - let the trailing stop do its job

        # No exit needed
        return False, levels.reason, levels.urgency

    def get_position_status(
        self,
        symbol: str,
        entry_price: float,
        current_price: float,
        side: str,
        entry_signal: float,
        current_signal: float,
        confidence: float,
        regime: str,
        vix_level: Optional[float],
        atr: Optional[float],
        entry_time: datetime
    ) -> Dict:
        """Get comprehensive status for a position."""
        levels = self.calculate_exit_levels(
            symbol, entry_price, current_price, side,
            entry_signal, current_signal, confidence,
            regime, vix_level, atr, entry_time
        )

        if side == 'LONG':
            pnl_pct = (current_price / entry_price - 1)
            stop_price = entry_price * (1 - levels.stop_loss_pct)
            target_price = entry_price * (1 + levels.take_profit_pct)
        else:
            pnl_pct = (entry_price / current_price - 1)
            stop_price = entry_price * (1 + levels.stop_loss_pct)
            target_price = entry_price * (1 - levels.take_profit_pct)

        holding_days = _safe_elapsed_days(entry_time)

        return {
            'symbol': symbol,
            'side': side,
            'entry_price': entry_price,
            'current_price': current_price,
            'pnl_pct': pnl_pct,
            'holding_days': holding_days,
            'stop_price': stop_price,
            'target_price': target_price,
            'stop_pct': levels.stop_loss_pct,
            'target_pct': levels.take_profit_pct,
            'max_hold_days': levels.max_hold_days,
            'days_remaining': levels.max_hold_days - holding_days,
            'signal_exit_threshold': levels.signal_exit_threshold,
            'current_signal': current_signal,
            'urgency': levels.urgency.value,
            'status': levels.reason,
            'regime': regime,
            'vix': vix_level
        }


# Singleton instance
_exit_manager: Optional[DynamicExitManager] = None


def get_exit_manager() -> DynamicExitManager:
    """Get or create the dynamic exit manager singleton."""
    global _exit_manager
    if _exit_manager is None:
        _exit_manager = DynamicExitManager()
    return _exit_manager
