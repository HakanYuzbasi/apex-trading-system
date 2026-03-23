"""
monitoring/signal_auto_tuner.py — Self-Improving Signal Threshold Tuner

Reads the last N EOD digests and auto-adjusts per-regime signal entry
thresholds based on empirical win rates observed in live trading.

The improvement loop:
    signal → trade → outcome → EOD digest → auto_tuner → tighter/looser thresholds
    → better signal quality → repeat

Rules:
  - Requires at least min_samples digests with ≥ min_trades_per_regime trades
    before adjusting any threshold.
  - If a regime's rolling win rate < urgency_threshold for consecutive days:
    → raise threshold by step (harder to enter, fewer but higher quality trades)
  - If a regime's rolling win rate ≥ strong_threshold for consecutive days:
    → lower threshold by step/2 (exploit more opportunities)
  - Adjustments are bounded: thresholds stay in [0.10, 0.40] hard limits.
  - Written to data/auto_tuned_thresholds.json (read by execution_loop on startup
    and every 1000 cycles to pick up live changes without restart).
  - Changes are logged clearly so operators can review and revert if needed.

Anti-over-fitting guards:
  - Minimum 3 consecutive days of evidence before adjusting.
  - Maximum ±0.03 total adjustment from original config value per regime.
  - Auto-reset: if win rate recovers, partial revert.
"""
from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field
from datetime import date, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# Bounds on auto-adjusted thresholds (protect against runaway adjustments)
_HARD_MIN = 0.10
_HARD_MAX = 0.40
_MAX_DRIFT_FROM_BASE = 0.03   # never drift more than ±0.03 from original config value


@dataclass
class RegimeThresholdState:
    regime: str
    base_value: float          # Original config value (never changes)
    current_value: float       # Active value (may be adjusted)
    consecutive_low: int = 0   # Days below urgency_threshold
    consecutive_high: int = 0  # Days above strong_threshold
    total_adjustment: float = 0.0  # Cumulative adjustment from base

    @property
    def clamped_value(self) -> float:
        lo = max(_HARD_MIN, self.base_value - _MAX_DRIFT_FROM_BASE)
        hi = min(_HARD_MAX, self.base_value + _MAX_DRIFT_FROM_BASE)
        return round(max(lo, min(hi, self.current_value)), 4)


@dataclass
class TuningResult:
    date: str
    changes: List[Dict]      # [{regime, old_value, new_value, direction, reason}]
    no_change_regimes: List[str]
    insufficient_data: List[str]


class SignalAutoTuner:
    """
    Reads EOD digest history and emits adjusted per-regime signal thresholds.

    Designed to run once per day (e.g. right after EOD digest is written)
    or at engine startup to pick up overnight adjustments.
    """

    def __init__(
        self,
        data_dir: "str | Path" = "data",
        min_samples: int = 3,               # min consecutive days of evidence
        min_trades_per_regime: int = 3,     # min trades to trust WR estimate
        urgency_threshold: float = 0.40,    # WR below this → raise threshold
        strong_threshold: float = 0.62,     # WR above this → lower threshold
        step: float = 0.01,                 # adjustment per evaluation
    ):
        self.data_dir = Path(data_dir)
        self.min_samples = min_samples
        self.min_trades_per_regime = min_trades_per_regime
        self.urgency_threshold = urgency_threshold
        self.strong_threshold = strong_threshold
        self.step = step

        self._output_path = self.data_dir / "auto_tuned_thresholds.json"
        self._state_path = self.data_dir / "auto_tuner_state.json"
        self._state: Dict[str, RegimeThresholdState] = {}

        # Default base thresholds (mirrors config.py SIGNAL_THRESHOLDS_BY_REGIME)
        self._default_bases: Dict[str, float] = {
            "strong_bull": 0.14,
            "bull":        0.17,
            "neutral":     0.18,
            "bear":        0.21,
            "strong_bear": 0.24,
            "volatile":    0.20,
            "crisis":      0.22,
        }

        self._load_state()

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------

    def run(
        self,
        eod_reports: Optional[List[Dict]] = None,
        days_back: int = 7,
    ) -> TuningResult:
        """
        Execute one tuning pass.

        Args:
            eod_reports: Pre-loaded list of EOD report dicts (most recent first).
                         If None, loads from disk.
            days_back: How many days to look back when loading from disk.

        Returns:
            TuningResult describing what changed.
        """
        if eod_reports is None:
            eod_reports = self._load_digests(days_back)

        if len(eod_reports) < self.min_samples:
            logger.info(
                "AutoTuner: only %d digests available (need %d) — skipping",
                len(eod_reports), self.min_samples,
            )
            return TuningResult(
                date=date.today().isoformat(),
                changes=[],
                no_change_regimes=[],
                insufficient_data=list(self._default_bases.keys()),
            )

        # Aggregate per-regime win rates across the loaded digests
        regime_wrs = self._aggregate_regime_win_rates(eod_reports)

        # Significance gate: import once (graceful fallback if unavailable)
        try:
            from monitoring.stat_significance import is_significant as _is_sig
        except ImportError:
            _is_sig = None  # no gate if module missing

        changes = []
        no_change = []
        insufficient = []

        for regime, bases_value in self._default_bases.items():
            state = self._get_or_init_state(regime, bases_value)
            wr_data = regime_wrs.get(regime)

            if wr_data is None or wr_data["total_trades"] < self.min_trades_per_regime:
                insufficient.append(regime)
                # Reset streaks since we have no evidence
                state.consecutive_low = 0
                state.consecutive_high = 0
                continue

            wr = wr_data["win_rate"]
            n_trades = wr_data["total_trades"]
            wins = round(wr * n_trades)

            # Statistical significance gate: require win-rate deviation to be non-random
            if _is_sig is not None and not _is_sig(wins=wins, n=n_trades, alpha=0.05):
                logger.debug(
                    "AutoTuner [%s]: WR=%.1f%% (%d/%d) not significant — skipping adjustment",
                    regime, wr * 100, wins, n_trades,
                )
                no_change.append(regime)
                continue

            direction, reason = self._evaluate(state, wr)

            if direction == 0:
                no_change.append(regime)
                continue

            old_value = state.current_value
            state.current_value += direction * self.step
            state.total_adjustment += direction * self.step
            new_value = state.clamped_value
            state.current_value = new_value

            if abs(new_value - old_value) < 1e-6:
                no_change.append(regime)   # clamped — no effective change
                continue

            changes.append({
                "regime": regime,
                "old_value": round(old_value, 4),
                "new_value": new_value,
                "direction": "raise" if direction > 0 else "lower",
                "win_rate": round(wr, 3),
                "trades": n_trades,
                "reason": reason,
            })
            logger.info(
                "AutoTuner [%s]: %s threshold %.4f → %.4f  (WR=%.1f%% n=%d  %s)",
                regime,
                "RAISE" if direction > 0 else "LOWER",
                old_value, new_value, wr * 100, n_trades, reason,
            )

        result = TuningResult(
            date=date.today().isoformat(),
            changes=changes,
            no_change_regimes=no_change,
            insufficient_data=insufficient,
        )

        self._save_state()
        self._write_output()

        if changes:
            logger.info(
                "AutoTuner: adjusted %d regime thresholds, %d unchanged, %d insufficient data",
                len(changes), len(no_change), len(insufficient),
            )
        else:
            logger.debug(
                "AutoTuner: no adjustments needed (%d unchanged, %d insufficient)",
                len(no_change), len(insufficient),
            )

        return result

    def get_thresholds(self) -> Dict[str, float]:
        """Return current adjusted thresholds dict."""
        return {
            regime: self._get_or_init_state(regime, base).clamped_value
            for regime, base in self._default_bases.items()
        }

    def load_thresholds_from_disk(self) -> Optional[Dict[str, float]]:
        """
        Load previously auto-tuned thresholds from disk.
        Returns None if no file exists (use config defaults).
        """
        if not self._output_path.exists():
            return None
        try:
            with open(self._output_path, "r", encoding="utf-8") as fh:
                data = json.load(fh)
            thresholds = data.get("thresholds", {})
            logger.info("AutoTuner: loaded %d auto-tuned thresholds from disk", len(thresholds))
            return thresholds
        except Exception as exc:
            logger.warning("AutoTuner: failed to load thresholds: %s", exc)
            return None

    def reset_regime(self, regime: str) -> None:
        """Reset a specific regime back to its base config value."""
        if regime in self._state:
            base = self._default_bases.get(regime, self._state[regime].base_value)
            self._state[regime] = RegimeThresholdState(
                regime=regime, base_value=base, current_value=base
            )
            logger.info("AutoTuner: reset %s to base value %.4f", regime, base)
            self._save_state()
            self._write_output()

    def reset_all(self) -> None:
        """Reset all regimes to base config values."""
        self._state = {}
        self._save_state()
        self._write_output()
        logger.info("AutoTuner: reset all regimes to base values")

    # ------------------------------------------------------------------
    # Private
    # ------------------------------------------------------------------

    def _evaluate(
        self, state: RegimeThresholdState, win_rate: float
    ) -> Tuple[int, str]:
        """
        Determine adjustment direction.

        Returns:
            (direction, reason)
            direction: +1 = raise threshold, -1 = lower, 0 = no change
        """
        if win_rate < self.urgency_threshold:
            state.consecutive_low += 1
            state.consecutive_high = 0
            if state.consecutive_low >= self.min_samples:
                return +1, f"{state.consecutive_low}d WR {win_rate:.1%} < {self.urgency_threshold:.0%}"
            return 0, f"low WR {win_rate:.1%} (day {state.consecutive_low}/{self.min_samples})"
        elif win_rate >= self.strong_threshold:
            state.consecutive_high += 1
            state.consecutive_low = 0
            if state.consecutive_high >= self.min_samples:
                return -1, f"{state.consecutive_high}d WR {win_rate:.1%} ≥ {self.strong_threshold:.0%}"
            return 0, f"high WR {win_rate:.1%} (day {state.consecutive_high}/{self.min_samples})"
        else:
            # WR in acceptable range — partial streak reset
            state.consecutive_low = max(0, state.consecutive_low - 1)
            state.consecutive_high = max(0, state.consecutive_high - 1)

            # If we over-adjusted and WR is now good, gently revert
            if state.total_adjustment > self.step and win_rate > 0.50:
                state.total_adjustment -= self.step
                return -1, f"partial revert (WR recovered to {win_rate:.1%})"
            elif state.total_adjustment < -self.step and win_rate < 0.55:
                state.total_adjustment += self.step
                return +1, f"partial revert (WR weakened to {win_rate:.1%})"

            return 0, f"WR {win_rate:.1%} within target range"

    def _aggregate_regime_win_rates(self, reports: List[Dict]) -> Dict[str, Dict]:
        """
        Aggregate regime win rates across all loaded EOD reports.

        Returns: {regime: {win_rate: float, total_trades: int}}
        """
        totals: Dict[str, Dict] = {}

        for report in reports:
            by_regime = report.get("by_regime", {})
            for regime, rd in by_regime.items():
                trades = int(rd.get("trades", 0))
                wr = rd.get("win_rate")
                if trades == 0 or wr is None:
                    continue
                if regime not in totals:
                    totals[regime] = {"wins": 0, "total_trades": 0}
                wins = round(wr * trades)
                totals[regime]["wins"] += wins
                totals[regime]["total_trades"] += trades

        return {
            r: {
                "win_rate": d["wins"] / d["total_trades"],
                "total_trades": d["total_trades"],
            }
            for r, d in totals.items()
            if d["total_trades"] > 0
        }

    def _load_digests(self, days_back: int) -> List[Dict]:
        """Load EOD digest JSON files from data/eod_reports/."""
        reports = []
        reports_dir = self.data_dir / "eod_reports"
        today = date.today()
        for i in range(days_back):
            d = today - timedelta(days=i)
            path = reports_dir / f"{d.isoformat()}_digest.json"
            if path.exists():
                try:
                    with open(path, "r", encoding="utf-8") as fh:
                        reports.append(json.load(fh))
                except Exception:
                    pass
        return reports

    def _get_or_init_state(self, regime: str, base: float) -> RegimeThresholdState:
        if regime not in self._state:
            self._state[regime] = RegimeThresholdState(
                regime=regime, base_value=base, current_value=base
            )
        return self._state[regime]

    def _load_state(self) -> None:
        if not self._state_path.exists():
            return
        try:
            with open(self._state_path, "r", encoding="utf-8") as fh:
                raw = json.load(fh)
            for regime, d in raw.items():
                base = self._default_bases.get(regime, d.get("base_value", 0.18))
                self._state[regime] = RegimeThresholdState(
                    regime=regime,
                    base_value=float(d.get("base_value", base)),
                    current_value=float(d.get("current_value", base)),
                    consecutive_low=int(d.get("consecutive_low", 0)),
                    consecutive_high=int(d.get("consecutive_high", 0)),
                    total_adjustment=float(d.get("total_adjustment", 0.0)),
                )
            logger.info("AutoTuner: loaded state for %d regimes", len(self._state))
        except Exception as exc:
            logger.warning("AutoTuner: failed to load state: %s", exc)

    def _save_state(self) -> None:
        try:
            self.data_dir.mkdir(parents=True, exist_ok=True)
            raw = {
                regime: {
                    "base_value": s.base_value,
                    "current_value": s.current_value,
                    "consecutive_low": s.consecutive_low,
                    "consecutive_high": s.consecutive_high,
                    "total_adjustment": s.total_adjustment,
                }
                for regime, s in self._state.items()
            }
            tmp = str(self._state_path) + ".tmp"
            with open(tmp, "w", encoding="utf-8") as fh:
                json.dump(raw, fh, indent=2)
            os.replace(tmp, str(self._state_path))
        except Exception as exc:
            logger.warning("AutoTuner: failed to save state: %s", exc)

    def _write_output(self) -> None:
        """Write the active thresholds to data/auto_tuned_thresholds.json."""
        try:
            self.data_dir.mkdir(parents=True, exist_ok=True)
            thresholds = self.get_thresholds()
            output = {
                "generated_at": date.today().isoformat(),
                "thresholds": thresholds,
                "note": "Auto-tuned by SignalAutoTuner. Edit auto_tuner_state.json to reset.",
            }
            tmp = str(self._output_path) + ".tmp"
            with open(tmp, "w", encoding="utf-8") as fh:
                json.dump(output, fh, indent=2)
            os.replace(tmp, str(self._output_path))
        except Exception as exc:
            logger.warning("AutoTuner: failed to write output: %s", exc)
