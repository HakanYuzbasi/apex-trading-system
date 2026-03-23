"""
risk/threshold_calibrator.py - Automatic Threshold Calibration

Calibrates exit thresholds and builds slippage blacklists from live trade history.
Runs at startup (loads persisted values) and periodically every 6h (recalibrates).
Results are pushed to trading_excellence._ACTIVE_PARAMS and execution loop blacklist.
"""
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# ── Defaults (used when insufficient trade history) ───────────────────────────
DEFAULTS: Dict = {
    "weak_signal_loss_threshold_pct": -0.50,   # Excellence weak-signal exit threshold
    "no_signal_loss_threshold_pct":   -0.01,   # Excellence no-signal exit threshold
    "slippage_blacklist":             [],        # Symbols blacklisted (avg slippage > ban bps)
    "slippage_watchlist":             [],        # Symbols monitored (40-80 bps)
    "symbol_probation":               {},        # {symbol: min_signal_threshold} from live win rates
    "calibrated_at":                  None,
    "validation":                     None,     # Walk-forward validation results
}

# ── Per-symbol probation thresholds ───────────────────────────────────────────
PROBATION_MIN_TRADES   = 5      # Below this: apply light probation regardless
PROBATION_WIN_RATE_LO  = 0.30   # Win rate < 30% → strong probation
PROBATION_WIN_RATE_MID = 0.45   # Win rate 30-45% → light probation
PROBATION_THRESH_NONE_DATA = 0.10  # No history → below model output floor (signals max ~0.272); let normal gates handle
PROBATION_THRESH_STRONG    = 0.45  # Win rate < 30%
PROBATION_THRESH_LIGHT     = 0.35  # Win rate 30-45%

# ── Calibration constants ─────────────────────────────────────────────────────
ATR_SCALE_FACTOR   = 1.5    # Threshold = median_win_pct * ATR_SCALE * 0.30
ATR_FLOOR_PCT      = 0.30   # Min exit threshold (never tighter than -0.30%)
ATR_CEIL_PCT       = 2.50   # Max exit threshold (never looser than -2.50%)
SLIPPAGE_BAN_BPS   = 80     # Blacklist symbol if avg slippage > this
SLIPPAGE_WATCH_BPS = 40     # Watchlist symbol if avg slippage > this
MIN_TRADES_FOR_CALIBRATION = 20   # Minimum live_entry trades (raised from 5 — need statistical validity)
VALIDATION_SPLIT   = 0.80   # 80% train, 20% validation (chronological)
# Revert if val net_pnl/trade degrades by more than this vs training set
VALIDATION_DEGRADE_THRESHOLD = 0.30  # 30% degradation triggers revert


class ThresholdCalibrator:
    """
    Calibrates trading thresholds from live execution data.

    All I/O is synchronous — call run_calibration() via asyncio.to_thread()
    from async context to avoid blocking the event loop.
    """

    def __init__(self, data_dir: Path):
        self.data_dir = Path(data_dir)
        self.perf_file = self.data_dir / "users" / "admin" / "performance_attribution.json"
        self.latency_file = self.data_dir / "users" / "admin" / "audit" / "execution_latency.jsonl"
        self.out_file = self.data_dir / "calibrated_thresholds.json"

    # ── Public interface ──────────────────────────────────────────────────────

    def run_calibration(self) -> Dict:
        """
        Run full calibration with walk-forward validation.

        Steps:
        1. Sort closed trades chronologically.
        2. Train on first 80%, validate on last 20%.
        3. If validation metrics degrade significantly vs training, revert to
           previous persisted threshold instead of applying the new one.

        Synchronous — call via asyncio.to_thread() from async context.
        """
        trades = self._load_closed_trades()
        if len(trades) < MIN_TRADES_FOR_CALIBRATION:
            logger.info(
                f"ThresholdCalibrator: only {len(trades)} closed trades "
                f"(need {MIN_TRADES_FOR_CALIBRATION}), keeping defaults"
            )
            return DEFAULTS.copy()

        # ── Chronological split ───────────────────────────────────────────────
        trades_sorted = sorted(trades, key=lambda t: t.get("exit_time") or t.get("entry_time") or "")
        split_idx = max(1, int(len(trades_sorted) * VALIDATION_SPLIT))
        train_trades = trades_sorted[:split_idx]
        val_trades   = trades_sorted[split_idx:]

        # ── Compute new threshold on training set ─────────────────────────────
        new_threshold = self._compute_exit_threshold(train_trades)

        # ── Walk-forward validation ───────────────────────────────────────────
        validation = self._validate_threshold(train_trades, val_trades, new_threshold)

        # ── Decide: apply or revert ───────────────────────────────────────────
        previous = self.load_or_defaults(self.data_dir)
        prev_thresh = previous.get("weak_signal_loss_threshold_pct", DEFAULTS["weak_signal_loss_threshold_pct"])

        if validation["reverted"]:
            applied_threshold = prev_thresh
            logger.warning(
                f"ThresholdCalibrator: validation failed — reverting to previous "
                f"threshold {prev_thresh:.3f}% "
                f"(val_pnl_per_trade={validation['val_pnl_per_trade']:.2f} vs "
                f"train_pnl_per_trade={validation['train_pnl_per_trade']:.2f})"
            )
        else:
            applied_threshold = new_threshold

        result = DEFAULTS.copy()
        result["weak_signal_loss_threshold_pct"] = applied_threshold
        result["no_signal_loss_threshold_pct"] = round(applied_threshold * 0.10, 4)
        bl, wl = self._compute_slippage_lists()
        result["slippage_blacklist"] = bl
        result["slippage_watchlist"] = wl
        result["symbol_probation"] = self._compute_symbol_probation(trades)
        result["calibrated_at"] = datetime.utcnow().isoformat()
        result["validation"] = validation

        # Persist to disk
        try:
            self.out_file.write_text(json.dumps(result, indent=2))
        except Exception as e:
            logger.warning(f"ThresholdCalibrator: failed to persist calibration: {e}")

        logger.info(
            f"ThresholdCalibrator: exit_thresh={result['weak_signal_loss_threshold_pct']:.3f}% "
            f"({'reverted' if validation['reverted'] else 'applied'}), "
            f"blacklist={result['slippage_blacklist']}"
        )
        return result

    @staticmethod
    def load_or_defaults(data_dir: Path) -> Dict:
        """
        Load persisted calibration file. Never raises — always returns a valid dict.
        Call this at startup for instant threshold loading without re-running calibration.
        """
        try:
            f = Path(data_dir) / "calibrated_thresholds.json"
            if f.exists():
                d = json.loads(f.read_text())
                # Backfill any keys added after initial calibration
                for k, v in DEFAULTS.items():
                    d.setdefault(k, v)
                return d
        except Exception as e:
            logger.warning(f"ThresholdCalibrator: failed to load persisted calibration: {e}")
        return DEFAULTS.copy()

    # ── Private helpers ───────────────────────────────────────────────────────

    def _load_closed_trades(self) -> List[Dict]:
        """Load closed_trades list from performance_attribution.json.

        Excludes startup_restore trades — those were opened with a hardcoded
        entry_signal=±0.5 placeholder that bears no relation to real ML output.
        Including them corrupts threshold estimates (all closed at loss → val_win_rate=0).

        The "source" field on closed trades records the EXIT source ("live_exit"),
        not the entry source.  The reliable filter is entry_signal not in (0.5, -0.5).
        """
        if not self.perf_file.exists():
            return []
        try:
            data = json.loads(self.perf_file.read_text())
            all_trades = data.get("closed_trades", [])
            # Filter out startup_restore trades: they used a hardcoded entry_signal of ±0.5
            # which is unrelated to real ML output. The "source" field on closed trades records
            # the EXIT source ("live_exit"), not the entry source, so we must filter by
            # entry_signal value instead.
            live_trades = [
                t for t in all_trades
                if t.get("net_pnl") is not None
                and abs(float(t.get("entry_signal", 0.5) or 0.5)) != 0.5
            ]
            excluded = len(all_trades) - len(live_trades)
            if excluded:
                logger.info(
                    f"ThresholdCalibrator: excluded {excluded} startup_restore trades "
                    f"(entry_signal=±0.5 fake signal; "
                    f"{len(live_trades)} genuine ML trades remain for calibration)"
                )
            return live_trades
        except Exception as e:
            logger.warning(f"ThresholdCalibrator: cannot read trade history: {e}")
            return []

    def _validate_threshold(
        self,
        train_trades: List[Dict],
        val_trades: List[Dict],
        new_threshold: float,
    ) -> Dict:
        """
        Walk-forward validation: compare key metrics between training and
        validation sets to decide whether the new threshold should be applied.

        Metrics compared:
        - net_pnl_per_trade: did profitability improve in validation period?
        - excellence_exit_rate: did premature exits decrease?

        Reverts if val_pnl_per_trade degrades by >VALIDATION_DEGRADE_THRESHOLD
        AND val excellence_exit_rate is worse than training.
        """

        def _metrics(trades_list: List[Dict]) -> Dict:
            if not trades_list:
                return {"n": 0, "net_pnl_per_trade": 0.0, "excellence_rate": 0.0, "win_rate": 0.0}
            n = len(trades_list)
            total_pnl = sum(float(t.get("net_pnl", 0)) for t in trades_list)
            wins = sum(1 for t in trades_list if float(t.get("net_pnl", 0)) > 0)
            excellence_exits = sum(
                1 for t in trades_list
                if "Excellence" in str(t.get("exit_reason", ""))
                or "Weak signal" in str(t.get("exit_reason", ""))
                or "No signal" in str(t.get("exit_reason", ""))
            )
            return {
                "n": n,
                "net_pnl_per_trade": round(total_pnl / n, 2),
                "excellence_rate": round(excellence_exits / n, 3),
                "win_rate": round(wins / n, 3),
            }

        train_m = _metrics(train_trades)
        val_m   = _metrics(val_trades)

        # Revert if val pnl/trade is much worse AND excellence rate worsened
        pnl_degraded = False
        excellence_worsened = False
        reverted = False

        if val_m["n"] >= 3:  # need at least 3 val trades to make a decision
            if train_m["net_pnl_per_trade"] != 0:
                degrade = (val_m["net_pnl_per_trade"] - train_m["net_pnl_per_trade"]) / abs(train_m["net_pnl_per_trade"])
                pnl_degraded = degrade < -VALIDATION_DEGRADE_THRESHOLD
            excellence_worsened = val_m["excellence_rate"] > train_m["excellence_rate"] + 0.10
            # Revert only if BOTH signals are negative (avoid false positives)
            reverted = pnl_degraded and excellence_worsened

        return {
            "train_n": train_m["n"],
            "val_n": val_m["n"],
            "train_pnl_per_trade": train_m["net_pnl_per_trade"],
            "val_pnl_per_trade": val_m["net_pnl_per_trade"],
            "train_excellence_rate": train_m["excellence_rate"],
            "val_excellence_rate": val_m["excellence_rate"],
            "train_win_rate": train_m["win_rate"],
            "val_win_rate": val_m["win_rate"],
            "new_threshold": new_threshold,
            "pnl_degraded": pnl_degraded,
            "excellence_worsened": excellence_worsened,
            "reverted": reverted,
        }

    def _compute_exit_threshold(self, trades: List[Dict]) -> float:
        """
        Derive exit threshold from winning trade returns.

        Logic: winning trades tell us how much a trade can move in our favour.
        We set the exit threshold to 30% of that move × scale factor, so we only
        exit when the loss is meaningful relative to the expected winner payoff.

        Floor/ceil prevent extreme values.
        """
        win_pnl_pcts: List[float] = []
        for t in trades:
            # Prefer pnl_pct field; fall back to net_pnl / entry_notional
            pnl_pct = t.get("pnl_pct")
            if pnl_pct is None:
                net_pnl = t.get("net_pnl", 0)
                notional = t.get("entry_notional") or t.get("entry_price", 1) * t.get("quantity", 1)
                notional = max(float(notional), 1.0)
                pnl_pct = float(net_pnl) / notional * 100
            else:
                pnl_pct = float(pnl_pct)

            if pnl_pct > 0:
                win_pnl_pcts.append(pnl_pct)

        if not win_pnl_pcts:
            logger.info("ThresholdCalibrator: no winning trades found, using default threshold")
            return DEFAULTS["weak_signal_loss_threshold_pct"]

        median_win = float(np.median(win_pnl_pcts))
        threshold = -(median_win * ATR_SCALE_FACTOR * 0.30)
        # Clamp to [−ATR_CEIL_PCT, −ATR_FLOOR_PCT]
        threshold = max(-ATR_CEIL_PCT, min(-ATR_FLOOR_PCT, threshold))
        logger.info(
            f"ThresholdCalibrator: median_win={median_win:.2f}%, "
            f"raw_thresh={-(median_win*ATR_SCALE_FACTOR*0.30):.3f}%, "
            f"clamped_thresh={threshold:.3f}%"
        )
        return round(threshold, 3)

    def _compute_symbol_probation(self, trades: List[Dict]) -> Dict[str, float]:
        """
        Derive per-symbol signal thresholds from live win rates.

        Tiers:
          < PROBATION_MIN_TRADES live trades  → 0.40 (no history, apply light caution)
          win_rate < 30%                       → 0.45 (poor performer)
          win_rate 30-45%                      → 0.35 (below-average performer)
          win_rate >= 45%                      → no entry (symbol passes normal threshold)

        Only crypto symbols are probated (equities have different signal dynamics).
        Returns {symbol: min_signal_threshold}.
        """
        from collections import defaultdict
        sym_pnls: Dict[str, List[float]] = defaultdict(list)
        for t in trades:
            sym = t.get("symbol", "")
            pnl = t.get("net_pnl")
            if sym and pnl is not None:
                sym_pnls[sym].append(float(pnl))

        probation: Dict[str, float] = {}
        for sym, pnls in sym_pnls.items():
            # Only probate crypto symbols
            if not (sym.upper().startswith("CRYPTO:") or "/" in sym):
                continue
            n = len(pnls)
            if n < PROBATION_MIN_TRADES:
                probation[sym] = PROBATION_THRESH_NONE_DATA
                logger.info(
                    f"ThresholdCalibrator: {sym} probation (insufficient data: {n} trades) "
                    f"→ min_signal={PROBATION_THRESH_NONE_DATA}"
                )
            else:
                wins = sum(1 for p in pnls if p > 0)
                win_rate = wins / n
                # Significance gate: only probate if win-rate deviation is statistically real
                try:
                    from monitoring.stat_significance import is_significant as _is_sig
                    _sig = _is_sig(wins=wins, n=n, alpha=0.10)  # 90% CI for probation (less strict)
                except ImportError:
                    _sig = True  # no gate available → allow as before
                if not _sig:
                    logger.debug(
                        "ThresholdCalibrator: %s WR=%.0f%% (n=%d) not significant — skipping probation",
                        sym, win_rate * 100, n,
                    )
                    continue
                if win_rate < PROBATION_WIN_RATE_LO:
                    probation[sym] = PROBATION_THRESH_STRONG
                    logger.info(
                        f"ThresholdCalibrator: {sym} strong probation "
                        f"(win_rate={win_rate:.0%}, n={n}) → min_signal={PROBATION_THRESH_STRONG}"
                    )
                elif win_rate < PROBATION_WIN_RATE_MID:
                    probation[sym] = PROBATION_THRESH_LIGHT
                    logger.info(
                        f"ThresholdCalibrator: {sym} light probation "
                        f"(win_rate={win_rate:.0%}, n={n}) → min_signal={PROBATION_THRESH_LIGHT}"
                    )
                # else: win_rate >= 45% → graduated out of probation, no entry

        return probation

    def _compute_slippage_lists(self) -> Tuple[List[str], List[str]]:
        """
        Read execution_latency.jsonl and compute per-symbol average slippage.
        Returns (blacklist, watchlist).
        """
        if not self.latency_file.exists():
            return [], []

        slippage_by_sym: Dict[str, List[float]] = {}
        try:
            for line in self.latency_file.read_text().splitlines():
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                sym = rec.get("symbol", "")
                bps = rec.get("slippage_bps")
                if sym and bps is not None:
                    slippage_by_sym.setdefault(sym, []).append(float(bps))
        except Exception as e:
            logger.warning(f"ThresholdCalibrator: cannot read latency file: {e}")
            return [], []

        blacklist: List[str] = []
        watchlist: List[str] = []
        for sym, vals in slippage_by_sym.items():
            avg = float(np.mean(vals))
            if avg > SLIPPAGE_BAN_BPS:
                blacklist.append(sym)
                logger.info(f"ThresholdCalibrator: {sym} blacklisted (avg={avg:.1f} bps > {SLIPPAGE_BAN_BPS})")
            elif avg > SLIPPAGE_WATCH_BPS:
                watchlist.append(sym)
                logger.info(f"ThresholdCalibrator: {sym} on watchlist (avg={avg:.1f} bps > {SLIPPAGE_WATCH_BPS})")

        return sorted(blacklist), sorted(watchlist)
