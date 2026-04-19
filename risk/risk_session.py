"""
risk/risk_session.py - User-Specific Risk State

Encapsulates the risk state for a single user/session, including:
- Capital tracking (starting, peak, day_start)
- Circuit breaker state
- Daily loss and drawdown logic
"""

import asyncio
import logging
import json
from datetime import datetime, timedelta
from typing import Dict, Tuple
import collections
from pathlib import Path

from config import ApexConfig
from monitoring.alert_aggregator import fire_alert, AlertSeverity as AlertSev

logger = logging.getLogger(__name__)

class CircuitBreaker:
    """
    Circuit breaker to halt trading during adverse conditions.
    """

    def __init__(self):
        self.is_tripped = False
        self.trip_reason = None
        self.trip_time = None
        self.consecutive_losses = 0
        self.recent_trades: collections.deque = collections.deque(maxlen=20)
        # Round 8 / GAP-12C: bar-based cooldown counter.
        # Incremented once per execution-loop cycle via ``tick_bar()`` when the
        # breaker is tripped. Zeroed on every trip and on every reset.
        self.bars_since_trip: int = 0

    def record_trade(self, pnl: float):
        """Record a trade for consecutive loss tracking."""
        self.recent_trades.append({
            'timestamp': datetime.now(),
            'pnl': pnl
        })

        # Only count a trade as a "loss" if the absolute loss exceeds the minimum
        # threshold. This prevents micro-losses from partial fills (e.g., 10 TWAP
        # tranches each losing $5) from falsely tripping the circuit breaker.
        min_loss = float(getattr(ApexConfig, 'CIRCUIT_BREAKER_MIN_LOSS_USD', 25.0))
        if pnl < -min_loss:
            self.consecutive_losses += 1
        elif pnl > 0:
            # Only reset on a meaningful profit (any positive PnL resets the streak)
            self.consecutive_losses = 0
        # pnl in [-min_loss, 0]: micro-loss — don't increment OR reset the counter

        # Check consecutive loss limit
        if ApexConfig.CIRCUIT_BREAKER_ENABLED:
            if self.consecutive_losses >= ApexConfig.CIRCUIT_BREAKER_CONSECUTIVE_LOSSES:
                self.trip(f"Consecutive losses: {self.consecutive_losses}")

    def trip(self, reason: str):
        """Trip the circuit breaker."""
        if not self.is_tripped:
            self.is_tripped = True
            self.trip_reason = reason
            self.trip_time = datetime.now()
            self.bars_since_trip = 0
            logger.error(f"🚨 CIRCUIT BREAKER TRIPPED: {reason}")
            logger.error(f"   Trading halted at {self.trip_time}")
            logger.error(f"   Cooldown: {ApexConfig.CIRCUIT_BREAKER_COOLDOWN_HOURS} hours")
            logger.error(
                f"   Bar cooldown: {ApexConfig.CIRCUIT_BREAKER_COOLDOWN_BARS} bars"
            )
            fire_alert("circuit_breaker", f"Circuit breaker TRIPPED: {reason}", AlertSev.CRITICAL)

    def tick_bar(self) -> bool:
        """
        Advance the bar-cooldown counter by one execution cycle
        (Round 8 / GAP-12C).

        The caller invokes this once per execution-loop cycle *regardless*
        of whether the breaker is tripped. When the breaker is tripped and
        ``CIRCUIT_BREAKER_COOLDOWN_BARS > 0``, this method auto-resets the
        breaker once ``bars_since_trip >= CIRCUIT_BREAKER_COOLDOWN_BARS``.
        This is ORed with the hours-based cooldown in
        :meth:`check_and_reset` — whichever expires first clears the halt.

        Returns:
            ``True`` if the breaker is now clear (either it was never
            tripped, or this tick triggered the auto-reset). ``False``
            while the breaker remains tripped.
        """
        if not self.is_tripped:
            return True
        self.bars_since_trip += 1
        cooldown_bars = int(
            getattr(ApexConfig, "CIRCUIT_BREAKER_COOLDOWN_BARS", 0) or 0
        )
        if cooldown_bars > 0 and self.bars_since_trip >= cooldown_bars:
            logger.info(
                "✅ Circuit breaker bar cooldown complete (%d/%d bars) — trading resumed",
                self.bars_since_trip, cooldown_bars,
            )
            self.reset()
            fire_alert(
                "circuit_breaker",
                f"Circuit breaker bar cooldown complete "
                f"({self.bars_since_trip}/{cooldown_bars} bars) — trading resumed",
                AlertSev.INFO,
            )
            return True
        return False

    def check_and_reset(self) -> bool:
        """
        Check if circuit breaker can be reset.

        Returns:
            True if trading is allowed, False if still tripped
        """
        if not self.is_tripped:
            return True

        if self.trip_time is None:
            return True

        # Check cooldown period
        cooldown = timedelta(hours=ApexConfig.CIRCUIT_BREAKER_COOLDOWN_HOURS)
        if datetime.now() - self.trip_time >= cooldown:
            logger.info("✅ Circuit breaker cooldown complete - trading resumed")
            self.reset()
            fire_alert("circuit_breaker", "Circuit breaker cooldown complete — trading resumed", AlertSev.INFO)
            return True

        remaining = cooldown - (datetime.now() - self.trip_time)
        logger.warning(
            f"⏳ Circuit breaker active - {remaining.total_seconds() / 3600:.1f}h remaining"
        )
        return False

    def check_early_reset(
        self,
        window_hours: float = 2.0,
        max_loss_usd: float = 50.0,
        daily_loss_pct: float = 0.0,
        max_daily_loss_pct: float = 0.015,
    ) -> bool:
        """
        Attempt an early reset of the circuit breaker.

        Conditions (all must be true):
        1. CB is currently tripped.
        2. daily_loss_pct < max_daily_loss_pct (day not still bleeding badly).
        3. No trade in the last `window_hours` had a loss > `max_loss_usd`.

        Returns True if reset was performed.
        """
        if not self.is_tripped:
            return False
        if daily_loss_pct >= max_daily_loss_pct:
            return False
        cutoff = datetime.now() - timedelta(hours=window_hours)
        recent_big_losses = [
            t for t in self.recent_trades
            if t["timestamp"] > cutoff and t["pnl"] < -max_loss_usd
        ]
        if recent_big_losses:
            return False
        # All conditions met — reset
        self.is_tripped = False
        self.trip_reason = None
        self.trip_time = None
        logger.info(
            f"CircuitBreaker: early reset approved "
            f"(no losses >${max_loss_usd:.0f} in last {window_hours:.1f}h, "
            f"daily_loss={daily_loss_pct:.2%} < {max_daily_loss_pct:.2%})"
        )
        return True

    def reset(self):
        """Reset the circuit breaker."""
        self.is_tripped = False
        self.trip_reason = None
        self.trip_time = None
        self.consecutive_losses = 0
        self.bars_since_trip = 0
        logger.info("🔄 Circuit breaker reset")

    def get_status(self) -> Dict:
        """Get circuit breaker status."""
        return {
            'is_tripped': self.is_tripped,
            'reason': self.trip_reason,
            'trip_time': self.trip_time.isoformat() if self.trip_time else None,
            'consecutive_losses': self.consecutive_losses,
            'recent_trades': len(self.recent_trades),
            'bars_since_trip': int(self.bars_since_trip),
        }


class RiskSession:
    """
    Manages risk state for a specific user session.
    """

    def __init__(self, user_id: str, max_daily_loss: float = 0.02, max_drawdown: float = 0.10, session_type: str = "unified"):
        self.user_id = user_id
        self.session_type = session_type
        self.max_daily_loss = max_daily_loss
        self.max_drawdown = max_drawdown

        self.starting_capital: float = 0.0
        self.peak_capital: float = 0.0
        self.day_start_capital: float = 0.0
        self.current_day: str = datetime.now().strftime('%Y-%m-%d')

        self.circuit_breaker = CircuitBreaker()

        # Crypto uses a rolling 24h loss window (not calendar-day) because crypto trades 24/7
        self._crypto_24h_ref_capital: float = 0.0
        self._crypto_24h_ref_time: datetime = datetime.utcnow()

        # ── 60-minute rolling intraday drawdown gate ──────────────────────
        # Stores (utc_timestamp, capital_value) tuples — maxlen caps memory at
        # ~1 value/s × 3600s = 3600 entries max even in the hottest loop.
        # NOT persisted to disk: ephemeral intraday state, intentionally resets
        # on engine restart to avoid a cold-start false-trigger.
        self._intraday_pnl_snapshots: collections.deque = collections.deque(maxlen=3600)

        # Rate-limit the heal_baselines implausible-ratio warning (once per hour)
        self._last_implausible_warn_ts: float = 0.0

        # Load state on init
        self.load_state()

    def heal_baselines(self, current_capital: float, source: str = "runtime") -> bool:
        """Self-heal invalid baseline state."""
        try:
            value = float(current_capital)
        except Exception:
            return False

        if value <= 0:
            return False

        changed = False
        today = datetime.now().strftime('%Y-%m-%d')

        if self.starting_capital <= 0:
            self.starting_capital = value
            changed = True

        if self.peak_capital <= 0:
            self.peak_capital = value
            changed = True

        if self.current_day != today:
            self.current_day = today

        if self.day_start_capital <= 0:
            self.day_start_capital = value
            changed = True
        else:
            # Sanity check: if persisted day_start_capital is more than 60% away from
            # current capital it means it was saved under a completely different equity
            # level (e.g., a paper-startup value of $8k vs real $100k) — reset it.
            ratio = self.day_start_capital / value
            if ratio < 0.40 or ratio > 2.50:
                import time as _time
                now = _time.monotonic()
                if now - self._last_implausible_warn_ts > 3600:  # warn at most once/hour
                    logger.warning(
                        f"[RiskSession/{self.user_id}] day_start_capital {self.day_start_capital:,.0f} "
                        f"is implausible vs current {value:,.0f} (ratio={ratio:.2f}). "
                        f"Resetting all baselines to current value to prevent false circuit trip."
                    )
                    self._last_implausible_warn_ts = now
                self.day_start_capital = value
                self.starting_capital = value
                self.peak_capital = value
                changed = True

        return changed

    def _get_state_file(self) -> Path:
        """Get path to the risk state file for this session."""
        if self.user_id == "default":
             return ApexConfig.DATA_DIR / "risk_state.json"

        # Ensure user directory exists
        user_dir = ApexConfig.DATA_DIR / "users" / self.user_id
        user_dir.mkdir(parents=True, exist_ok=True)
        # In dual-session mode, namespace by session_type to prevent core ($1.26M)
        # and crypto ($80K) sessions from overwriting each other's baseline.
        if self.session_type not in ("unified", ""):
            return user_dir / f"risk_state_{self.session_type}.json"
        return user_dir / "risk_state.json"

    def save_state(self):
        """Save risk state to disk and mirror to Redis (fire-and-forget)."""
        try:
            state = {
                'day_start_capital': self.day_start_capital,
                'peak_capital': self.peak_capital,
                'starting_capital': self.starting_capital,
                'current_day': self.current_day,
                'circuit_breaker': self.circuit_breaker.get_status()
            }

            state_file = self._get_state_file()
            # Ensure parent dir exists (redundant for default but good for users)
            state_file.parent.mkdir(parents=True, exist_ok=True)

            with open(state_file, "w") as f:
                json.dump(state, f, indent=2)

            # Mirror to Redis so async readers get sub-ms access.
            # Uses ensure_future — safe to call from both sync and async contexts.
            try:
                import asyncio
                from services.common.redis_client import cache_set as _redis_set
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    asyncio.ensure_future(
                        _redis_set(f"apex:risk:{self.user_id}", state, ttl_seconds=3600)
                    )
            except Exception:
                pass  # Redis unavailable — JSON file is the durable source of truth

        except Exception as e:
            logger.error(f"Error saving risk state for {self.user_id}: {e}")

    async def save_state_async(self):
        """Non-blocking version of save_state for use inside the async event loop."""
        await asyncio.to_thread(self.save_state)

    def load_state(self):
        """Load risk state from disk."""
        try:
            state_file = self._get_state_file()
            if not state_file.exists():
                return
            
            with open(state_file, "r") as f:
                state = json.load(f)
            
            today = datetime.now().strftime('%Y-%m-%d')
            if state.get('current_day') == today:
                self.day_start_capital = float(state.get('day_start_capital', 0))
                self.current_day = today
            else:
                pass # New day logic happens in heal or check methods
            
            self.peak_capital = float(state.get('peak_capital', self.peak_capital))
            self.starting_capital = float(state.get('starting_capital', self.starting_capital))
            
            cb_state = state.get('circuit_breaker', {})
            if cb_state.get('is_tripped'):
                self.circuit_breaker.is_tripped = True
                self.circuit_breaker.trip_reason = cb_state.get('reason')
                self.circuit_breaker.trip_time = datetime.fromisoformat(cb_state['trip_time']) if cb_state.get('trip_time') else None

        except Exception as e:
            logger.error(f"Error loading risk state for {self.user_id}: {e}")

    def can_trade(self) -> Tuple[bool, str]:
        """Check if trading is allowed."""
        if not ApexConfig.CIRCUIT_BREAKER_ENABLED:
            return True, "Circuit breaker disabled"

        if self.circuit_breaker.check_and_reset():
            return True, "OK"
        else:
            return False, f"Circuit breaker tripped: {self.circuit_breaker.trip_reason}"

    def record_trade_result(self, pnl: float):
        """Record trade result. Persistence is deferred to the async save cycle."""
        self.circuit_breaker.record_trade(pnl)
        # Do NOT call save_state() here — it blocks the event loop.
        # The execution loop calls save_state_async() on every risk-check tick.

    def manual_reset_circuit_breaker(self, requested_by: str = "admin", reason: str = "manual_reset") -> bool:
        """Manually reset the circuit breaker."""
        logger.info(f"🔄 Manual circuit breaker reset requested by {requested_by} for {self.user_id}. Reason: {reason}")
        self.circuit_breaker.reset()
        self.save_state()
        return True

    def check_early_circuit_breaker_reset(self, daily_loss_pct: float = 0.0) -> bool:
        """
        Attempt an early automatic circuit breaker reset using calibrated thresholds.
        Returns True if the CB was reset.
        """
        return self.circuit_breaker.check_early_reset(
            window_hours=getattr(ApexConfig, "CIRCUIT_BREAKER_EARLY_RESET_HOURS", 2.0),
            max_loss_usd=getattr(ApexConfig, "CIRCUIT_BREAKER_EARLY_RESET_MAX_LOSS_USD", 50.0),
            daily_loss_pct=daily_loss_pct,
            max_daily_loss_pct=getattr(ApexConfig, "CIRCUIT_BREAKER_EARLY_RESET_MAX_DAILY_LOSS_PCT", 0.015),
        )

    def set_starting_capital(self, capital: float):
        """Set starting capital and initialize tracking."""
        try:
            capital = float(capital)
        except Exception:
            return
        if capital <= 0:
            return

        self.starting_capital = capital

        if self.peak_capital == 0:
            self.peak_capital = capital

        today = datetime.now().strftime('%Y-%m-%d')
        if self.day_start_capital == 0 or self.current_day != today:
            self.day_start_capital = capital
            self.current_day = today
        # Defer persistence — caller is in async context; periodic save_state_async handles it.

    def check_daily_loss(self, current_value: float) -> Dict:
        """Check if daily loss limit breached."""
        try:
            current_value = float(current_value)
            self.heal_baselines(current_capital=current_value, source="check_daily_loss")

            today = datetime.now().strftime('%Y-%m-%d')
            if today != self.current_day:
                self.current_day = today
                self.day_start_capital = current_value

            daily_pnl = current_value - self.day_start_capital
            daily_return = daily_pnl / self.day_start_capital if self.day_start_capital > 0 else 0

            # Guard: if daily_return is implausibly extreme (>50% loss or gain in one day)
            # it means day_start_capital is stale/corrupt — reset and skip the check.
            if abs(daily_return) > 0.50:
                logger.warning(
                    f"[RiskSession/{self.user_id}] Implausible daily_return {daily_return:.2%} "
                    f"(start={self.day_start_capital:,.0f}, current={current_value:,.0f}). "
                    f"Treating as stale baseline — resetting day_start_capital."
                )
                self.day_start_capital = current_value
                return {
                    'daily_pnl': 0.0,
                    'daily_return': 0.0,
                    'breached': False,
                    'limit': self.max_daily_loss,
                    'circuit_breaker_tripped': self.circuit_breaker.is_tripped
                }

            breached = daily_return < -self.max_daily_loss

            if ApexConfig.CIRCUIT_BREAKER_ENABLED:
                if daily_return < -ApexConfig.CIRCUIT_BREAKER_DAILY_LOSS:
                    self.circuit_breaker.trip(
                        f"Daily loss {daily_return*100:.2f}% exceeds {ApexConfig.CIRCUIT_BREAKER_DAILY_LOSS*100:.1f}%"
                    )

            if breached:
                logger.error(f"🚨 DAILY LOSS LIMIT BREACHED for {self.user_id}!")

            return {
                'daily_pnl': daily_pnl,
                'daily_return': daily_return,
                'breached': breached,
                'limit': self.max_daily_loss,
                'circuit_breaker_tripped': self.circuit_breaker.is_tripped
            }

        except Exception as e:
            logger.error(f"Error checking daily loss for {self.user_id}: {e}")
            return {'daily_pnl': 0, 'daily_return': 0, 'breached': False, 'limit': self.max_daily_loss}

    def check_crypto_rolling_loss(self, current_value: float) -> Dict:
        """
        Check crypto P&L against a rolling 24-hour window.
        Unlike equities, crypto trades 24/7 so a calendar-day reset at midnight is meaningless.
        The reference capital resets every 24h from the first time this is called.
        """
        try:
            current_value = float(current_value)
            now = datetime.utcnow()
            limit = getattr(ApexConfig, "CRYPTO_MAX_DAILY_LOSS", 0.05)

            # Initialise reference on first call or after 24h window expires
            if self._crypto_24h_ref_capital <= 0:
                self._crypto_24h_ref_capital = current_value
                self._crypto_24h_ref_time = now

            elapsed_hours = (now - self._crypto_24h_ref_time).total_seconds() / 3600
            if elapsed_hours >= 24:
                logger.info(
                    "[RiskSession/%s] Crypto 24h window expired (%.1fh); resetting reference capital.",
                    self.user_id, elapsed_hours,
                )
                self._crypto_24h_ref_capital = current_value
                self._crypto_24h_ref_time = now

            pnl_24h = current_value - self._crypto_24h_ref_capital
            ret_24h = pnl_24h / self._crypto_24h_ref_capital if self._crypto_24h_ref_capital > 0 else 0.0
            breached = ret_24h < -limit

            if breached:
                logger.error(
                    "🚨 CRYPTO 24H LOSS LIMIT BREACHED for %s: %.2f%% (limit %.1f%%)",
                    self.user_id, ret_24h * 100, limit * 100,
                )

            return {
                "crypto_pnl_24h": pnl_24h,
                "crypto_return_24h": ret_24h,
                "breached": breached,
                "limit": limit,
                "window_start": self._crypto_24h_ref_time.isoformat(),
                "hours_elapsed": round(elapsed_hours, 2),
            }
        except Exception as e:
            logger.error(f"Error checking crypto rolling loss for {self.user_id}: {e}")
            return {"crypto_pnl_24h": 0.0, "crypto_return_24h": 0.0, "breached": False,
                    "limit": getattr(ApexConfig, "CRYPTO_MAX_DAILY_LOSS", 0.05)}

    def push_capital_snapshot(self, capital: float) -> None:
        """Record a capital snapshot for the rolling intraday drawdown gate.

        Called every risk-check cycle (typically every 60 s).  Pure in-memory
        append — zero I/O, zero blocking.  The deque is bounded (maxlen=3600)
        so memory is always O(1) regardless of session length.

        Stale broker values (repeated identical readings during an API outage)
        are safe: the gate computes the *peak* inside the window, so duplicate
        readings simply preserve the last known high-water mark without
        artificially depressing it.
        """
        try:
            val = float(capital)
        except (TypeError, ValueError):
            return
        if val > 0:
            self._intraday_pnl_snapshots.append((datetime.utcnow(), val))

    def check_intraday_rolling_dd(self, current_value: float) -> Dict:
        """Check whether the rolling intraday drawdown gate should block new entries.

        Scans every capital snapshot taken within the last WINDOW_MINUTES and
        finds the peak.  If (current_value - peak) / peak < -MAX_LOSS_PCT the
        gate is *active* — the execution loop will refuse NEW entries.

        CRITICAL INVARIANT: this method NEVER blocks exits.  The caller is
        responsible for only consulting this result for entry decisions.

        Returns a dict with at minimum:
            breached (bool), rolling_loss_pct (float), window_peak (float),
            window_minutes (int), limit (float)
        """
        _default = {
            "breached": False,
            "rolling_loss_pct": 0.0,
            "window_peak": float(current_value or 0),
            "window_minutes": int(getattr(ApexConfig, "INTRADAY_DD_WINDOW_MINUTES", 60)),
            "limit": float(getattr(ApexConfig, "INTRADAY_DD_MAX_LOSS_PCT", 0.015)),
        }

        if not getattr(ApexConfig, "INTRADAY_DD_GATE_ENABLED", True):
            return _default

        try:
            current_value = float(current_value)
        except (TypeError, ValueError):
            return _default

        if current_value <= 0:
            return _default

        window_min = int(getattr(ApexConfig, "INTRADAY_DD_WINDOW_MINUTES", 60))
        max_loss   = float(getattr(ApexConfig, "INTRADAY_DD_MAX_LOSS_PCT", 0.015))

        if not self._intraday_pnl_snapshots:
            return _default

        # Only consider snapshots within the rolling window.
        # Using a list comprehension over a bounded deque is O(n) with n ≤ 3600
        # (≈ 3 600 floats) — negligible latency even at 1-second tick rate.
        cutoff = datetime.utcnow() - timedelta(minutes=window_min)
        window_values = [v for ts, v in self._intraday_pnl_snapshots if ts >= cutoff]

        if not window_values:
            return _default

        window_peak = max(window_values)
        if window_peak <= 0:
            return _default

        rolling_loss_pct = (current_value - window_peak) / window_peak  # negative = loss
        breached = rolling_loss_pct < -max_loss

        if breached:
            logger.warning(
                "🚨 INTRADAY DD GATE [%s]: rolling loss %.2f%% exceeds limit %.1f%% "
                "(window=%d min, peak=$%,.2f now=$%,.2f) — new entries BLOCKED",
                self.user_id,
                rolling_loss_pct * 100,
                max_loss * 100,
                window_min,
                window_peak,
                current_value,
            )

        return {
            "breached":          breached,
            "rolling_loss_pct":  rolling_loss_pct,
            "window_peak":       window_peak,
            "window_minutes":    window_min,
            "limit":             max_loss,
        }

    def check_drawdown(self, current_value: float) -> Dict:
        """Check if drawdown limit breached."""
        try:
            current_value = float(current_value)
            self.heal_baselines(current_capital=current_value, source="check_drawdown")

            if current_value > self.peak_capital:
                self.peak_capital = current_value

            drawdown = (current_value - self.peak_capital) / self.peak_capital if self.peak_capital > 0 else 0
            breached = drawdown < -self.max_drawdown

            if ApexConfig.CIRCUIT_BREAKER_ENABLED:
                if drawdown < -ApexConfig.CIRCUIT_BREAKER_DRAWDOWN:
                    self.circuit_breaker.trip(
                        f"Drawdown {abs(drawdown)*100:.2f}% exceeds {ApexConfig.CIRCUIT_BREAKER_DRAWDOWN*100:.1f}%"
                    )

            if breached:
                logger.error(f"🚨 MAX DRAWDOWN BREACHED for {self.user_id}!")

            return {
                'drawdown': abs(drawdown),
                'breached': breached,
                'peak': self.peak_capital,
                'current': current_value,
                'limit': self.max_drawdown,
                'circuit_breaker_tripped': self.circuit_breaker.is_tripped
            }

        except Exception as e:
            logger.error(f"Error checking drawdown for {self.user_id}: {e}")
            return {'drawdown': 0, 'breached': False, 'peak': self.peak_capital, 'current': current_value, 'limit': self.max_drawdown}
