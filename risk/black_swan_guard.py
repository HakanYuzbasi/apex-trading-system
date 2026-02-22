"""
risk/black_swan_guard.py - Real-Time Crash Detection & Emergency Response

Monitors for flash crashes, VIX spikes, and correlation breakdowns in real-time.
Implements 4 escalating threat levels with automatic defensive actions:

NORMAL    → Full trading
ELEVATED  → Block new entries, reduce size 50%
SEVERE    → Close worst 25% positions, size 25%
CRITICAL  → Emergency liquidation, all trading halted

Three independent detection triggers:
1. Price velocity (SPY drop rate)
2. VIX spike (intraday jump)
3. Correlation spike (risk-off herding)
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import IntEnum
from collections import deque
import logging

logger = logging.getLogger(__name__)


class ThreatLevel(IntEnum):
    NORMAL = 0
    ELEVATED = 1
    SEVERE = 2
    CRITICAL = 3


@dataclass
class CrashDetection:
    """Result of threat assessment."""
    threat_level: ThreatLevel
    triggers: List[str]
    portfolio_impact_pct: float
    recommended_action: str
    position_size_multiplier: float
    entry_blocked: bool
    block_duration_minutes: int
    detected_at: datetime = field(default_factory=datetime.now)


class BlackSwanGuard:
    """
    Real-time crash detection with escalating defensive responses.

    Monitors price velocity, VIX spikes, and correlation breakdowns.
    Auto-escalates through threat levels and auto-recovers when conditions normalize.
    """

    def __init__(
        self,
        crash_velocity_10m: float = 0.02,
        crash_velocity_30m: float = 0.04,
        vix_spike_elevated: float = 0.30,
        vix_spike_severe: float = 0.50,
        correlation_crisis_threshold: float = 0.85,
    ):
        self.crash_velocity_10m = crash_velocity_10m
        self.crash_velocity_30m = crash_velocity_30m
        self.vix_spike_elevated = vix_spike_elevated
        self.vix_spike_severe = vix_spike_severe
        self.correlation_crisis_threshold = correlation_crisis_threshold

        # State
        self._threat_level = ThreatLevel.NORMAL
        self._threat_expiry: Optional[datetime] = None
        self._active_triggers: List[str] = []

        # Price history for velocity computation (multi-index)
        self._index_prices: Dict[str, deque] = {}
        self._index_timestamps: Dict[str, deque] = {}
        self._monitored_indices = ["SPY", "QQQ", "IWM"]
        
        for idx in self._monitored_indices:
            self._index_prices[idx] = deque(maxlen=200)
            self._index_timestamps[idx] = deque(maxlen=200)
            
        self._vix_open: Optional[float] = None
        self._vix_current: Optional[float] = None

        # Block entry tracking
        self._entry_blocked_until: Optional[datetime] = None

        logger.info(
            f"BlackSwanGuard initialized: "
            f"velocity_10m={crash_velocity_10m}, velocity_30m={crash_velocity_30m}"
        )

    def record_index_price(self, symbol: str, price: float, timestamp: Optional[datetime] = None):
        """Record an index price observation for velocity tracking."""
        if symbol not in self._monitored_indices:
             return # Ignore non-index symbols
             
        ts = timestamp or datetime.now()
        self._index_prices[symbol].append(price)
        self._index_timestamps[symbol].append(ts)

    def record_vix(self, vix_current: float, vix_open: Optional[float] = None):
        """Record current VIX level and today's open."""
        self._vix_current = vix_current
        if vix_open is not None:
            self._vix_open = vix_open

    def assess_threat(
        self,
        spy_prices: Optional[List[float]] = None,
        vix_level: Optional[float] = None,
        vix_open: Optional[float] = None,
        portfolio_correlations: Optional[List[float]] = None,
    ) -> CrashDetection:
        """
        Run all crash detection triggers and determine threat level.

        Args:
            spy_prices: Deprecated (kept for compat). Use record_index_price.
            vix_level: Current VIX level
            vix_open: Today's VIX open
            portfolio_correlations: List of pairwise correlations in portfolio

        Returns:
            CrashDetection with threat level and recommended actions
        """
        triggers = []

        # Update internal state from legacy params
        if spy_prices:
            for p in spy_prices:
                self.record_index_price("SPY", p)
        if vix_level is not None:
            self.record_vix(vix_level, vix_open)

        # Check price velocity
        velocity_threat = self._check_price_velocity()
        if velocity_threat is not None:
            triggers.append(f"price_velocity:{velocity_threat.name}")

        # Check VIX spike
        vix_threat = self._check_vix_spike()
        if vix_threat is not None:
            triggers.append(f"vix_spike:{vix_threat.name}")

        # Check correlation spike
        corr_threat = self._check_correlation_spike(portfolio_correlations)
        if corr_threat is not None:
            triggers.append(f"correlation_spike:{corr_threat.name}")

        # Determine overall threat level
        individual_levels = []
        if velocity_threat is not None:
            individual_levels.append(velocity_threat)
        if vix_threat is not None:
            individual_levels.append(vix_threat)
        if corr_threat is not None:
            individual_levels.append(corr_threat)

        if not individual_levels:
            new_level = ThreatLevel.NORMAL
        else:
            # Multi-trigger escalation: 2+ triggers = bump up one level
            max_level = max(individual_levels)
            if len(individual_levels) >= 2:
                new_level = ThreatLevel(min(max_level + 1, ThreatLevel.CRITICAL))
            else:
                new_level = max_level

        # Update state (only escalate, auto-deescalate after expiry)
        self._update_threat_level(new_level, triggers)

        # Compute actions based on current threat level
        size_mult = self._get_size_multiplier()
        entry_blocked = self.should_block_entry()
        block_mins = self._get_block_duration()
        action = self._get_recommended_action()
        impact = self._estimate_portfolio_impact()

        return CrashDetection(
            threat_level=self._threat_level,
            triggers=self._active_triggers.copy(),
            portfolio_impact_pct=impact,
            recommended_action=action,
            position_size_multiplier=size_mult,
            entry_blocked=entry_blocked,
            block_duration_minutes=block_mins,
        )

    def should_block_entry(self) -> bool:
        """Check if new entries should be blocked."""
        if self._threat_level >= ThreatLevel.ELEVATED:
            return True
        if self._entry_blocked_until and datetime.now() < self._entry_blocked_until:
            return True
        return False

    def get_position_size_multiplier(self) -> float:
        """Get position size multiplier for current threat level."""
        return self._get_size_multiplier()

    def get_positions_to_close(
        self,
        positions: Dict[str, Dict],
        prices: Optional[Dict[str, float]] = None,
    ) -> List[str]:
        """
        Get list of positions to close based on threat level.

        SEVERE: close worst 25% of positions
        CRITICAL: close ALL positions

        Args:
            positions: Dict of symbol -> position info (must have 'pnl' or 'entry_price')
            prices: Current prices for P&L calculation

        Returns:
            List of symbols to close, ordered worst-first
        """
        if self._threat_level < ThreatLevel.SEVERE:
            return []

        if not positions:
            return []

        # Sort by P&L (worst first)
        scored = []
        for symbol, pos in positions.items():
            pnl = pos.get("pnl", pos.get("unrealized_pnl", 0.0))
            if prices and symbol in prices and "entry_price" in pos:
                entry = pos["entry_price"]
                current = prices[symbol]
                qty = pos.get("quantity", pos.get("shares", 1))
                pnl = (current - entry) * qty
            scored.append((symbol, pnl))

        scored.sort(key=lambda x: x[1])

        if self._threat_level == ThreatLevel.CRITICAL:
            return [s for s, _ in scored]

        # SEVERE: close worst 25%
        n_close = max(1, len(scored) // 4)
        return [s for s, _ in scored[:n_close]]

    # ─── Internal Detection Methods ───────────────────────────────

    def _check_price_velocity(self) -> Optional[ThreatLevel]:
        """Check price drop velocity across monitored indices."""
        crashes_detected = []
        
        for symbol in self._monitored_indices:
            prices_deque = self._index_prices[symbol]
            times_deque = self._index_timestamps[symbol]
            
            if len(prices_deque) < 3:
                continue
                
            prices = list(prices_deque)
            timestamps = list(times_deque)
            current_price = prices[-1]
            now = timestamps[-1]
            
            symbol_threat = None
            
            # Check 10-minute window
            for i in range(len(timestamps) - 2, -1, -1):
                elapsed = (now - timestamps[i]).total_seconds()
                if elapsed >= 600:  # 10 minutes
                    ref_price = prices[i]
                    if ref_price > 0:
                        drop = (ref_price - current_price) / ref_price
                        if drop >= self.crash_velocity_10m:
                            symbol_threat = ThreatLevel.ELEVATED
                    break

            # Check 30-minute window
            for i in range(len(timestamps) - 2, -1, -1):
                elapsed = (now - timestamps[i]).total_seconds()
                if elapsed >= 1800:  # 30 minutes
                    ref_price = prices[i]
                    if ref_price > 0:
                        drop = (ref_price - current_price) / ref_price
                        if drop >= self.crash_velocity_30m:
                            # 30m crash is more severe
                            symbol_threat = ThreatLevel.SEVERE
                    break
            
            if symbol_threat:
                crashes_detected.append((symbol, symbol_threat))

        if not crashes_detected:
            return None
            
        # Analyze collective threat
        # If any index crashing -> ELEVATED
        # If ALL indices crashing -> SEVERE
        # If any index has SEVERE crash -> SEVERE
        
        max_severity = max(c[1] for c in crashes_detected)
        count = len(crashes_detected)
        
        if max_severity >= ThreatLevel.SEVERE:
            return ThreatLevel.SEVERE
            
        if count >= 2: # Multiple indices showing weakness
             return ThreatLevel.SEVERE
             
        return ThreatLevel.ELEVATED

    def _check_vix_spike(self) -> Optional[ThreatLevel]:
        """Check VIX intraday spike."""
        if self._vix_current is None or self._vix_open is None:
            return None
        if self._vix_open <= 0:
            return None

        pct_change = (self._vix_current - self._vix_open) / self._vix_open

        if pct_change >= self.vix_spike_severe:
            return ThreatLevel.SEVERE
        if pct_change >= self.vix_spike_elevated:
            return ThreatLevel.ELEVATED

        return None

    def _check_correlation_spike(
        self, correlations: Optional[List[float]]
    ) -> Optional[ThreatLevel]:
        """Check portfolio-wide correlation spike."""
        if not correlations or len(correlations) < 3:
            return None

        avg_corr = float(np.mean([abs(c) for c in correlations]))

        if avg_corr >= self.correlation_crisis_threshold:
            return ThreatLevel.ELEVATED

        return None

    def _update_threat_level(self, new_level: ThreatLevel, triggers: List[str]):
        """Update threat level with escalation/de-escalation logic."""
        now = datetime.now()

        if new_level > self._threat_level:
            # Escalate immediately
            self._threat_level = new_level
            self._active_triggers = triggers
            # Set expiry based on level
            durations = {
                ThreatLevel.ELEVATED: 10,
                ThreatLevel.SEVERE: 30,
                ThreatLevel.CRITICAL: 60,
            }
            mins = durations.get(new_level, 10)
            self._threat_expiry = now + timedelta(minutes=mins)
            self._entry_blocked_until = self._threat_expiry
            logger.warning(
                f"THREAT ESCALATED to {new_level.name}: triggers={triggers}"
            )

        elif new_level < self._threat_level:
            # Only de-escalate after expiry
            if self._threat_expiry and now >= self._threat_expiry:
                self._threat_level = new_level
                self._active_triggers = triggers
                if new_level == ThreatLevel.NORMAL:
                    self._threat_expiry = None
                    self._entry_blocked_until = None
                logger.info(f"Threat de-escalated to {new_level.name}")
        else:
            # Same level — refresh triggers
            self._active_triggers = triggers

    def _get_size_multiplier(self) -> float:
        """Position size multiplier per threat level."""
        return {
            ThreatLevel.NORMAL: 1.0,
            ThreatLevel.ELEVATED: 0.5,
            ThreatLevel.SEVERE: 0.25,
            ThreatLevel.CRITICAL: 0.0,
        }[self._threat_level]

    def _get_block_duration(self) -> int:
        """Entry block duration in minutes per threat level."""
        return {
            ThreatLevel.NORMAL: 0,
            ThreatLevel.ELEVATED: 10,
            ThreatLevel.SEVERE: 30,
            ThreatLevel.CRITICAL: 60,
        }[self._threat_level]

    def _get_recommended_action(self) -> str:
        """Human-readable action recommendation."""
        return {
            ThreatLevel.NORMAL: "Normal trading",
            ThreatLevel.ELEVATED: "Block new entries, reduce position sizes 50%",
            ThreatLevel.SEVERE: "Close worst 25% positions, reduce sizes 75%",
            ThreatLevel.CRITICAL: "EMERGENCY: Close ALL positions immediately",
        }[self._threat_level]

    def _estimate_portfolio_impact(self) -> float:
        """Rough estimate of portfolio impact percentage."""
        spy_prices = self._index_prices.get("SPY", [])
        if len(spy_prices) < 2:
            return 0.0
        prices = list(spy_prices)
        current = prices[-1]
        recent_high = max(prices[-min(30, len(prices)):])
        if recent_high <= 0:
            return 0.0
        return float((recent_high - current) / recent_high * 100)

    def get_diagnostics(self) -> Dict:
        """Return guard state for monitoring."""
        spy_count = len(self._index_prices.get("SPY", []))
        return {
            "threat_level": self._threat_level.name,
            "active_triggers": self._active_triggers,
            "entry_blocked": self.should_block_entry(),
            "size_multiplier": self.get_position_size_multiplier(),
            "spy_price_count": spy_count,
            "vix_current": self._vix_current,
            "vix_open": self._vix_open,
            "threat_expiry": (
                self._threat_expiry.isoformat() if self._threat_expiry else None
            ),
        }


# --- PHASE 3: Incident Response Dispatcher ---
import os
import httpx
import logging

dispatcher_logger = logging.getLogger("AlertDispatcher")

class AlertDispatcher:
    @staticmethod
    async def trigger_pagerduty(reason: str) -> None:
        """Dispatches a critical alert to PagerDuty/Slack during Black Swan events."""
        webhook_url = os.getenv("PAGERDUTY_WEBHOOK_URL", "")
        if not webhook_url:
            dispatcher_logger.warning(f"🚨 BLACK SWAN EVENT: {reason}. (Webhook URL not configured).")
            return
        
        payload = {
            "routing_key": webhook_url,
            "event_action": "trigger",
            "payload": {
                "summary": f"🚨 APEX BLACK SWAN ALERT: {reason}",
                "severity": "critical",
                "source": "apex_trading_system_prod"
            }
        }
        try:
            async with httpx.AsyncClient() as client:
                await client.post("https://events.pagerduty.com/v2/enqueue", json=payload)
                dispatcher_logger.info("📡 PagerDuty alert successfully dispatched.")
        except Exception as e:
            dispatcher_logger.error(f"Failed to dispatch PagerDuty alert: {e}")
