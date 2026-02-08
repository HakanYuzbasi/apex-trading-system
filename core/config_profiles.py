"""
core/config_profiles.py - Configuration Profiles

Provides predefined configuration profiles for different trading styles:
- Conservative: Lower risk, tighter stops, fewer positions
- Moderate: Balanced risk/reward
- Aggressive: Higher risk tolerance, more positions
- Scalping: High frequency, tight exits
- Swing: Longer holds, wider stops

Also provides dynamic parameter adjustment based on market conditions.
"""

import os
import json
import logging
from dataclasses import dataclass, field, asdict
from typing import Dict, Optional, Any
from pathlib import Path
from enum import Enum

from config import ApexConfig

logger = logging.getLogger(__name__)


class ProfileType(Enum):
    """Available profile types."""
    CONSERVATIVE = "conservative"
    MODERATE = "moderate"
    AGGRESSIVE = "aggressive"
    SCALPING = "scalping"
    SWING = "swing"
    CUSTOM = "custom"


@dataclass
class TradingProfile:
    """Trading configuration profile."""

    name: str
    profile_type: ProfileType

    # Position Sizing
    max_position_pct: float = 0.02       # Max position as % of capital
    max_positions: int = 15              # Maximum concurrent positions
    position_size_usd: float = 20000     # Default position size

    # Risk Management
    max_portfolio_risk_pct: float = 0.02 # Max portfolio risk per day
    max_sector_exposure: float = 0.40    # Max exposure to single sector
    max_correlation: float = 0.70        # Max portfolio correlation

    # Stop Loss / Take Profit
    stop_loss_pct: float = 0.03          # Default stop loss
    take_profit_pct: float = 0.06        # Default take profit
    trailing_stop_activation: float = 0.025  # When to activate trailing
    trailing_stop_distance: float = 0.02     # Trailing stop distance
    max_hold_days: int = 14              # Maximum holding period

    # Signal Thresholds
    min_signal_threshold: float = 0.25   # Minimum signal to enter
    min_confidence: float = 0.25         # Minimum confidence required
    require_mtf_confirmation: bool = False  # Multi-timeframe confirmation

    # Quality Filters
    min_model_agreement: float = 0.50    # Minimum model agreement
    min_expected_return: float = 0.003   # Minimum expected return
    volume_threshold_multiple: float = 0.6  # Volume filter

    # VIX Thresholds
    vix_reduce_exposure: float = 25      # Start reducing at this VIX
    vix_halt_new_entries: float = 35     # Stop new entries at this VIX

    # Regime Adjustments
    regime_thresholds: Dict[str, float] = field(default_factory=dict)

    def __post_init__(self):
        if not self.regime_thresholds:
            self.regime_thresholds = {
                'strong_bull': self.min_signal_threshold * 0.8,
                'bull': self.min_signal_threshold * 0.9,
                'neutral': self.min_signal_threshold,
                'bear': self.min_signal_threshold * 1.1,
                'strong_bear': self.min_signal_threshold * 1.2,
                'high_volatility': self.min_signal_threshold * 1.3
            }

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        d = asdict(self)
        d['profile_type'] = self.profile_type.value
        return d

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'TradingProfile':
        """Create from dictionary."""
        data['profile_type'] = ProfileType(data.get('profile_type', 'custom'))
        return cls(**data)


# Predefined Profiles
PROFILES: Dict[ProfileType, TradingProfile] = {
    ProfileType.CONSERVATIVE: TradingProfile(
        name="Conservative",
        profile_type=ProfileType.CONSERVATIVE,
        max_position_pct=0.015,
        max_positions=10,
        position_size_usd=15000,
        max_portfolio_risk_pct=0.015,
        max_sector_exposure=0.30,
        max_correlation=0.60,
        stop_loss_pct=0.02,
        take_profit_pct=0.04,
        trailing_stop_activation=0.02,
        trailing_stop_distance=0.015,
        max_hold_days=10,
        min_signal_threshold=0.35,
        min_confidence=0.35,
        require_mtf_confirmation=True,
        min_model_agreement=0.60,
        min_expected_return=0.005,
        volume_threshold_multiple=0.8,
        vix_reduce_exposure=22,
        vix_halt_new_entries=30
    ),

    ProfileType.MODERATE: TradingProfile(
        name="Moderate",
        profile_type=ProfileType.MODERATE,
        max_position_pct=0.02,
        max_positions=15,
        position_size_usd=20000,
        max_portfolio_risk_pct=0.02,
        max_sector_exposure=0.35,
        max_correlation=0.65,
        stop_loss_pct=0.03,
        take_profit_pct=0.06,
        trailing_stop_activation=0.025,
        trailing_stop_distance=0.02,
        max_hold_days=14,
        min_signal_threshold=0.28,
        min_confidence=0.28,
        require_mtf_confirmation=False,
        min_model_agreement=0.55,
        min_expected_return=0.004,
        volume_threshold_multiple=0.7,
        vix_reduce_exposure=25,
        vix_halt_new_entries=35
    ),

    ProfileType.AGGRESSIVE: TradingProfile(
        name="Aggressive",
        profile_type=ProfileType.AGGRESSIVE,
        max_position_pct=0.03,
        max_positions=20,
        position_size_usd=30000,
        max_portfolio_risk_pct=0.03,
        max_sector_exposure=0.45,
        max_correlation=0.75,
        stop_loss_pct=0.04,
        take_profit_pct=0.08,
        trailing_stop_activation=0.03,
        trailing_stop_distance=0.025,
        max_hold_days=21,
        min_signal_threshold=0.20,
        min_confidence=0.20,
        require_mtf_confirmation=False,
        min_model_agreement=0.45,
        min_expected_return=0.003,
        volume_threshold_multiple=0.5,
        vix_reduce_exposure=30,
        vix_halt_new_entries=40
    ),

    ProfileType.SCALPING: TradingProfile(
        name="Scalping",
        profile_type=ProfileType.SCALPING,
        max_position_pct=0.01,
        max_positions=8,
        position_size_usd=10000,
        max_portfolio_risk_pct=0.01,
        max_sector_exposure=0.25,
        max_correlation=0.50,
        stop_loss_pct=0.01,
        take_profit_pct=0.015,
        trailing_stop_activation=0.008,
        trailing_stop_distance=0.005,
        max_hold_days=2,
        min_signal_threshold=0.40,
        min_confidence=0.40,
        require_mtf_confirmation=False,
        min_model_agreement=0.65,
        min_expected_return=0.002,
        volume_threshold_multiple=1.0,
        vix_reduce_exposure=20,
        vix_halt_new_entries=28
    ),

    ProfileType.SWING: TradingProfile(
        name="Swing Trading",
        profile_type=ProfileType.SWING,
        max_position_pct=0.025,
        max_positions=12,
        position_size_usd=25000,
        max_portfolio_risk_pct=0.025,
        max_sector_exposure=0.40,
        max_correlation=0.70,
        stop_loss_pct=0.05,
        take_profit_pct=0.12,
        trailing_stop_activation=0.04,
        trailing_stop_distance=0.03,
        max_hold_days=30,
        min_signal_threshold=0.30,
        min_confidence=0.30,
        require_mtf_confirmation=True,
        min_model_agreement=0.55,
        min_expected_return=0.006,
        volume_threshold_multiple=0.6,
        vix_reduce_exposure=28,
        vix_halt_new_entries=38
    )
}


class DynamicConfigAdjuster:
    """
    Dynamically adjusts configuration based on market conditions.

    Adjustments include:
    - Reducing position sizes when VIX is elevated
    - Tightening stops in high volatility
    - Reducing exposure during drawdowns
    """

    def __init__(self, base_profile: TradingProfile):
        """
        Initialize with a base profile.

        Args:
            base_profile: Starting configuration profile
        """
        self.base_profile = base_profile
        self.current_profile = TradingProfile(**asdict(base_profile))

        # Market state
        self.current_vix = 15.0
        self.current_drawdown = 0.0
        self.daily_pnl_pct = 0.0
        self.recent_win_rate = 0.5

        logger.info(f"DynamicConfigAdjuster initialized with {base_profile.name} profile")

    def update_market_state(
        self,
        vix: Optional[float] = None,
        drawdown: Optional[float] = None,
        daily_pnl_pct: Optional[float] = None,
        win_rate: Optional[float] = None
    ):
        """Update market state for dynamic adjustments."""
        if vix is not None:
            self.current_vix = vix
        if drawdown is not None:
            self.current_drawdown = abs(drawdown)
        if daily_pnl_pct is not None:
            self.daily_pnl_pct = daily_pnl_pct
        if win_rate is not None:
            self.recent_win_rate = win_rate

        self._recalculate_profile()

    def _recalculate_profile(self):
        """Recalculate profile based on current conditions."""
        # Start fresh from base
        self.current_profile = TradingProfile(**asdict(self.base_profile))

        # VIX adjustments
        vix_factor = self._calculate_vix_factor()
        if vix_factor < 1.0:
            self.current_profile.position_size_usd *= vix_factor
            self.current_profile.max_positions = int(
                self.current_profile.max_positions * vix_factor
            )
            self.current_profile.stop_loss_pct *= 0.8  # Tighter stops

        # Drawdown adjustments
        dd_factor = self._calculate_drawdown_factor()
        if dd_factor < 1.0:
            self.current_profile.position_size_usd *= dd_factor
            self.current_profile.min_signal_threshold *= 1.2  # Raise bar

        # Performance adjustments
        perf_factor = self._calculate_performance_factor()
        if perf_factor < 1.0:
            self.current_profile.min_confidence *= 1.1
            self.current_profile.min_model_agreement *= 1.1

        logger.debug(
            f"Profile adjusted: VIX={vix_factor:.2f}, DD={dd_factor:.2f}, "
            f"Perf={perf_factor:.2f}"
        )

    def _calculate_vix_factor(self) -> float:
        """Calculate VIX-based scaling factor."""
        base_vix = self.base_profile.vix_reduce_exposure
        halt_vix = self.base_profile.vix_halt_new_entries

        if self.current_vix <= base_vix:
            return 1.0
        elif self.current_vix >= halt_vix:
            return 0.3  # Minimum 30%
        else:
            # Linear interpolation
            range_vix = halt_vix - base_vix
            excess_vix = self.current_vix - base_vix
            return 1.0 - (excess_vix / range_vix) * 0.7

    def _calculate_drawdown_factor(self) -> float:
        """Calculate drawdown-based scaling factor."""
        if self.current_drawdown <= 0.02:
            return 1.0
        elif self.current_drawdown >= 0.08:
            return 0.3
        else:
            # Reduce as drawdown increases
            return 1.0 - (self.current_drawdown - 0.02) / 0.06 * 0.7

    def _calculate_performance_factor(self) -> float:
        """Calculate performance-based scaling factor."""
        # Reduce exposure if win rate is poor
        if self.recent_win_rate >= 0.5:
            return 1.0
        elif self.recent_win_rate <= 0.35:
            return 0.6
        else:
            return 0.6 + (self.recent_win_rate - 0.35) / 0.15 * 0.4

    def get_current_profile(self) -> TradingProfile:
        """Get currently adjusted profile."""
        return self.current_profile

    def get_adjustment_summary(self) -> Dict[str, Any]:
        """Get summary of current adjustments."""
        return {
            'base_profile': self.base_profile.name,
            'current_vix': self.current_vix,
            'current_drawdown': self.current_drawdown,
            'vix_factor': self._calculate_vix_factor(),
            'drawdown_factor': self._calculate_drawdown_factor(),
            'performance_factor': self._calculate_performance_factor(),
            'adjusted_position_size': self.current_profile.position_size_usd,
            'adjusted_max_positions': self.current_profile.max_positions,
            'adjusted_stop_loss': self.current_profile.stop_loss_pct
        }


class ProfileManager:
    """
    Manages trading profiles and persistence.

    Usage:
        manager = ProfileManager()
        manager.load_profile(ProfileType.MODERATE)

        # Get current settings
        profile = manager.get_current_profile()

        # Update dynamically
        manager.update_for_market_conditions(vix=28, drawdown=0.03)
    """

    def __init__(self, storage_path: Optional[str] = None):
        """
        Initialize profile manager.

        Args:
            storage_path: Path to store custom profiles
        """
        self.storage_path = Path(storage_path) if storage_path else (ApexConfig.DATA_DIR / "profiles")
        self.storage_path.mkdir(parents=True, exist_ok=True)

        self.current_profile: Optional[TradingProfile] = None
        self.adjuster: Optional[DynamicConfigAdjuster] = None

        # Load default profile from env or use moderate
        default_type = os.getenv("APEX_PROFILE", "moderate")
        try:
            profile_type = ProfileType(default_type.lower())
            self.load_profile(profile_type)
        except ValueError:
            self.load_profile(ProfileType.MODERATE)

    def load_profile(self, profile_type: ProfileType):
        """Load a predefined profile."""
        if profile_type in PROFILES:
            self.current_profile = TradingProfile(**asdict(PROFILES[profile_type]))
            self.adjuster = DynamicConfigAdjuster(self.current_profile)
            logger.info(f"Loaded profile: {profile_type.value}")
        else:
            raise ValueError(f"Unknown profile type: {profile_type}")

    def load_custom_profile(self, name: str) -> bool:
        """Load a custom profile from storage."""
        profile_file = self.storage_path / f"{name}.json"

        if not profile_file.exists():
            logger.warning(f"Custom profile not found: {name}")
            return False

        try:
            with open(profile_file) as f:
                data = json.load(f)

            self.current_profile = TradingProfile.from_dict(data)
            self.adjuster = DynamicConfigAdjuster(self.current_profile)
            logger.info(f"Loaded custom profile: {name}")
            return True

        except Exception as e:
            logger.error(f"Error loading custom profile: {e}")
            return False

    def save_custom_profile(self, name: str, profile: TradingProfile):
        """Save a custom profile."""
        profile_file = self.storage_path / f"{name}.json"

        try:
            with open(profile_file, 'w') as f:
                json.dump(profile.to_dict(), f, indent=2)

            logger.info(f"Saved custom profile: {name}")

        except Exception as e:
            logger.error(f"Error saving custom profile: {e}")

    def update_for_market_conditions(
        self,
        vix: Optional[float] = None,
        drawdown: Optional[float] = None,
        daily_pnl_pct: Optional[float] = None,
        win_rate: Optional[float] = None
    ):
        """Update profile for current market conditions."""
        if self.adjuster:
            self.adjuster.update_market_state(
                vix=vix,
                drawdown=drawdown,
                daily_pnl_pct=daily_pnl_pct,
                win_rate=win_rate
            )

    def get_current_profile(self) -> TradingProfile:
        """Get current (adjusted) profile."""
        if self.adjuster:
            return self.adjuster.get_current_profile()
        return self.current_profile

    def get_base_profile(self) -> TradingProfile:
        """Get base (unadjusted) profile."""
        return self.current_profile

    def list_profiles(self) -> Dict[str, str]:
        """List all available profiles."""
        profiles = {p.value: PROFILES[p].name for p in ProfileType if p in PROFILES}

        # Add custom profiles
        for file in self.storage_path.glob("*.json"):
            profiles[file.stem] = f"Custom: {file.stem}"

        return profiles

    def get_adjustment_summary(self) -> Dict[str, Any]:
        """Get summary of current adjustments."""
        if self.adjuster:
            return self.adjuster.get_adjustment_summary()
        return {}


# Global instance
_profile_manager: Optional[ProfileManager] = None


def get_profile_manager() -> ProfileManager:
    """Get global profile manager instance."""
    global _profile_manager
    if _profile_manager is None:
        _profile_manager = ProfileManager()
    return _profile_manager


def get_current_profile() -> TradingProfile:
    """Get current trading profile."""
    return get_profile_manager().get_current_profile()
