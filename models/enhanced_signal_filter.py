"""
models/enhanced_signal_filter.py - Signal Quality Enhancement Layer

Filters and enhances raw signals to improve quality:
- Volume confirmation
- Multi-model agreement
- Multi-timeframe confirmation
- VIX regime filtering
- Minimum expected return filter
- Cross-sectional momentum integration

This wraps existing signal generators to produce higher-quality, filtered signals.
"""

import numpy as np
import pandas as pd
from typing import Dict, Optional, Tuple, List
from datetime import datetime
import logging

from core.symbols import parse_symbol, AssetClass
from config import ApexConfig

logger = logging.getLogger(__name__)


class EnhancedSignalFilter:
    """
    Signal quality enhancement layer.

    Takes raw signals from ML models and applies rigorous filtering
    to ensure only high-quality signals get executed.
    """

    def __init__(
        self,
        min_model_agreement: float = 0.60,
        min_expected_return: float = 0.005,
        volume_threshold_multiple: float = 1.2,
        vix_no_longs_above: float = 30.0,
        vix_no_shorts_below: float = 12.0
    ):
        """
        Initialize signal filter.

        Args:
            min_model_agreement: Minimum fraction of models agreeing on direction
            min_expected_return: Minimum expected return after costs (0.005 = 0.5%)
            volume_threshold_multiple: Volume must be this multiple of average
            vix_no_longs_above: Don't open longs when VIX above this
            vix_no_shorts_below: Don't open shorts when VIX below this
        """
        self.min_model_agreement = min_model_agreement
        self.min_expected_return = min_expected_return
        self.volume_threshold_multiple = volume_threshold_multiple
        self.vix_no_longs_above = vix_no_longs_above
        self.vix_no_shorts_below = vix_no_shorts_below

        logger.info("EnhancedSignalFilter initialized")
        logger.info(f"  Min model agreement: {min_model_agreement:.0%}")
        logger.info(f"  Min expected return: {min_expected_return:.2%}")
        logger.info(f"  Volume threshold: {volume_threshold_multiple:.1f}x average")

    def filter_signal(
        self,
        symbol: str,
        raw_signal: float,
        confidence: float,
        component_signals: Dict[str, float],
        prices: pd.Series,
        volume: Optional[pd.Series] = None,
        vix_level: Optional[float] = None,
        cross_sectional_rank: Optional[float] = None
    ) -> Dict:
        """
        Apply quality filters to a raw signal.

        Args:
            symbol: Stock ticker
            raw_signal: Raw signal from ML model [-1, 1]
            confidence: Original confidence score [0, 1]
            component_signals: Dict of individual model signals
            prices: Price history
            volume: Volume history (optional)
            vix_level: Current VIX level (optional)
            cross_sectional_rank: Cross-sectional momentum rank [0, 1] (optional)

        Returns:
            Dict with filtered signal, confidence, and rejection reasons
        """
        rejection_reasons = []
        adjustments = []

        filtered_signal = raw_signal
        filtered_confidence = confidence

        # === 1. MODEL AGREEMENT (informational, mild adjustment) ===
        agreement_score = self._calculate_model_agreement(component_signals)
        if agreement_score < self.min_model_agreement:
            adjustments.append(f"Model agreement low: {agreement_score:.1%}")
            filtered_confidence *= 0.85  # Mild penalty only
        else:
            adjustments.append(f"Model agreement: {agreement_score:.1%}")
            filtered_confidence = min(1.0, filtered_confidence * 1.05)  # Small boost

        # === 2. VOLUME CONFIRMATION (informational boost/penalty) ===
        if volume is not None and len(volume) >= 20:
            volume_confirmation = self._check_volume_confirmation(
                raw_signal, prices, volume
            )
            if not volume_confirmation['confirmed']:
                adjustments.append(f"Volume below avg: {volume_confirmation['reason']}")
                filtered_confidence *= 0.90  # Very mild penalty
            else:
                adjustments.append("Volume confirmed")
                if volume_confirmation.get('strong', False):
                    filtered_confidence = min(1.0, filtered_confidence * 1.15)

        # === 3. VIX REGIME FILTER (only hard block in extremes) ===
        if vix_level is not None:
            vix_filter = self._apply_vix_filter(raw_signal, vix_level)
            if vix_filter['blocked']:
                rejection_reasons.append(vix_filter['reason'])
                filtered_signal = 0.0  # Only VIX extreme actually blocks
            elif vix_filter['reduce_size']:
                adjustments.append(f"VIX size adj: {vix_filter['size_multiplier']:.0%}")
                filtered_confidence *= vix_filter['size_multiplier']

        # === 4. MULTI-TIMEFRAME (boost if aligned, mild penalty if not) ===
        mtf_confirmation = self._check_multi_timeframe(prices, raw_signal)
        if mtf_confirmation['confirmed']:
            adjustments.append(f"MTF aligned ({mtf_confirmation['score']:.0%})")
            filtered_confidence = min(1.0, filtered_confidence * 1.10)  # Boost for alignment
        else:
            adjustments.append(f"MTF mixed: {mtf_confirmation['reason']}")
            filtered_confidence *= 0.90  # Mild penalty

        # === 5. EXPECTED RETURN (informational only) ===
        expected_return = self._estimate_expected_return(
            prices, raw_signal, confidence
        )
        if abs(expected_return) < self.min_expected_return:
            adjustments.append(f"Low expected return: {expected_return:.2%}")
            # Don't penalize - just informational

        # === 5b. HARD NET-EDGE FILTER (block if costs exceed edge) ===
        est_cost_bps = self._estimate_cost_bps(symbol)
        est_cost_pct = est_cost_bps / 10000.0
        if abs(expected_return) < est_cost_pct:
            rejection_reasons.append(
                f"Net edge below costs: edge={expected_return:.2%}, cost≈{est_cost_pct:.2%}"
            )
            filtered_signal = 0.0

        # === 6. CROSS-SECTIONAL MOMENTUM (boost/penalty) ===
        if cross_sectional_rank is not None:
            cs_filter = self._apply_cross_sectional_filter(raw_signal, cross_sectional_rank)
            if cs_filter['boost']:
                adjustments.append(f"CS momentum boost: top {(1-cross_sectional_rank)*100:.0f}%")
                filtered_confidence = min(1.0, filtered_confidence * 1.15)
            elif cs_filter['penalty']:
                adjustments.append("CS mid-quintile")
                filtered_confidence *= 0.90

        # === 7. TREND ALIGNMENT (boost if aligned, mild penalty if not) ===
        trend_aligned = self._check_trend_alignment(prices, raw_signal)
        if trend_aligned['aligned']:
            adjustments.append("Trend aligned")
            filtered_confidence = min(1.0, filtered_confidence * 1.10)
        else:
            adjustments.append(f"Counter-trend: {trend_aligned['reason']}")
            filtered_confidence *= 0.85

        # === 8. ANTI-CHASING CHECK (only real hard filter besides VIX) ===
        momentum_check = self._check_momentum_quality(prices, raw_signal)
        if not momentum_check['valid']:
            rejection_reasons.append(momentum_check['reason'])
            filtered_confidence *= 0.7

        # Final decision: only block on hard rejections (VIX extreme, chasing)
        passed = filtered_confidence >= 0.20 and filtered_signal != 0.0

        # Only block if ALL hard filters reject (VIX + chasing)
        if len(rejection_reasons) >= 2 and filtered_signal == 0.0:
            passed = False

        return {
            'symbol': symbol,
            'original_signal': raw_signal,
            'filtered_signal': filtered_signal,
            'original_confidence': confidence,
            'filtered_confidence': float(np.clip(filtered_confidence, 0, 1)),
            'passed': passed,
            'rejection_reasons': rejection_reasons,
            'adjustments': adjustments,
            'model_agreement': agreement_score,
            'expected_return': expected_return,
            'timestamp': datetime.now().isoformat()
        }

    def _estimate_cost_bps(self, symbol: str) -> float:
        """Estimate round-trip costs in basis points."""
        try:
            parsed = parse_symbol(symbol)
        except ValueError:
            parsed = None

        if parsed is None or parsed.asset_class == AssetClass.EQUITY:
            commission_bps = 2 * (ApexConfig.COMMISSION_PER_TRADE / max(ApexConfig.POSITION_SIZE_USD, 1)) * 10000
            return ApexConfig.SLIPPAGE_BPS * 2 + commission_bps
        if parsed.asset_class == AssetClass.FOREX:
            return 2 * (ApexConfig.FX_SPREAD_BPS + ApexConfig.FX_COMMISSION_BPS)
        return 2 * (ApexConfig.CRYPTO_SPREAD_BPS + ApexConfig.CRYPTO_COMMISSION_BPS)

    def _calculate_model_agreement(self, component_signals: Dict[str, float]) -> float:
        """Calculate what fraction of models agree on signal direction."""
        if not component_signals:
            return 0.0

        # Filter out zero/neutral signals
        signals = [s for s in component_signals.values() if abs(s) > 0.1]

        if len(signals) < 2:
            return 0.5  # Not enough signals to measure agreement

        # Count direction agreement
        positive = sum(1 for s in signals if s > 0)
        negative = len(signals) - positive

        # Agreement is the dominant direction's fraction
        agreement = max(positive, negative) / len(signals)

        return agreement

    def _check_volume_confirmation(
        self,
        signal: float,
        prices: pd.Series,
        volume: pd.Series
    ) -> Dict:
        """
        Check if volume confirms the price signal.

        Good signals should have:
        - Above-average volume on signal bars
        - Rising volume in direction of signal
        """
        if len(volume) < 20 or len(prices) < 20:
            return {'confirmed': True, 'reason': 'Insufficient data', 'strong': False}

        avg_volume_20 = volume.iloc[-20:].mean()
        recent_volume = volume.iloc[-5:].mean()
        current_volume = volume.iloc[-1]

        # Basic check: current volume above average
        volume_ratio = current_volume / avg_volume_20 if avg_volume_20 > 0 else 1.0

        if volume_ratio < self.volume_threshold_multiple:
            return {
                'confirmed': False,
                'reason': f"Volume {volume_ratio:.1f}x < {self.volume_threshold_multiple:.1f}x threshold",
                'strong': False
            }

        # Check volume trend matches signal direction
        volume_trend = (recent_volume - avg_volume_20) / avg_volume_20 if avg_volume_20 > 0 else 0

        # Strong signal: high volume (>1.5x) with matching trend
        strong = volume_ratio > 1.5

        return {
            'confirmed': True,
            'reason': f"Volume {volume_ratio:.1f}x average",
            'strong': strong
        }

    def _apply_vix_filter(self, signal: float, vix_level: float) -> Dict:
        """Apply VIX-based signal filtering."""
        result = {
            'blocked': False,
            'reduce_size': False,
            'size_multiplier': 1.0,
            'reason': ''
        }

        # Block new longs in extreme fear
        if signal > 0 and vix_level > self.vix_no_longs_above:
            result['blocked'] = True
            result['reason'] = f"VIX {vix_level:.1f} > {self.vix_no_longs_above} - no new longs"
            return result

        # Block new shorts in extreme complacency
        if signal < 0 and vix_level < self.vix_no_shorts_below:
            result['blocked'] = True
            result['reason'] = f"VIX {vix_level:.1f} < {self.vix_no_shorts_below} - no new shorts"
            return result

        # Reduce size in elevated VIX
        if vix_level > 20:
            result['reduce_size'] = True
            # Linear reduction: VIX 20 = 100%, VIX 30 = 50%
            result['size_multiplier'] = max(0.5, 1.0 - (vix_level - 20) / 20)

        return result

    def _check_multi_timeframe(self, prices: pd.Series, signal: float) -> Dict:
        """Check if signal aligns across multiple timeframes."""
        if len(prices) < 60:
            return {'confirmed': True, 'score': 0.5, 'reason': 'Insufficient data'}

        # Calculate momentum at different timeframes
        mom_5d = (prices.iloc[-1] / prices.iloc[-5] - 1) if len(prices) >= 5 else 0
        mom_10d = (prices.iloc[-1] / prices.iloc[-10] - 1) if len(prices) >= 10 else 0
        mom_20d = (prices.iloc[-1] / prices.iloc[-20] - 1) if len(prices) >= 20 else 0
        mom_60d = (prices.iloc[-1] / prices.iloc[-60] - 1) if len(prices) >= 60 else 0

        momentums = [mom_5d, mom_10d, mom_20d, mom_60d]

        # Check agreement with signal direction
        signal_direction = np.sign(signal)
        agreements = sum(1 for m in momentums if np.sign(m) == signal_direction)
        score = agreements / len(momentums)

        confirmed = score >= 0.5  # At least half must agree

        reason = f"{agreements}/{len(momentums)} timeframes agree"

        return {'confirmed': confirmed, 'score': score, 'reason': reason}

    def _estimate_expected_return(
        self,
        prices: pd.Series,
        signal: float,
        confidence: float
    ) -> float:
        """Estimate expected return based on signal strength and historical volatility."""
        if len(prices) < 20:
            return 0.0

        # Historical volatility (annualized)
        returns = prices.pct_change().dropna()
        daily_vol = returns.iloc[-20:].std()

        # Expected return = signal strength * volatility * confidence
        # This is a rough approximation of expected 5-day return
        expected_daily_return = abs(signal) * daily_vol * confidence * 0.5
        expected_5d_return = expected_daily_return * np.sqrt(5)

        # Apply sign based on signal direction
        return expected_5d_return * np.sign(signal)

    def _apply_cross_sectional_filter(
        self,
        signal: float,
        rank: float  # 0 = best momentum, 1 = worst
    ) -> Dict:
        """Apply cross-sectional momentum filter."""
        result = {'boost': False, 'penalty': False}

        if signal > 0:  # Long signal
            # Boost if in top 20% momentum
            if rank <= 0.20:
                result['boost'] = True
            # Penalty if in middle 40-60%
            elif 0.40 <= rank <= 0.60:
                result['penalty'] = True
        elif signal < 0:  # Short signal
            # Boost if in bottom 20% momentum
            if rank >= 0.80:
                result['boost'] = True
            # Penalty if in middle 40-60%
            elif 0.40 <= rank <= 0.60:
                result['penalty'] = True

        return result

    def _check_trend_alignment(self, prices: pd.Series, signal: float) -> Dict:
        """Check if signal aligns with the dominant trend."""
        if len(prices) < 50:
            return {'aligned': True, 'reason': 'Insufficient data'}

        # Calculate trend using MA crossover
        ma_20 = prices.iloc[-20:].mean()
        ma_50 = prices.iloc[-50:].mean()

        trend_bullish = ma_20 > ma_50
        signal_bullish = signal > 0

        aligned = (trend_bullish == signal_bullish) or abs(signal) < 0.3

        if not aligned:
            if signal_bullish:
                reason = "Long signal against bearish trend (MA20 < MA50)"
            else:
                reason = "Short signal against bullish trend (MA20 > MA50)"
        else:
            reason = "Signal aligned with trend"

        return {'aligned': aligned, 'reason': reason}

    def _check_momentum_quality(self, prices: pd.Series, signal: float) -> Dict:
        """Check momentum quality - avoid exhausted moves."""
        if len(prices) < 20:
            return {'valid': True, 'reason': ''}

        # Check for exhaustion: price far from recent range
        high_20 = prices.iloc[-20:].max()
        low_20 = prices.iloc[-20:].min()
        range_20 = high_20 - low_20

        if range_20 == 0:
            return {'valid': True, 'reason': ''}

        current = prices.iloc[-1]
        position = (current - low_20) / range_20

        # Avoid chasing: don't go long at top of range or short at bottom
        if signal > 0.3 and position > 0.90:
            return {
                'valid': False,
                'reason': f"Chasing: long signal at {position:.0%} of 20d range"
            }
        if signal < -0.3 and position < 0.10:
            return {
                'valid': False,
                'reason': f"Chasing: short signal at {position:.0%} of 20d range"
            }

        return {'valid': True, 'reason': ''}


def create_enhanced_filter(config=None) -> EnhancedSignalFilter:
    """Factory function to create signal filter from config."""
    if config is None:
        from config import ApexConfig
        config = ApexConfig

    return EnhancedSignalFilter(
        min_model_agreement=getattr(config, 'MIN_MODEL_AGREEMENT', 0.60),
        min_expected_return=getattr(config, 'MIN_EXPECTED_RETURN', 0.005),
        volume_threshold_multiple=getattr(config, 'VOLUME_THRESHOLD_MULTIPLE', 1.2),
        vix_no_longs_above=getattr(config, 'VIX_NO_LONGS_ABOVE', 30.0),
        vix_no_shorts_below=getattr(config, 'VIX_NO_SHORTS_BELOW', 12.0)
    )
