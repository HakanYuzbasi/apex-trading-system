from __future__ import annotations

import asyncio
import logging
from collections.abc import Mapping
from datetime import datetime, timezone

from core.symbols import AssetClass, parse_symbol
from quant_system.core.bus import InMemoryEventBus, Subscription
from quant_system.events import OrderEvent, SignalEvent
from quant_system.instruments.instrument import Instrument
from quant_system.portfolio.ledger import PortfolioLedger
from quant_system.risk.factors import FactorMonitor
from quant_system.risk.protector import EquityProtector
from quant_system.risk.regime_detector import RegimeDetector
from quant_system.risk.bayesian_vol import BayesianVolatilityAdjuster
from core.logic.ml.meta_labeler import MetaLabeler
from quant_system.risk.sentiment_warden import SentimentWarden
from quant_system.risk.social_pulse import SocialPulse
from quant_system.portfolio.hrp import HRPOptimizer

logger = logging.getLogger(__name__)


class RiskManager:
    DEFAULT_MARGIN_BY_ASSET_CLASS: Mapping[AssetClass, float] = {
        AssetClass.CRYPTO: 1.0,
        AssetClass.EQUITY: 0.50,
        AssetClass.FOREX: 0.05,
        AssetClass.OPTION: 1.0,
    }

    def __init__(
        self,
        portfolio_ledger: PortfolioLedger,
        event_bus: InMemoryEventBus,
        *,
        instruments: Mapping[str, Instrument] | None = None,
        margin_by_asset_class: Mapping[AssetClass, float] | None = None,
        regime_detector: RegimeDetector | None = None,
        factor_monitor: FactorMonitor | None = None,
        equity_protector: EquityProtector | None = None,
        meta_labeler: MetaLabeler | None = None,
        bayesian_vol: BayesianVolatilityAdjuster | None = None,
        sentiment_warden: SentimentWarden | None = None,
        social_pulse: SocialPulse | None = None,
        hrp_optimizer: HRPOptimizer | None = None,
    ) -> None:
        self.portfolio_ledger = portfolio_ledger
        self.event_bus = event_bus
        self.instruments = dict(instruments or {})
        self.margin_by_asset_class = dict(self.DEFAULT_MARGIN_BY_ASSET_CLASS)
        self.regime_detector = regime_detector
        self.factor_monitor = factor_monitor
        self.equity_protector = equity_protector
        self.meta_labeler = meta_labeler
        self.bayesian_vol = bayesian_vol
        self.sentiment_warden = sentiment_warden
        self.social_pulse = social_pulse
        self.hrp_optimizer = hrp_optimizer
        
        self.latest_meta_confidence: float = 1.0
        self.latest_bayesian_prob: float = 0.0
        self.global_leverage_limit: float = 1.0  # 1.0 = 100% of target notional
        
        self._peak_equity: float = 0.0
        self._dual_drawdown_halt: bool = False
        self._drawdown_halt_threshold: float = 0.05 # 5% drawdown triggers halt if margin is high
        
        if margin_by_asset_class is not None:
            self.margin_by_asset_class.update(margin_by_asset_class)
        for asset_class, margin_rate in self.margin_by_asset_class.items():
            if not 0.0 < float(margin_rate) <= 1.0:
                raise ValueError(f"margin rate for {asset_class.value} must be in (0.0, 1.0]")
        self._subscription: Subscription = self.event_bus.subscribe("signal", self._on_signal)

    @property
    def subscription(self) -> Subscription:
        return self._subscription

    def close(self) -> None:
        self.event_bus.unsubscribe(self._subscription.token)

    def _on_signal(self, event: SignalEvent) -> None:
        self._check_circuit_breaker()
        
        if self._dual_drawdown_halt:
            if event.side not in {"flatten", "cover"}:
                logger.warning("RiskManager VETO: Dual-Drawdown Halt is active. Rejecting %s signal for %s", event.side, event.instrument_id)
                return
                
        order = self._translate_signal_to_order(event)
        if order is None:
            return
        self._dispatch_order(order)

    def _check_circuit_breaker(self) -> None:
        current_equity = self.portfolio_ledger.total_equity()
        if current_equity > self._peak_equity:
            self._peak_equity = current_equity
            
        if self._peak_equity <= 0:
            return
            
        drawdown = (self._peak_equity - current_equity) / self._peak_equity
        margin_used = self.current_required_margin()
        margin_utilization = margin_used / current_equity if current_equity > 0 else 0
        
        # If we are in extreme drawdown and actively consuming margin (> 40% margin utilized)
        # Flip the breaker.
        if drawdown >= self._drawdown_halt_threshold and margin_utilization > 0.40 and not self._dual_drawdown_halt:
            logger.critical("CIRCUIT BREAKER: Dual-Drawdown Halt triggered! DD=%.2f%%, MarginUtil=%.2f%%", drawdown * 100, margin_utilization * 100)
            self._dual_drawdown_halt = True
            self._emit_flatten_all()
            
    def _emit_flatten_all(self) -> None:
        """Emits flatten signals for all active positions to aggressively de-risk."""
        now = datetime.now(timezone.utc)
        for instrument_id, position in self.portfolio_ledger.positions.items():
            if abs(position.quantity) > 0:
                flatten_order = OrderEvent(
                    instrument_id=instrument_id,
                    exchange_ts=now,
                    received_ts=now,
                    processed_ts=now,
                    sequence_id=0,
                    source="circuit_breaker",
                    strategy_id="risk_manager_breaker",
                    order_action="submit",
                    order_scope="parent",
                    side="sell" if position.quantity > 0 else "buy",
                    order_type="market",
                    quantity=abs(position.quantity),
                    time_in_force="day",
                    execution_algo="direct",
                )
                self._dispatch_order(flatten_order)

    def _translate_signal_to_order(self, event: SignalEvent) -> OrderEvent | None:
        position = self.portfolio_ledger.get_position(event.instrument_id)
        reference_price = self.portfolio_ledger.get_reference_price(event.instrument_id) or 1.0
        instrument = self._resolve_instrument(event.instrument_id)
        if event.side == "flatten":
            adjusted_target_value = event.target_value
            if position.quantity == 0:
                return None
            quantity = abs(position.quantity)
            order_side = "sell" if position.quantity > 0 else "buy"
        else:
            # Apply leverage limit to the signal target value for non-flatten entries
            adjusted_target_value = event.target_value * self.global_leverage_limit
            quantity = self._quantity_from_signal_v2(event, adjusted_target_value, reference_price, self.portfolio_ledger.cash)
            if quantity <= 0:
                logger.warning("RiskManager rejected signal with non-positive quantity: %s", event.signal_id)
                return None

            if event.side == "buy":
                order_side = "buy"
            elif event.side == "sell":
                order_side = "sell"
            else:
                logger.warning("RiskManager rejected unsupported signal side '%s' for %s", event.side, event.instrument_id)
                return None

        projected_quantity = position.quantity + (quantity if order_side == "buy" else -quantity)
        if projected_quantity < -1e-12 and not instrument.shortable:
            logger.warning(
                "RiskManager rejected short signal for %s: instrument is not shortable",
                event.instrument_id,
            )
            return None
        # Alpaca does not support crypto short-selling (long-only).
        # projected_quantity == 0 means sell-to-close an existing long → allowed.
        # projected_quantity < 0 means a short-open → not supported.
        if projected_quantity < -1e-12 and instrument.asset_class.value == "CRYPTO":
            logger.info(
                "Skipping crypto short-open for %s (Alpaca crypto is long-only; "
                "pairs strategy will run long-leg only for this signal).",
                event.instrument_id,
            )
            return None

        if (
            event.side != "flatten"
            and self.regime_detector is not None
            and self.regime_detector.is_extreme_volatility(event.instrument_id)
        ):
            logger.warning(
                "RiskManager vetoed entry signal for %s due to extreme volatility regime=%s",
                event.instrument_id,
                self.regime_detector.regime_for(event.instrument_id),
            )
            return None
            
        # ── Sentiment Warden Veto ───────────────────────────────────────────
        if (
            event.side not in {"flatten", "rebalance", "cover"}
            and self.sentiment_warden is not None
            and self.sentiment_warden.is_vetoed(event.instrument_id)
        ):
            details = self.sentiment_warden.get_veto_details(event.instrument_id)
            logger.warning(
                "RiskManager vetoed %s entry for %s: Structural news detected. Headline: %s",
                event.side,
                event.instrument_id,
                details.get("headline") if details else "Unknown"
            )
            return None

        # ── Social Pulse Veto ───────────────────────────────────────────────
        if (
            event.side not in {"flatten", "rebalance", "cover"}
            and self.social_pulse is not None
        ):
            # By default assume trades are mean reversion unless tagged as breakout
            is_mean_reversion = event.metadata.get("strategy_type", "mean_reversion") != "breakout"
            
            if is_mean_reversion:
                buzz_zscore = self.social_pulse.get_retail_buzz_zscore(event.instrument_id)
                if buzz_zscore > 3.0:
                    logger.warning(
                        "RiskManager vetoed %s entry for %s: Social Retail Pulse > 3 Sigma (%.2f). Don't bet against the crowd.",
                        event.side,
                        event.instrument_id,
                        buzz_zscore
                    )
                    return None

        # ── Meta-Labeler Veto ───────────────────────────────────────────────
        if (
            event.side not in {"flatten", "rebalance", "cover"}
            and self.meta_labeler is not None
        ):
            kalman_residual = event.metadata.get("kalman_residual", 0.0) if event.metadata else 0.0
            vix_level = event.metadata.get("vix_level", 18.5) if event.metadata else 18.5
            
            bayesian_prob = 0.5
            if self.bayesian_vol is not None:
                bayesian_prob = self.bayesian_vol.probability_of_high_vol(event.instrument_id)
                
            sector_concentration = 0.0
            if self.factor_monitor is not None:
                sector = self.factor_monitor.sector_for(event.instrument_id)
                equity = self.portfolio_ledger.total_equity()
                if equity > 0:
                    sector_concentration = self.factor_monitor.sector_gross_notional(sector) / equity

            self.latest_bayesian_prob = bayesian_prob
            
            logger.info("[RISK] Processing signal for %s (Residual: %.4f, Vol Prob: %.4f)", event.instrument_id, kalman_residual, bayesian_prob)
            
            if hasattr(self.meta_labeler, "predict_confidence"):
                confidence = self.meta_labeler.predict_confidence(
                    kalman_residual=kalman_residual,
                    bayesian_prob=bayesian_prob,
                    vix_level=vix_level,
                    sector_concentration=sector_concentration
                )
            else:
                logger.warning("[RISK] MetaLabeler missing predict_confidence method! Using pass-through confidence=1.0")
                confidence = 1.0
                
            self.latest_meta_confidence = confidence
            logger.info("[RISK] MetaLabeler confidence for %s: %.4f", event.instrument_id, confidence)
            
            # ── HRP Regime Scaling ───────────────────────────────────────────
            hrp_scalar = 1.0
            if self.hrp_optimizer is not None:
                # We use the current regime from HRP (1: Normal, 2: High Vol, 3: Crash)
                # to scale the signal value.
                regime = getattr(self.hrp_optimizer, "current_regime", 1)
                if regime == 2: hrp_scalar = 0.7  # Scale down in High Vol
                elif regime == 3: hrp_scalar = 0.3 # Scale down heavily in Crash
                
                if hrp_scalar < 1.0:
                    logger.info("[RISK] HRP Regime Scaling active: regime=%d, scalar=%.2f", regime, hrp_scalar)
                    adjusted_target_value *= hrp_scalar
                    # Recalculate quantity based on HRP-scaled target
                    quantity = self._quantity_from_signal_v2(event, adjusted_target_value, reference_price, self.portfolio_ledger.cash)

            if confidence < 0.65:
                logger.warning(
                    "RiskManager vetoed entry for %s: Meta-Labeler confidence %.2f < 0.65",
                    event.instrument_id,
                    confidence
                )
                return None

        # ── EquityProtector halt veto (equity-scoped) ───────────────────────
        # When the daily loss limit has been breached the bot is halted for
        # the rest of the UTC day.  Only risk-reducing signals (flatten,
        # cover, rebalance) are allowed through so the protector's own
        # flatten signals can reach the broker.
        #
        # SCOPE: equity and options only.  Crypto and forex run on
        # independent capital envelopes — a US-equity drawdown event must
        # not freeze overnight crypto P&L.  A separate CryptoProtector
        # threshold can be added when needed.
        if (
            event.side not in {"flatten", "rebalance", "cover"}
            and self.equity_protector is not None
            and self.equity_protector.is_halted()
        ):
            _parsed = parse_symbol(event.instrument_id)
            if _parsed.asset_class in {AssetClass.EQUITY, AssetClass.OPTION}:
                logger.warning(
                    "RiskManager vetoed %s entry for %s: EquityProtector halt active (equity-only scope)",
                    event.side,
                    event.instrument_id,
                )
                return None
            # Crypto / Forex: pass through — independent capital, no US-equity halt applies.
            logger.debug(
                "EquityProtector halted but %s is %s — passing through (equity-only scope)",
                event.instrument_id,
                _parsed.asset_class.value,
            )

        # ── FactorMonitor sector concentration veto ──────────────────────────
        if (
            self.factor_monitor is not None
            and not self.factor_monitor.check_signal(event, reference_price)
        ):
            return None

        if not self._passes_margin_check(
            instrument=instrument,
            instrument_id=event.instrument_id,
            order_side=order_side,
            quantity=quantity,
            reference_price=reference_price,
        ):
            logger.warning(
                "RiskManager rejected %s signal for %s: insufficient margin for quantity=%.4f at reference_price=%.4f",
                order_side,
                event.instrument_id,
                quantity,
                reference_price,
            )
            return None

        now = datetime.now(timezone.utc)
        
        # ── Dynamic Limit Pricing ──────────────────────────────────────────
        # Emergency or circuit breaker orders use market for guaranteed execution
        if event.source in {"circuit_breaker", "protector"} or event.side == "flatten":
            order_type = "market"
            limit_price = None
        else:
            order_type = "limit"
            limit_price = self._calculate_limit_price(
                reference_price=reference_price,
                side=order_side,
                confidence=self.latest_meta_confidence
            )

        return OrderEvent(
            instrument_id=event.instrument_id,
            exchange_ts=now,
            received_ts=now,
            processed_ts=now,
            sequence_id=event.sequence_id,
            source="risk_manager",
            strategy_id=event.strategy_id,
            order_action="submit",
            order_scope="parent",
            side=order_side,
            order_type=order_type,
            quantity=quantity,
            time_in_force="day",
            execution_algo="direct",
            limit_price=limit_price,
            notional=abs(adjusted_target_value) if event.target_type == "notional" else None,
        )

    def _calculate_limit_price(self, reference_price: float, side: str, confidence: float) -> float:
        """
        Calculates a limit price with an offset based on trade confidence.
        High confidence trades are more aggressive (closer to the market) to ensure fill.
        """
        # Base offset: 5 basis points (0.0005)
        base_offset = 0.0005
        
        # Aggression adjustment based on Meta-Labeler confidence
        if confidence > 0.90:
            aggression = 0.0005 # More aggressive (add 5bps more towards the market)
        elif confidence > 0.80:
            aggression = 0.0002
        elif confidence < 0.70:
            aggression = -0.0002 # Passive (bid lower/ask higher)
        else:
            aggression = 0.0
            
        total_offset = base_offset + aggression
        
        if side == "buy":
            return reference_price * (1 + total_offset)
        else:
            return reference_price * (1 - total_offset)

    @staticmethod
    def _quantity_from_signal_v2(event: SignalEvent, target_value: float, reference_price: float, available_cash: float) -> float:
        if event.target_type == "units":
            return abs(float(target_value))
        if event.target_type == "notional":
            if reference_price <= 0:
                return 0.0
            return abs(float(target_value)) / reference_price
        if event.target_type == "weight":
            if reference_price <= 0:
                return 0.0
            notional = abs(float(target_value)) * available_cash
            return notional / reference_price
        return abs(float(target_value))

    @staticmethod
    def _quantity_from_signal(event: SignalEvent, reference_price: float, available_cash: float) -> float:
        return RiskManager._quantity_from_signal_v2(event, event.target_value, reference_price, available_cash)
        return abs(float(event.target_value))

    def _passes_margin_check(
        self,
        *,
        instrument: Instrument,
        instrument_id: str,
        order_side: str,
        quantity: float,
        reference_price: float,
    ) -> bool:
        if quantity <= 0 or reference_price <= 0:
            return False

        current_position = self.portfolio_ledger.get_position(instrument_id)
        signed_qty = quantity if order_side == "buy" else -quantity
        projected_quantity = current_position.quantity + signed_qty
        current_used_margin = self.current_required_margin()
        current_margin_for_instrument = abs(current_position.quantity) * reference_price * self.margin_rate_for_instrument(instrument)
        projected_margin_for_instrument = abs(projected_quantity) * reference_price * self.margin_rate_for_instrument(instrument)
        new_trade_margin = max(0.0, projected_margin_for_instrument - current_margin_for_instrument)
        available_buying_power = self.portfolio_ledger.total_equity() - current_used_margin
        return available_buying_power + 1e-12 >= new_trade_margin

    def current_required_margin(self) -> float:
        required_margin = 0.0
        for instrument_id, position in self.portfolio_ledger.positions.items():
            if abs(position.quantity) < 1e-12:
                continue
            reference_price = self.portfolio_ledger.get_reference_price(instrument_id)
            if reference_price is None or reference_price <= 0:
                continue
            instrument = self._resolve_instrument(instrument_id)
            required_margin += abs(position.quantity) * reference_price * self.margin_rate_for_instrument(instrument)
        return required_margin

    def margin_rate_for_instrument(self, instrument: Instrument) -> float:
        if instrument.margin_requirement is not None:
            return float(instrument.margin_requirement)
        return float(self.margin_by_asset_class[instrument.asset_class])

    def _resolve_instrument(self, instrument_id: str) -> Instrument:
        instrument = self.instruments.get(instrument_id)
        if instrument is not None:
            return instrument

        parsed_symbol = parse_symbol(instrument_id)
        instrument = Instrument(
            instrument_id=parsed_symbol.normalized,
            asset_class=parsed_symbol.asset_class,
        )
        self.instruments[parsed_symbol.normalized] = instrument
        return instrument

    def set_leverage_limit(self, limit: float) -> None:
        """Dynamically adjust the global leverage limit (e.g., from Monte Carlo feedback)."""
        old_limit = self.global_leverage_limit
        self.global_leverage_limit = max(0.1, min(2.0, float(limit)))
        logger.warning(
            "RiskManager leverage limit adjusted: %.2f -> %.2f",
            old_limit,
            self.global_leverage_limit
        )

    def _dispatch_order(self, event: OrderEvent) -> None:
        subscriptions = self.event_bus.subscriptions_for(event.event_type)
        has_async_subscribers = any(subscription.is_async for subscription in subscriptions)
        if not has_async_subscribers:
            self.event_bus.publish(event)
            return

        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            asyncio.run(self.event_bus.publish_async(event))
            return

        loop.create_task(self.event_bus.publish_async(event))
