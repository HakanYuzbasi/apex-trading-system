from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict

from quant_system.core.bus import InMemoryEventBus, Subscription
from quant_system.events import BarEvent


@dataclass(slots=True)
class BayesianVolState:
    prob_high_vol: float = 0.1  # Prior probability of being in high vol state
    last_price: float | None = None


class BayesianVolatilityAdjuster:
    """
    Recursive Bayesian volatility estimator for pre-emptive risk scaling.
    Estimates the probability that the current market regime is 'High Volatility'
    based on observed returns and a two-state Gaussian mixture model.
    """

    def __init__(
        self,
        event_bus: InMemoryEventBus,
        *,
        low_vol_annualized: float = 0.15,
        high_vol_annualized: float = 0.60,
        p_stay_low: float = 0.95,  # Skeptic of low vol
        p_stay_high: float = 0.99, # Hyper-aware/Sticky high vol
    ) -> None:
        self._event_bus = event_bus
        # Convert annualized vol to per-period (assuming daily bars for now, 
        # or scaling by time if bars are intraday)
        # We use a base of 252 days/year for the likelihoods.
        self._low_vol = low_vol_annualized / math.sqrt(252)
        self._high_vol = high_vol_annualized / math.sqrt(252)
        self._p_stay_low = p_stay_low
        self._p_stay_high = p_stay_high
        self._states: Dict[str, BayesianVolState] = {}
        self._subscription: Subscription = self._event_bus.subscribe("bar", self._on_bar)

    def probability_of_high_vol(self, instrument_id: str) -> float:
        return self._states.get(instrument_id, BayesianVolState()).prob_high_vol

    def close(self) -> None:
        self._event_bus.unsubscribe(self._subscription.token)

    def _on_bar(self, event: BarEvent) -> None:
        state = self._states.setdefault(event.instrument_id, BayesianVolState())
        
        if state.last_price is None or state.last_price <= 0:
            state.last_price = event.close_price
            return

        # 1. Calculate log return
        ret = math.log(event.close_price / state.last_price)
        state.last_price = event.close_price

        # 2. Prediction Step (Asymmetric Transition)
        # P(S_t | r_{1:t-1}) = sum_{S_{t-1}} P(S_t | S_{t-1}) P(S_{t-1} | r_{1:t-1})
        p_low_prev = 1.0 - state.prob_high_vol
        p_high_prev = state.prob_high_vol

        p_low_pred = p_low_prev * self._p_stay_low + p_high_prev * (1.0 - self._p_stay_high)
        p_high_pred = p_high_prev * self._p_stay_high + p_low_prev * (1.0 - self._p_stay_low)

        # 3. Update Step (Likelihood)
        # P(r_t | S_t) ~ N(0, sigma^2)
        l_low = self._gaussian_pdf(ret, 0.0, self._low_vol)
        l_high = self._gaussian_pdf(ret, 0.0, self._high_vol)

        # 4. Posterior
        denom = (l_low * p_low_pred) + (l_high * p_high_pred)
        if denom > 0:
            state.prob_high_vol = (l_high * p_high_pred) / denom
        
        # Clamp to avoid numerical issues
        state.prob_high_vol = max(0.01, min(0.99, state.prob_high_vol))

    @staticmethod
    def _gaussian_pdf(x: float, mu: float, sigma: float) -> float:
        if sigma <= 0:
            return 0.0
        exponent = -0.5 * ((x - mu) / sigma) ** 2
        return (1.0 / (sigma * math.sqrt(2 * math.pi))) * math.exp(exponent)
