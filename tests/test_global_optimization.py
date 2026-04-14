import unittest
import math
import pandas as pd
import numpy as np
from datetime import datetime, timedelta, timezone
from quant_system.risk.bayesian_vol import BayesianVolatilityAdjuster
from quant_system.analytics.performance import PerformanceAnalyzer
from quant_system.core.bus import InMemoryEventBus
from quant_system.events import BarEvent

class TestGlobalOptimization(unittest.TestCase):
    def setUp(self):
        self.event_bus = InMemoryEventBus()

    def test_bayesian_vol_update(self):
        # Initialize with a high persistence and distinct vols
        adjuster = BayesianVolatilityAdjuster(self.event_bus, low_vol_annualized=0.1, high_vol_annualized=1.0)
        
        # Test Case 1: Low volatility sequence
        # Small changes should keep prob_high_vol low
        instrument = "TEST"
        prices = [100.0, 100.1, 100.2, 100.1, 100.0]
        now = datetime.now(timezone.utc)
        for i, price in enumerate(prices):
            self.event_bus.publish(BarEvent(
                instrument_id=instrument,
                exchange_ts=now + timedelta(minutes=i),
                received_ts=now + timedelta(minutes=i),
                processed_ts=now + timedelta(minutes=i),
                sequence_id=i,
                source="test",
                open_price=price, high_price=price, low_price=price, close_price=price, volume=100
            ))
        
        low_prob = adjuster.probability_of_high_vol(instrument)
        self.assertLess(low_prob, 0.5)

        # Test Case 2: Sudden high volatility jump
        # A 10% move in one step
        jump_ts = now + timedelta(minutes=len(prices))
        self.event_bus.publish(BarEvent(
            instrument_id=instrument,
            exchange_ts=jump_ts,
            received_ts=jump_ts,
            processed_ts=jump_ts,
            sequence_id=len(prices),
            source="test",
            open_price=110.0, high_price=110.0, low_price=110.0, close_price=110.0, volume=100
        ))
        high_prob = adjuster.probability_of_high_vol(instrument)
        self.assertGreater(high_prob, low_prob)

    def test_performance_metrics(self):
        # Sample equity curve with a drawdown
        # 100 -> 110 -> 105 -> 120
        # Returns: 10%, -4.54%, 14.28%
        dates = [datetime(2026, 1, 1) + timedelta(days=i) for i in range(4)]
        equity = pd.Series([100.0, 110.0, 105.0, 120.0], index=pd.DatetimeIndex(dates))
        
        returns = equity.pct_change().dropna()
        
        # Omega Ratio calculation (threshold 0)
        # Gains: 0.10 + 0.1428 = 0.2428
        # Losses: 0.0454
        # Omega: 0.2428 / 0.0454 ~= 5.34
        omega = PerformanceAnalyzer._omega_ratio(returns, threshold=0.0)
        self.assertGreater(omega, 0)
        
        # Ulcer Index
        # Drawdowns: 0, 0, -0.0454, 0
        # Squared Drawdowns (pct): 0, 0, (4.54)^2, 0
        # Mean Squared: (4.54^2) / 4 = 20.61 / 4 = 5.15
        # UI: sqrt(5.15) ~= 2.27
        ui = PerformanceAnalyzer._ulcer_index(equity)
        self.assertGreater(ui, 0)
        self.assertLess(ui, 5.0)

if __name__ == "__main__":
    unittest.main()
