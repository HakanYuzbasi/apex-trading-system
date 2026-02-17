from monitoring.performance_tracker import PerformanceTracker


def test_performance_tracker_reset_history(tmp_path):
    tracker = PerformanceTracker()
    tracker.data_dir = tmp_path
    tracker.history_file = tmp_path / "performance_history.json"
    tracker.trades = [{"symbol": "AAPL", "side": "BUY", "quantity": 1, "price": 100.0}]
    tracker.equity_curve = [("2026-01-01T00:00:00", 1_250_000.0)]
    tracker.starting_capital = 1_250_000.0

    tracker.reset_history(starting_capital=100_000.0, reason="unit_test")

    assert tracker.trades == []
    assert len(tracker.equity_curve) == 1
    assert float(tracker.equity_curve[0][1]) == 100_000.0
    assert tracker.starting_capital == 100_000.0
    assert tracker.history_file.exists()
