from monitoring.model_tracker import ModelPerformanceTracker, PROMETHEUS_AVAILABLE


def test_model_tracker_reuses_prometheus_collectors() -> None:
    first = ModelPerformanceTracker(window_size=50)
    second = ModelPerformanceTracker(window_size=100)

    if not PROMETHEUS_AVAILABLE:
        assert first.model_accuracy is None
        assert second.model_accuracy is None
        return

    assert first.model_accuracy is second.model_accuracy
    assert first.prediction_error is second.prediction_error
