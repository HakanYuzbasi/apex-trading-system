import logging

from core.hot_path_profiler import HotPathProfiler


def test_hot_path_profiler_records_summary():
    profiler = HotPathProfiler(
        logger=logging.getLogger("test.hot_path_profiler"),
        enabled=True,
        slow_threshold_ms=50.0,
        summary_interval_cycles=10,
    )

    profiler.record_duration("refresh_data", 20.0)
    profiler.record_duration("refresh_data", 40.0)
    profiler.record_duration("process_symbols_parallel", 100.0)

    snapshot = profiler.snapshot()

    assert snapshot["refresh_data"]["count"] == 2.0
    assert snapshot["refresh_data"]["avg_ms"] == 30.0
    assert snapshot["refresh_data"]["max_ms"] == 40.0
    assert snapshot["process_symbols_parallel"]["p95_ms"] == 100.0


def test_hot_path_profiler_disabled_is_noop():
    profiler = HotPathProfiler(
        logger=logging.getLogger("test.hot_path_profiler"),
        enabled=False,
    )

    profiler.record_duration("refresh_data", 25.0)

    assert profiler.snapshot() == {}
