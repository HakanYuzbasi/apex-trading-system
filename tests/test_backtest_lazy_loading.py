from types import SimpleNamespace

from api.routers import backtest as backtest_router


def test_backtest_runtime_imports_are_lazy_and_cached(monkeypatch):
    calls = []

    class DummyBacktester:
        pass

    def fake_import_module(name):
        calls.append(name)
        if name == "pandas":
            return SimpleNamespace(date_range=lambda *args, **kwargs: [])
        if name == "numpy":
            return SimpleNamespace(random=SimpleNamespace(randn=lambda *args, **kwargs: 0.0))
        if name == "backtesting.advanced_backtester":
            return SimpleNamespace(AdvancedBacktester=DummyBacktester)
        raise AssertionError(f"unexpected import: {name}")

    monkeypatch.setattr(backtest_router.importlib, "import_module", fake_import_module)
    backtest_router._load_backtest_runtime.cache_clear()

    pandas_module, numpy_module, backtester_cls = backtest_router._load_backtest_runtime()
    assert pandas_module is not None
    assert numpy_module is not None
    assert backtester_cls is DummyBacktester
    assert calls == ["pandas", "numpy", "backtesting.advanced_backtester"]

    backtest_router._load_backtest_runtime()
    assert calls == ["pandas", "numpy", "backtesting.advanced_backtester"]
