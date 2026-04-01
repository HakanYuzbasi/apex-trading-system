import pytest
from types import SimpleNamespace

from config import ApexConfig
from core.execution_loop import ApexTradingSystem


class _DummyCircuitBreaker:
    def __init__(self):
        self.is_tripped = True
        self.reset_called = False

    def reset(self):
        self.is_tripped = False
        self.reset_called = True


class _DummyRiskManager:
    def __init__(self):
        self.starting_capital = 1_250_000.0
        self.peak_capital = 1_300_000.0
        self.day_start_capital = 1_250_000.0
        self.current_day = "2026-01-01"
        self.circuit_breaker = _DummyCircuitBreaker()
        self.save_called = False

    def save_state(self):
        self.save_called = True

    async def save_state_async(self):
        self.save_called = True

    def set_starting_capital(self, value):
        self.starting_capital = value
        self.peak_capital = value
        self.day_start_capital = value


class _DummyPerformanceTracker:
    def __init__(self):
        self.equity_curve = [("2026-01-01T00:00:00", 1_300_000.0)]
        self.reset_calls = []
        self.rebase_calls = []

    async def reset_history(self, *, starting_capital: float, reason: str):
        self.reset_calls.append((float(starting_capital), str(reason)))

    async def rebase_baseline(self, *, starting_capital: float, reason: str, reset_trades: bool = False, clear_benchmark: bool = False):
        self.rebase_calls.append(
            (float(starting_capital), str(reason), bool(reset_trades), bool(clear_benchmark))
        )


class _DummyDrawdownBreaker:
    def __init__(self):
        self.reset_calls = []

    def reset_peak(self, value: float):
        self.reset_calls.append(float(value))


class _DummyEquityOutlierGuard:
    def __init__(self):
        self.seed_calls = []

    def seed(self, value: float):
        self.seed_calls.append(float(value))


@pytest.mark.asyncio
async def test_unified_latch_rebase_for_paper(monkeypatch):
    monkeypatch.setattr(ApexConfig, "UNIFIED_LATCH_RESET_REBASE_RISK_BASELINES", True, raising=False)
    monkeypatch.setattr(ApexConfig, "UNIFIED_LATCH_RESET_REBASE_PERFORMANCE", True, raising=False)
    monkeypatch.setattr(ApexConfig, "LIVE_TRADING", True, raising=False)
    monkeypatch.setattr(ApexConfig, "ALPACA_BASE_URL", "https://paper-api.alpaca.markets", raising=False)

    system = ApexTradingSystem.__new__(ApexTradingSystem)
    system.ibkr = SimpleNamespace(port=7497)
    system.alpaca = None
    system.capital = 120_000.0
    system.risk_manager = _DummyRiskManager()
    system.performance_tracker = _DummyPerformanceTracker()
    system.drawdown_breaker = _DummyDrawdownBreaker()
    system._last_good_total_equity = 0.0

    notes = []
    await system._rebase_latches_after_reset_for_paper(
        requested_by="ops-test",
        reason="manual_reset",
        reset_notes=notes,
    )

    assert system.risk_manager.starting_capital == 120_000.0
    assert system.risk_manager.peak_capital == 120_000.0
    assert system.risk_manager.day_start_capital == 120_000.0
    assert system.performance_tracker.rebase_calls == [
        (120_000.0, "unified_latch_reset", False, False)
    ]
    assert system.performance_tracker.reset_calls == []
    assert system.drawdown_breaker.reset_calls == [120_000.0]
    assert system._last_good_total_equity == 120_000.0
    assert "paper_risk_rebase=applied" in notes
    assert "paper_performance_rebase=applied" in notes


@pytest.mark.asyncio
async def test_startup_state_sanitize_rebases_paper_state(monkeypatch):
    monkeypatch.setattr(ApexConfig, "PAPER_STARTUP_RISK_SELF_HEAL_ENABLED", True, raising=False)
    monkeypatch.setattr(ApexConfig, "PAPER_STARTUP_RISK_MISMATCH_RATIO", 0.30, raising=False)
    monkeypatch.setattr(ApexConfig, "PAPER_STARTUP_RESET_CIRCUIT_BREAKER", True, raising=False)
    monkeypatch.setattr(ApexConfig, "PAPER_STARTUP_PERFORMANCE_REBASE_ENABLED", True, raising=False)
    monkeypatch.setattr(ApexConfig, "PAPER_STARTUP_PERFORMANCE_REBASE_RATIO", 0.30, raising=False)
    monkeypatch.setattr(ApexConfig, "LIVE_TRADING", True, raising=False)
    monkeypatch.setattr(ApexConfig, "ALPACA_BASE_URL", "https://paper-api.alpaca.markets", raising=False)

    system = ApexTradingSystem.__new__(ApexTradingSystem)
    system.ibkr = SimpleNamespace(port=7497)
    system.alpaca = None
    system.capital = 100_000.0
    system.risk_manager = _DummyRiskManager()
    system.performance_tracker = _DummyPerformanceTracker()
    system.drawdown_breaker = _DummyDrawdownBreaker()

    await system._sanitize_startup_state_for_paper()

    assert system.risk_manager.starting_capital == 100_000.0
    assert system.risk_manager.peak_capital == 100_000.0
    assert system.risk_manager.day_start_capital == 100_000.0
    assert system.risk_manager.circuit_breaker.reset_called is True
    assert system.risk_manager.save_called is True
    assert system.performance_tracker.rebase_calls == [
        (100_000.0, "paper_startup_rebase", False, False)
    ]
    assert system.performance_tracker.reset_calls == []
    assert system.drawdown_breaker.reset_calls == [100_000.0]
    assert system._last_good_total_equity == 100_000.0


@pytest.mark.asyncio
async def test_paper_broker_mix_rebase_when_new_broker_joins(monkeypatch):
    monkeypatch.setattr(ApexConfig, "PAPER_STARTUP_RISK_MISMATCH_RATIO", 0.05, raising=False)
    monkeypatch.setattr(ApexConfig, "PAPER_STARTUP_RESET_CIRCUIT_BREAKER", True, raising=False)
    monkeypatch.setattr(ApexConfig, "PAPER_STARTUP_PERFORMANCE_REBASE_ENABLED", True, raising=False)
    monkeypatch.setattr(ApexConfig, "LIVE_TRADING", True, raising=False)
    monkeypatch.setattr(ApexConfig, "ALPACA_BASE_URL", "https://paper-api.alpaca.markets", raising=False)

    system = ApexTradingSystem.__new__(ApexTradingSystem)
    system.ibkr = SimpleNamespace(port=7497)
    system.alpaca = SimpleNamespace()
    system.capital = 1_180_000.0
    system.risk_manager = _DummyRiskManager()
    system.risk_manager.starting_capital = 1_180_000.0
    system.risk_manager.peak_capital = 1_180_000.0
    system.risk_manager.day_start_capital = 1_180_000.0
    system.performance_tracker = _DummyPerformanceTracker()
    system.drawdown_breaker = _DummyDrawdownBreaker()
    system.equity_outlier_guard = _DummyEquityOutlierGuard()
    system._last_good_total_equity = 1_180_000.0
    system._equity_baseline_brokers = {"ibkr"}
    system._current_equity_contributors = {"ibkr", "alpaca"}

    await system._maybe_rebase_paper_baselines_for_broker_mix(current_value=1_260_000.0)

    assert system.risk_manager.starting_capital == 1_260_000.0
    assert system.risk_manager.peak_capital == 1_260_000.0
    assert system.risk_manager.day_start_capital == 1_260_000.0
    assert system.performance_tracker.rebase_calls == [
        (1_260_000.0, "paper_broker_mix_rebase", False, False)
    ]
    assert system.performance_tracker.reset_calls == []
    assert system.drawdown_breaker.reset_calls == [1_260_000.0]
    assert system.equity_outlier_guard.seed_calls == [1_260_000.0]
    assert system.risk_manager.circuit_breaker.reset_called is True
    assert system.risk_manager.save_called is True
    assert system._last_good_total_equity == 1_260_000.0
    assert system._equity_baseline_brokers == {"ibkr", "alpaca"}


@pytest.mark.asyncio
async def test_paper_broker_mix_small_delta_updates_baseline_without_rebase(monkeypatch):
    monkeypatch.setattr(ApexConfig, "PAPER_STARTUP_RISK_MISMATCH_RATIO", 0.10, raising=False)
    monkeypatch.setattr(ApexConfig, "PAPER_STARTUP_PERFORMANCE_REBASE_ENABLED", True, raising=False)
    monkeypatch.setattr(ApexConfig, "LIVE_TRADING", True, raising=False)
    monkeypatch.setattr(ApexConfig, "ALPACA_BASE_URL", "https://paper-api.alpaca.markets", raising=False)

    system = ApexTradingSystem.__new__(ApexTradingSystem)
    system.ibkr = SimpleNamespace(port=7497)
    system.alpaca = SimpleNamespace()
    system.capital = 1_180_000.0
    system.risk_manager = _DummyRiskManager()
    system.risk_manager.starting_capital = 1_180_000.0
    system.risk_manager.peak_capital = 1_180_000.0
    system.risk_manager.day_start_capital = 1_180_000.0
    system.performance_tracker = _DummyPerformanceTracker()
    system.drawdown_breaker = _DummyDrawdownBreaker()
    system.equity_outlier_guard = _DummyEquityOutlierGuard()
    system._equity_baseline_brokers = {"ibkr"}
    system._current_equity_contributors = {"ibkr", "alpaca"}

    await system._maybe_rebase_paper_baselines_for_broker_mix(current_value=1_200_000.0)

    assert system.risk_manager.starting_capital == 1_180_000.0
    assert system.risk_manager.peak_capital == 1_180_000.0
    assert system.risk_manager.day_start_capital == 1_180_000.0
    assert system.performance_tracker.reset_calls == []
    assert system.drawdown_breaker.reset_calls == []
    assert system.equity_outlier_guard.seed_calls == []
    assert system._equity_baseline_brokers == {"ibkr", "alpaca"}
