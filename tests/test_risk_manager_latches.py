import tempfile
from pathlib import Path
from risk.risk_manager import RiskManager
from config import ApexConfig


def test_heal_baselines_recovers_invalid_state(monkeypatch):
    with tempfile.TemporaryDirectory() as tmp_dir:
        monkeypatch.setattr(ApexConfig, "DATA_DIR", Path(tmp_dir))
        manager = RiskManager(max_daily_loss=0.02, max_drawdown=0.1)
        
        # Manually clear to force healing (setter won't work for 0.0 due to guards)
        session = manager.sessions[manager.default_user_id]
        session.starting_capital = 0.0
        session.peak_capital = 0.0
        session.day_start_capital = 0.0

        changed = manager.heal_baselines(current_capital=1_250_000.0, source="test")

        assert changed is True
        assert manager.starting_capital == 1_250_000.0
        assert manager.peak_capital == 1_250_000.0
        assert manager.day_start_capital == 1_250_000.0


def test_manual_reset_circuit_breaker_resets_latch_when_tripped(monkeypatch):
    with tempfile.TemporaryDirectory() as tmp_dir:
        monkeypatch.setattr(ApexConfig, "DATA_DIR", Path(tmp_dir))
        manager = RiskManager(max_daily_loss=0.02, max_drawdown=0.1)
        manager.circuit_breaker.trip("test trip")
        assert manager.circuit_breaker.is_tripped is True

        did_reset = manager.manual_reset_circuit_breaker(
            requested_by="unit-test",
            reason="verified recovery",
        )

        assert did_reset is True
        assert manager.circuit_breaker.is_tripped is False
