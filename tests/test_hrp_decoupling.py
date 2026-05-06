from unittest.mock import MagicMock, patch
import pytest
from datetime import datetime, timezone
from quant_system.events import SignalEvent
from quant_system.risk.manager import RiskManager

@pytest.fixture
def risk_manager():
    bus = MagicMock()
    ledger = MagicMock()
    ledger.total_equity.return_value = 10000.0
    ledger.cash = 10000.0
    ledger.get_position.return_value.quantity = 0
    ledger.get_reference_price.return_value = 100.0
    
    rm = RiskManager(ledger, bus)
    rm.global_leverage_limit = 1.0
    return rm

def _create_signal(target_value=1000.0):
    return SignalEvent(
        instrument_id="AAPL",
        side="buy",
        target_type="notional",
        target_value=target_value,
        source="test",
        exchange_ts=datetime.now(timezone.utc),
        received_ts=datetime.now(timezone.utc),
        processed_ts=datetime.now(timezone.utc),
        sequence_id=1,
        strategy_id="ORB",
        confidence=1.0,
        stop_model="fixed"
    )

def test_hrp_applies_in_passthrough_mode(risk_manager):
    risk_manager.meta_labeler = MagicMock()
    risk_manager.meta_labeler.predict_confidence.return_value = 1.0
    
    risk_manager.hrp_optimizer = MagicMock()
    risk_manager.hrp_optimizer.current_regime = 2 # 0.7 scalar
    
    signal = _create_signal()
    
    with patch.object(risk_manager, '_dispatch_order') as mock_dispatch:
        risk_manager._on_signal(signal)
        order = mock_dispatch.call_args[0][0]
        assert order.notional == 700.0

def test_hrp_applies_in_active_mode(risk_manager):
    risk_manager.meta_labeler = MagicMock()
    risk_manager.meta_labeler.predict_confidence.return_value = 0.8
    
    risk_manager.hrp_optimizer = MagicMock()
    risk_manager.hrp_optimizer.current_regime = 2 
    
    signal = _create_signal()
    
    with patch.object(risk_manager, '_dispatch_order') as mock_dispatch:
        risk_manager._on_signal(signal)
        order = mock_dispatch.call_args[0][0]
        assert order.notional == pytest.approx(560.0)

def test_hrp_runs_before_metalabeler(risk_manager):
    call_order = []
    
    risk_manager.meta_labeler = MagicMock()
    def ml_side_effect(*args, **kwargs):
        call_order.append("metalabeler")
        return 1.0
    risk_manager.meta_labeler.predict_confidence.side_effect = ml_side_effect
    
    risk_manager.hrp_optimizer = MagicMock()
    risk_manager.hrp_optimizer.current_regime = 2
    
    with patch.object(risk_manager, '_quantity_from_signal_v2') as mock_qty:
        def qty_side_effect(*args, **kwargs):
            call_order.append("qty_calc")
            return 10.0
        mock_qty.side_effect = qty_side_effect
        
        signal = _create_signal()
        risk_manager._on_signal(signal)
        
        # Qty calc should happen for HRP (STEP 2) BEFORE MetaLabeler (STEP 3)
        assert "qty_calc" in call_order
        assert "metalabeler" in call_order
        first_ml = call_order.index("metalabeler")
        assert "qty_calc" in call_order[:first_ml]

def test_notional_pipeline_order_is_base_hrp_ml_allocator(risk_manager):
    risk_manager.meta_labeler = MagicMock()
    risk_manager.meta_labeler.predict_confidence.return_value = 0.8
    
    risk_manager.hrp_optimizer = MagicMock()
    risk_manager.hrp_optimizer.current_regime = 2 
    
    signal = _create_signal(1000.0)
    
    with patch.object(risk_manager, '_dispatch_order') as mock_dispatch:
        risk_manager._on_signal(signal)
        order = mock_dispatch.call_args[0][0]
        assert order.notional == pytest.approx(560.0)
