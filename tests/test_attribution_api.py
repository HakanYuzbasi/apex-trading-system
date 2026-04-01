"""tests/test_attribution_api.py — Performance attribution API tests."""
from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path

from monitoring.performance_attribution import PerformanceAttributionTracker as PerformanceAttribution


# ── Helpers ────────────────────────────────────────────────────────────────────

def _pa(tmp_path: Path) -> PerformanceAttribution:
    return PerformanceAttribution(data_dir=tmp_path)


def _closed_row(
    symbol: str = "AAPL",
    net_pnl: float = 100.0,
    gross_pnl: float = 110.0,
    asset_class: str = "EQUITY",
    sleeve: str = "ibkr",
    exit_time: str | None = None,
    holding_hours: float = 4.0,
    ml_signal: float = 0.0,
    tech_signal: float = 0.0,
    sentiment_signal: float = 0.0,
    cs_momentum_signal: float = 0.0,
    governor_regime: str = "neutral",
) -> dict:
    if exit_time is None:
        exit_time = datetime.now().isoformat()
    return {
        "symbol": symbol,
        "net_pnl": net_pnl,
        "gross_pnl": gross_pnl,
        "commissions": 5.0,
        "modeled_execution_drag": 2.0,
        "modeled_entry_slippage_cost": 1.0,
        "modeled_exit_slippage_cost": 2.0,
        "asset_class": asset_class,
        "sleeve": sleeve,
        "exit_time": exit_time,
        "holding_hours": holding_hours,
        "ml_signal": ml_signal,
        "tech_signal": tech_signal,
        "sentiment_signal": sentiment_signal,
        "cs_momentum_signal": cs_momentum_signal,
        "governor_regime": governor_regime,
        "pnl_bps_on_entry_notional": 50.0,
    }


def _inject_trades(pa: PerformanceAttribution, rows: list[dict]) -> None:
    pa.closed_trades = rows


# ── get_summary ────────────────────────────────────────────────────────────────

class TestGetSummary:
    def test_empty_summary_returns_zero_trades(self, tmp_path):
        pa = _pa(tmp_path)
        s = pa.get_summary()
        assert s["closed_trades"] == 0

    def test_single_trade_counted(self, tmp_path):
        pa = _pa(tmp_path)
        _inject_trades(pa, [_closed_row()])
        s = pa.get_summary()
        assert s["closed_trades"] == 1

    def test_gross_pnl_summed(self, tmp_path):
        pa = _pa(tmp_path)
        _inject_trades(pa, [
            _closed_row(gross_pnl=100.0),
            _closed_row(gross_pnl=200.0),
        ])
        s = pa.get_summary()
        assert abs(s["gross_pnl"] - 300.0) < 1e-6

    def test_net_pnl_summed(self, tmp_path):
        pa = _pa(tmp_path)
        _inject_trades(pa, [
            _closed_row(net_pnl=80.0),
            _closed_row(net_pnl=120.0),
        ])
        s = pa.get_summary()
        assert abs(s["net_pnl"] - 200.0) < 1e-6

    def test_by_asset_class_populated(self, tmp_path):
        pa = _pa(tmp_path)
        _inject_trades(pa, [
            _closed_row(asset_class="EQUITY"),
            _closed_row(asset_class="CRYPTO"),
        ])
        s = pa.get_summary()
        assert "EQUITY" in s["by_asset_class"]
        assert "CRYPTO" in s["by_asset_class"]

    def test_by_sleeve_populated(self, tmp_path):
        pa = _pa(tmp_path)
        _inject_trades(pa, [
            _closed_row(sleeve="ibkr"),
            _closed_row(sleeve="alpaca"),
        ])
        s = pa.get_summary()
        assert "ibkr" in s["by_sleeve"]
        assert "alpaca" in s["by_sleeve"]

    def test_lookback_filters_old_trades(self, tmp_path):
        pa = _pa(tmp_path)
        old_time = (datetime.now() - timedelta(days=60)).isoformat()
        _inject_trades(pa, [
            _closed_row(net_pnl=999.0, exit_time=old_time),
            _closed_row(net_pnl=1.0),
        ])
        s = pa.get_summary(lookback_days=30)
        assert s["closed_trades"] == 1
        assert abs(s["net_pnl"] - 1.0) < 1e-6

    def test_avg_net_pnl_per_bucket(self, tmp_path):
        pa = _pa(tmp_path)
        _inject_trades(pa, [
            _closed_row(net_pnl=100.0, sleeve="ibkr"),
            _closed_row(net_pnl=200.0, sleeve="ibkr"),
        ])
        s = pa.get_summary()
        assert abs(s["by_sleeve"]["ibkr"]["avg_net_pnl"] - 150.0) < 1e-6


# ── get_signal_source_summary ─────────────────────────────────────────────────

class TestSignalSourceSummary:
    def test_empty_returns_lookback_days(self, tmp_path):
        pa = _pa(tmp_path)
        ss = pa.get_signal_source_summary(lookback_days=14)
        assert ss["lookback_days"] == 14

    def test_ml_dominant_classified(self, tmp_path):
        pa = _pa(tmp_path)
        _inject_trades(pa, [_closed_row(ml_signal=0.8, tech_signal=0.1, net_pnl=50.0)])
        ss = pa.get_signal_source_summary()
        assert "ml" in ss["by_signal_source"]

    def test_tech_dominant_classified(self, tmp_path):
        pa = _pa(tmp_path)
        _inject_trades(pa, [_closed_row(tech_signal=0.9, ml_signal=0.1, net_pnl=50.0)])
        ss = pa.get_signal_source_summary()
        assert "technical" in ss["by_signal_source"]

    def test_composite_when_all_zero(self, tmp_path):
        pa = _pa(tmp_path)
        _inject_trades(pa, [_closed_row(net_pnl=50.0)])
        ss = pa.get_signal_source_summary()
        assert "composite" in ss["by_signal_source"]

    def test_win_rate_computed(self, tmp_path):
        pa = _pa(tmp_path)
        _inject_trades(pa, [
            _closed_row(ml_signal=0.8, net_pnl=100.0),
            _closed_row(ml_signal=0.8, net_pnl=-50.0),
            _closed_row(ml_signal=0.8, net_pnl=75.0),
        ])
        ss = pa.get_signal_source_summary()
        ml = ss["by_signal_source"]["ml"]
        assert abs(ml["win_rate"] - 2/3) < 0.01

    def test_avg_net_pnl_computed(self, tmp_path):
        pa = _pa(tmp_path)
        _inject_trades(pa, [
            _closed_row(tech_signal=0.9, net_pnl=100.0),
            _closed_row(tech_signal=0.9, net_pnl=200.0),
        ])
        ss = pa.get_signal_source_summary()
        tech = ss["by_signal_source"]["technical"]
        assert abs(tech["avg_net_pnl"] - 150.0) < 1e-6

    def test_multiple_sources_present(self, tmp_path):
        pa = _pa(tmp_path)
        _inject_trades(pa, [
            _closed_row(ml_signal=0.8, net_pnl=50.0),
            _closed_row(tech_signal=0.9, net_pnl=30.0),
            _closed_row(sentiment_signal=0.7, net_pnl=20.0),
        ])
        ss = pa.get_signal_source_summary()
        sources = ss["by_signal_source"]
        assert "ml" in sources
        assert "technical" in sources
        assert "sentiment" in sources

    def test_trades_per_source_counted(self, tmp_path):
        pa = _pa(tmp_path)
        _inject_trades(pa, [
            _closed_row(ml_signal=0.8, net_pnl=10.0),
            _closed_row(ml_signal=0.8, net_pnl=20.0),
            _closed_row(ml_signal=0.8, net_pnl=30.0),
        ])
        ss = pa.get_signal_source_summary()
        assert ss["by_signal_source"]["ml"]["trades"] == 3
