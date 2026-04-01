"""
tests/test_paper_account.py — Unit tests for monitoring/paper_account.py
"""
import json
import time
from pathlib import Path

import pytest

from monitoring.paper_account import PaperAccount, PaperPosition, PaperTrade


# ─── helpers ──────────────────────────────────────────────────────────────────

def make_account(tmp_path=None) -> PaperAccount:
    if tmp_path:
        return PaperAccount(state_dir=tmp_path)
    return PaperAccount()


# ─── TestRecordEntry ──────────────────────────────────────────────────────────

class TestRecordEntry:
    def test_basic_buy_entry(self):
        pa = make_account()
        pa.record_entry("AAPL", "BUY", 175.0, 5000.0)
        assert "AAPL" in pa._positions
        pos = pa._positions["AAPL"]
        assert pos.side == "BUY"
        assert pos.entry_price == 175.0
        assert pos.notional == 5000.0

    def test_sell_side_normalised_uppercase(self):
        pa = make_account()
        pa.record_entry("SPY", "sell", 450.0, 10000.0)
        assert pa._positions["SPY"].side == "SELL"

    def test_zero_price_ignored(self):
        pa = make_account()
        pa.record_entry("AAPL", "BUY", 0.0, 5000.0)
        assert "AAPL" not in pa._positions

    def test_negative_price_ignored(self):
        pa = make_account()
        pa.record_entry("AAPL", "BUY", -1.0, 5000.0)
        assert "AAPL" not in pa._positions

    def test_zero_notional_ignored(self):
        pa = make_account()
        pa.record_entry("AAPL", "BUY", 175.0, 0.0)
        assert "AAPL" not in pa._positions

    def test_overwrites_existing_position(self):
        pa = make_account()
        pa.record_entry("AAPL", "BUY", 175.0, 5000.0)
        pa.record_entry("AAPL", "BUY", 180.0, 6000.0)
        assert pa._positions["AAPL"].entry_price == 180.0


# ─── TestRecordExit ───────────────────────────────────────────────────────────

class TestRecordExit:
    def test_buy_profit(self):
        pa = make_account()
        pa.record_entry("AAPL", "BUY", 175.0, 5000.0)
        pnl = pa.record_exit("AAPL", 177.0)
        # shares = 5000/175 ≈ 28.57; pnl = 2 * 28.57 ≈ 57.14
        assert abs(pnl - 57.14) < 0.02

    def test_buy_loss(self):
        pa = make_account()
        pa.record_entry("AAPL", "BUY", 175.0, 5000.0)
        pnl = pa.record_exit("AAPL", 173.0)
        # shares = 5000/175; pnl = -2 * shares
        assert pnl < 0

    def test_sell_profit(self):
        pa = make_account()
        pa.record_entry("BTC", "SELL", 50000.0, 10000.0)
        pnl = pa.record_exit("BTC", 49000.0)
        assert pnl > 0  # short: entry - exit > 0

    def test_sell_loss(self):
        pa = make_account()
        pa.record_entry("BTC", "SELL", 50000.0, 10000.0)
        pnl = pa.record_exit("BTC", 51000.0)
        assert pnl < 0

    def test_no_open_position_returns_zero(self):
        pa = make_account()
        pnl = pa.record_exit("AAPL", 177.0)
        assert pnl == 0.0

    def test_position_removed_after_exit(self):
        pa = make_account()
        pa.record_entry("AAPL", "BUY", 175.0, 5000.0)
        pa.record_exit("AAPL", 177.0)
        assert "AAPL" not in pa._positions

    def test_closed_trade_appended(self):
        pa = make_account()
        pa.record_entry("AAPL", "BUY", 175.0, 5000.0)
        pa.record_exit("AAPL", 177.0)
        assert len(pa._closed) == 1
        assert pa._closed[0].symbol == "AAPL"

    def test_paper_total_pnl_accumulates(self):
        pa = make_account()
        pa.record_entry("AAPL", "BUY", 175.0, 5000.0)
        pnl1 = pa.record_exit("AAPL", 177.0)
        pa.record_entry("MSFT", "BUY", 400.0, 4000.0)
        pnl2 = pa.record_exit("MSFT", 402.0)
        assert abs(pa._paper_total_pnl - (pnl1 + pnl2)) < 0.01

    def test_history_capped_at_200(self):
        pa = make_account()
        for i in range(210):
            sym = f"SYM{i}"
            pa.record_entry(sym, "BUY", 100.0, 1000.0)
            pa.record_exit(sym, 101.0)
        assert len(pa._closed) == 200

    def test_zero_exit_price_returns_zero(self):
        pa = make_account()
        pa.record_entry("AAPL", "BUY", 175.0, 5000.0)
        pnl = pa.record_exit("AAPL", 0.0)
        assert pnl == 0.0
        # position should still be removed (pop was called)
        assert "AAPL" not in pa._positions


# ─── TestRecordLiveResult ─────────────────────────────────────────────────────

class TestRecordLiveResult:
    def test_shortfall_calculated(self):
        pa = make_account()
        pa.record_entry("AAPL", "BUY", 175.0, 5000.0)
        paper_pnl = pa.record_exit("AAPL", 177.0)
        shortfall = pa.record_live_result("AAPL", live_pnl_usd=50.0)
        assert abs(shortfall - (paper_pnl - 50.0)) < 0.01

    def test_live_total_pnl_increments(self):
        pa = make_account()
        pa.record_entry("AAPL", "BUY", 175.0, 5000.0)
        pa.record_exit("AAPL", 177.0)
        pa.record_live_result("AAPL", live_pnl_usd=50.0)
        assert abs(pa._live_total_pnl - 50.0) < 0.01

    def test_no_matching_trade_still_tracks_live(self):
        pa = make_account()
        shortfall = pa.record_live_result("AAPL", live_pnl_usd=30.0)
        assert shortfall == 0.0
        assert abs(pa._live_total_pnl - 30.0) < 0.01

    def test_optional_paper_pnl_override(self):
        pa = make_account()
        pa.record_entry("AAPL", "BUY", 175.0, 5000.0)
        pa.record_exit("AAPL", 177.0)
        pa.record_live_result("AAPL", live_pnl_usd=50.0, paper_pnl_usd=60.0)
        trade = pa._closed[-1]
        assert abs(trade.pnl_usd - 60.0) < 0.01


# ─── TestImplementationShortfall ─────────────────────────────────────────────

class TestImplementationShortfall:
    def test_positive_shortfall_paper_beats_live(self):
        pa = make_account()
        pa.record_entry("AAPL", "BUY", 175.0, 5000.0)
        pa.record_exit("AAPL", 177.0)
        pa.record_live_result("AAPL", 40.0)
        assert pa.implementation_shortfall_usd > 0

    def test_negative_shortfall_live_beats_paper(self):
        pa = make_account()
        pa.record_entry("AAPL", "BUY", 175.0, 5000.0)
        paper_pnl = pa.record_exit("AAPL", 177.0)
        pa.record_live_result("AAPL", paper_pnl + 10.0)  # live did better
        assert pa.implementation_shortfall_usd < 0

    def test_shortfall_pct_zero_when_no_paper_pnl(self):
        pa = make_account()
        assert pa.shortfall_pct_of_paper == 0.0


# ─── TestGetWinRates ──────────────────────────────────────────────────────────

class TestGetWinRates:
    def test_empty_no_matched_trades(self):
        pa = make_account()
        wr = pa.get_win_rates()
        assert wr == {"paper": 0.0, "live": 0.0, "n": 0}

    def test_matched_trades_counted(self):
        pa = make_account()
        for i in range(5):
            sym = f"SYM{i}"
            pa.record_entry(sym, "BUY", 100.0, 1000.0)
            pa.record_exit(sym, 101.0)
            pa.record_live_result(sym, 8.0)
        wr = pa.get_win_rates()
        assert wr["n"] == 5
        assert 0.0 <= wr["paper"] <= 1.0
        assert 0.0 <= wr["live"] <= 1.0

    def test_unmatched_trades_excluded(self):
        pa = make_account()
        pa.record_entry("AAPL", "BUY", 175.0, 5000.0)
        pa.record_exit("AAPL", 177.0)
        # no record_live_result → live_pnl_usd == 0.0 → unmatched
        wr = pa.get_win_rates()
        assert wr["n"] == 0


# ─── TestGetSnapshot ─────────────────────────────────────────────────────────

class TestGetSnapshot:
    def test_snapshot_keys_present(self):
        pa = make_account()
        snap = pa.get_snapshot()
        for key in (
            "available", "open_positions", "closed_trades",
            "paper_total_pnl", "live_total_pnl",
            "implementation_shortfall_usd", "shortfall_pct",
            "avg_shortfall_per_trade", "win_rates", "day_start_ts", "recent_trades",
        ):
            assert key in snap, f"Missing key: {key}"

    def test_available_true(self):
        pa = make_account()
        assert pa.get_snapshot()["available"] is True

    def test_open_positions_count(self):
        pa = make_account()
        pa.record_entry("AAPL", "BUY", 175.0, 5000.0)
        pa.record_entry("MSFT", "BUY", 400.0, 4000.0)
        assert pa.get_snapshot()["open_positions"] == 2

    def test_recent_trades_limited_to_20(self):
        pa = make_account()
        for i in range(30):
            sym = f"SYM{i}"
            pa.record_entry(sym, "BUY", 100.0, 1000.0)
            pa.record_exit(sym, 101.0)
        snap = pa.get_snapshot()
        assert len(snap["recent_trades"]) <= 20


# ─── TestResetDay ─────────────────────────────────────────────────────────────

class TestResetDay:
    def test_resets_daily_pnl(self):
        pa = make_account()
        pa.record_entry("AAPL", "BUY", 175.0, 5000.0)
        pa.record_exit("AAPL", 177.0)
        pa.record_live_result("AAPL", 40.0)
        pa.reset_day()
        assert pa._paper_total_pnl == 0.0
        assert pa._live_total_pnl == 0.0

    def test_does_not_clear_closed_history(self):
        pa = make_account()
        pa.record_entry("AAPL", "BUY", 175.0, 5000.0)
        pa.record_exit("AAPL", 177.0)
        pa.reset_day()
        assert len(pa._closed) == 1


# ─── TestPersistence ─────────────────────────────────────────────────────────

class TestPersistence:
    def test_save_and_load(self, tmp_path):
        pa = PaperAccount(state_dir=tmp_path)
        pa.record_entry("AAPL", "BUY", 175.0, 5000.0)
        pa.record_exit("AAPL", 177.0)
        pa.record_live_result("AAPL", 50.0)

        pa2 = PaperAccount(state_dir=tmp_path)
        assert abs(pa2._paper_total_pnl - pa._paper_total_pnl) < 0.01
        assert abs(pa2._live_total_pnl - pa._live_total_pnl) < 0.01
        assert len(pa2._closed) == 1
        assert pa2._closed[0].symbol == "AAPL"

    def test_state_file_is_valid_json(self, tmp_path):
        pa = PaperAccount(state_dir=tmp_path)
        pa.record_entry("AAPL", "BUY", 175.0, 5000.0)
        pa.record_exit("AAPL", 177.0)
        state_file = tmp_path / "paper_account_state.json"
        assert state_file.exists()
        data = json.loads(state_file.read_text())
        assert "paper_total_pnl" in data
        assert "closed" in data

    def test_load_missing_file_no_crash(self, tmp_path):
        pa = PaperAccount(state_dir=tmp_path)
        # No file — should just start fresh
        assert pa._paper_total_pnl == 0.0
        assert len(pa._closed) == 0

    def test_no_state_dir_no_save(self):
        pa = PaperAccount()  # no state_dir
        pa.record_entry("AAPL", "BUY", 175.0, 5000.0)
        pa.record_exit("AAPL", 177.0)
        # Should not raise
        assert pa._state_dir is None
