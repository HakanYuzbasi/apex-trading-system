"""tests/test_trade_postmortem.py — Trade post-mortem intelligence tests."""
from __future__ import annotations

from monitoring.trade_postmortem import (
    TradePostMortem,
    PostMortem,
    _STRONG_SIGNAL,
    _WEAK_SIGNAL,
    _GOOD_CONF,
    _SLIPPAGE_BAD,
    _PNL_WINNER,
    _PNL_LOSER,
)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _pm(tpm: TradePostMortem, **kwargs) -> PostMortem:
    defaults = dict(
        symbol="AAPL",
        pnl_pct=0.01,
        hold_hours=4.0,
        exit_reason="signal",
        signal_at_entry=0.18,
        confidence_at_entry=0.70,
        regime="neutral",
        slippage_bps=5.0,
        optimal_hold_hours=4.0,
    )
    defaults.update(kwargs)
    return tpm.analyze(**defaults)


# ── PostMortem dataclass ──────────────────────────────────────────────────────

class TestPostMortemDataclass:
    def test_to_dict_has_all_keys(self):
        tpm = TradePostMortem()
        pm = _pm(tpm)
        d = pm.to_dict()
        for key in (
            "symbol", "pnl_pct", "hold_hours", "exit_reason",
            "signal_quality", "timing", "regime_alignment", "execution_drag",
            "verdict", "primary_failure", "confidence_at_entry",
            "signal_at_entry", "slippage_bps", "regime", "timestamp",
        ):
            assert key in d, f"missing key: {key}"

    def test_symbol_preserved(self):
        tpm = TradePostMortem()
        pm = _pm(tpm, symbol="TSLA")
        assert pm.symbol == "TSLA"

    def test_pnl_preserved(self):
        tpm = TradePostMortem()
        pm = _pm(tpm, pnl_pct=0.025)
        assert abs(pm.pnl_pct - 0.025) < 1e-9


# ── Verdict classification ────────────────────────────────────────────────────

class TestVerdict:
    def test_winner(self):
        tpm = TradePostMortem()
        pm = _pm(tpm, pnl_pct=_PNL_WINNER + 0.001)
        assert pm.verdict == "winner"

    def test_loser(self):
        tpm = TradePostMortem()
        pm = _pm(tpm, pnl_pct=_PNL_LOSER - 0.001)
        assert pm.verdict == "loser"

    def test_breakeven(self):
        tpm = TradePostMortem()
        pm = _pm(tpm, pnl_pct=0.0)
        assert pm.verdict == "breakeven"

    def test_borderline_positive(self):
        tpm = TradePostMortem()
        pm = _pm(tpm, pnl_pct=0.001)  # between loser and winner
        assert pm.verdict == "breakeven"


# ── Signal quality assessment ─────────────────────────────────────────────────

class TestSignalQuality:
    def test_strong_signal_good_confidence_is_good(self):
        tpm = TradePostMortem()
        pm = _pm(tpm, signal_at_entry=_STRONG_SIGNAL + 0.01, confidence_at_entry=_GOOD_CONF + 0.01)
        assert pm.signal_quality == "good"

    def test_weak_signal_is_bad(self):
        tpm = TradePostMortem()
        pm = _pm(tpm, signal_at_entry=_WEAK_SIGNAL - 0.01, confidence_at_entry=0.70)
        assert pm.signal_quality == "bad"

    def test_low_confidence_is_bad(self):
        tpm = TradePostMortem()
        pm = _pm(tpm, signal_at_entry=0.18, confidence_at_entry=0.40)
        assert pm.signal_quality == "bad"

    def test_moderate_signal_is_neutral(self):
        tpm = TradePostMortem()
        pm = _pm(tpm, signal_at_entry=0.15, confidence_at_entry=0.55)
        assert pm.signal_quality == "neutral"


# ── Timing assessment ─────────────────────────────────────────────────────────

class TestTiming:
    def test_stop_loss_loser_is_bad(self):
        tpm = TradePostMortem()
        pm = _pm(tpm, exit_reason="stop_loss", pnl_pct=-0.03)
        assert pm.timing == "bad"

    def test_very_short_hold_loser_is_bad(self):
        tpm = TradePostMortem()
        pm = _pm(tpm, hold_hours=0.2, pnl_pct=-0.01)
        assert pm.timing == "bad"

    def test_optimal_hold_is_good(self):
        tpm = TradePostMortem()
        pm = _pm(tpm, hold_hours=4.0, optimal_hold_hours=4.0, pnl_pct=0.01, exit_reason="signal")
        assert pm.timing == "good"

    def test_way_too_long_hold_neutral_or_bad(self):
        tpm = TradePostMortem()
        pm = _pm(tpm, hold_hours=30.0, optimal_hold_hours=4.0, pnl_pct=-0.01, exit_reason="max_hold")
        assert pm.timing in ("neutral", "bad")


# ── Regime alignment assessment ───────────────────────────────────────────────

class TestRegimeAlignment:
    def test_bull_regime_is_good(self):
        tpm = TradePostMortem()
        pm = _pm(tpm, regime="bull", pnl_pct=0.01)
        assert pm.regime_alignment == "good"

    def test_bear_regime_loser_is_bad(self):
        tpm = TradePostMortem()
        pm = _pm(tpm, regime="bear", pnl_pct=-0.02)
        assert pm.regime_alignment == "bad"

    def test_crisis_regime_loser_is_bad(self):
        tpm = TradePostMortem()
        pm = _pm(tpm, regime="crisis", pnl_pct=-0.01)
        assert pm.regime_alignment == "bad"

    def test_neutral_regime_is_good(self):
        tpm = TradePostMortem()
        pm = _pm(tpm, regime="neutral", pnl_pct=0.01)
        assert pm.regime_alignment == "good"


# ── Execution drag assessment ─────────────────────────────────────────────────

class TestExecutionDrag:
    def test_low_slippage_is_good(self):
        tpm = TradePostMortem()
        pm = _pm(tpm, slippage_bps=5.0, pnl_pct=0.01)
        assert pm.execution_drag == "good"

    def test_high_slippage_loser_is_bad(self):
        tpm = TradePostMortem()
        pm = _pm(tpm, slippage_bps=_SLIPPAGE_BAD + 5.0, pnl_pct=-0.01)
        assert pm.execution_drag == "bad"

    def test_high_slippage_winner_is_neutral(self):
        tpm = TradePostMortem()
        pm = _pm(tpm, slippage_bps=_SLIPPAGE_BAD + 5.0, pnl_pct=0.03)
        assert pm.execution_drag == "neutral"


# ── Primary failure ───────────────────────────────────────────────────────────

class TestPrimaryFailure:
    def test_winner_has_no_failure(self):
        tpm = TradePostMortem()
        pm = _pm(tpm, pnl_pct=0.02, signal_at_entry=0.20, confidence_at_entry=0.70)
        assert pm.primary_failure == "none"

    def test_weak_signal_identified(self):
        tpm = TradePostMortem()
        pm = _pm(tpm, pnl_pct=-0.02, signal_at_entry=0.05, confidence_at_entry=0.70, regime="neutral")
        assert pm.primary_failure == "weak_signal"

    def test_bad_regime_identified(self):
        tpm = TradePostMortem()
        pm = _pm(tpm, pnl_pct=-0.02, signal_at_entry=0.18, confidence_at_entry=0.70, regime="crisis")
        assert pm.primary_failure == "bad_regime"

    def test_high_slippage_identified_as_primary(self):
        tpm = TradePostMortem()
        pm = _pm(tpm, pnl_pct=-0.02, signal_at_entry=0.18, confidence_at_entry=0.70,
                 regime="neutral", slippage_bps=50.0)
        assert pm.primary_failure == "slippage"

    def test_stop_hit_identified(self):
        tpm = TradePostMortem()
        pm = _pm(tpm, pnl_pct=-0.01, exit_reason="stop_loss", signal_at_entry=0.16,
                 confidence_at_entry=0.65, regime="neutral", slippage_bps=3.0)
        assert pm.primary_failure == "stop_hit"


# ── Ring buffer and history ───────────────────────────────────────────────────

class TestHistory:
    def test_get_recent_empty(self):
        tpm = TradePostMortem()
        assert tpm.get_recent(10) == []

    def test_get_recent_most_recent_first(self):
        tpm = TradePostMortem()
        for i in range(5):
            _pm(tpm, symbol=f"SYM{i}")
        recent = tpm.get_recent(3)
        assert recent[0]["symbol"] == "SYM4"
        assert recent[1]["symbol"] == "SYM3"

    def test_ring_buffer_capped_at_max(self):
        tpm = TradePostMortem(max_history=5)
        for _ in range(10):
            _pm(tpm)
        assert len(list(tpm._history)) == 5

    def test_get_recent_respects_n(self):
        tpm = TradePostMortem()
        for _ in range(10):
            _pm(tpm)
        assert len(tpm.get_recent(3)) == 3


# ── Summary aggregation ───────────────────────────────────────────────────────

class TestSummary:
    def test_empty_summary(self):
        tpm = TradePostMortem()
        s = tpm.get_summary()
        assert s["total"] == 0

    def test_win_rate_correct(self):
        tpm = TradePostMortem()
        for _ in range(7):
            _pm(tpm, pnl_pct=0.01)   # winners
        for _ in range(3):
            _pm(tpm, pnl_pct=-0.01)  # losers
        s = tpm.get_summary()
        assert abs(s["win_rate"] - 0.70) < 0.01

    def test_verdict_counts_populated(self):
        tpm = TradePostMortem()
        _pm(tpm, pnl_pct=0.02)
        _pm(tpm, pnl_pct=-0.02)
        _pm(tpm, pnl_pct=0.0)
        s = tpm.get_summary()
        assert s["verdict_counts"]["winner"] == 1
        assert s["verdict_counts"]["loser"] == 1
        assert s["verdict_counts"]["breakeven"] == 1

    def test_failure_counts_include_most_common(self):
        tpm = TradePostMortem()
        for _ in range(5):
            _pm(tpm, pnl_pct=-0.02, signal_at_entry=0.05)  # weak_signal
        s = tpm.get_summary()
        assert "weak_signal" in s["failure_counts"]

    def test_avg_pnl_correct(self):
        tpm = TradePostMortem()
        _pm(tpm, pnl_pct=0.02)
        _pm(tpm, pnl_pct=-0.02)
        s = tpm.get_summary()
        assert abs(s["avg_pnl_pct"]) < 1e-6
