"""
tests/test_stabilizer_backoff.py

Tests for JohansenStabilizer backoff improvements:
- fail_streak raised to 8
- failure-rate trigger (>0.75 over last 20)
- 15-min minimum Read-Only duration
- exponential backoff: 15 / 30 / 60 min
- WARN logged only once per Read-Only period
"""
from __future__ import annotations

import logging
from unittest.mock import patch

import numpy as np
import pytest

from quant_system.strategies.stabilizer import (
    JohansenStabilizer,
    _BACKOFF_SCHEDULE_MINUTES,
)


def _make_prices(n: int = 80) -> tuple[np.ndarray, np.ndarray]:
    """Return two random-walk series long enough to pass the min_window guard."""
    rng = np.random.default_rng(42)
    a = np.cumsum(rng.standard_normal(n)) + 100
    b = np.cumsum(rng.standard_normal(n)) + 100
    return a, b


# ---------------------------------------------------------------------------
# 1. Consecutive-failure threshold is now 8
# ---------------------------------------------------------------------------

def test_does_not_enter_readonly_before_8_failures():
    stab = JohansenStabilizer(fail_streak=8)
    stab.is_read_only = False

    # Inject 7 failures directly via _update_state
    t0 = 0.0
    with patch("quant_system.strategies.stabilizer.time.monotonic", return_value=t0):
        for _ in range(7):
            stab._update_state(False)

    assert not stab.is_read_only
    assert stab._consecutive_fails == 7


def test_enters_readonly_on_8th_failure():
    stab = JohansenStabilizer(fail_streak=8)

    t0 = 0.0
    with patch("quant_system.strategies.stabilizer.time.monotonic", return_value=t0):
        for _ in range(8):
            stab._history.append(False)
            stab._update_state(False)

    assert stab.is_read_only


# ---------------------------------------------------------------------------
# 2. Failure-rate trigger
# ---------------------------------------------------------------------------

def test_failure_rate_triggers_readonly():
    """>0.75 failure rate over 20 tests must enter Read-Only even if fail_streak not hit."""
    stab = JohansenStabilizer(
        fail_streak=8,
        failure_rate_window=20,
        failure_rate_threshold=0.75,
    )

    t0 = 0.0
    with patch("quant_system.strategies.stabilizer.time.monotonic", return_value=t0):
        # 16 failures + 4 passes = 80% failure rate; fail_streak only reaches 4 (broken by passes)
        for i in range(20):
            passed = (i % 5 == 0)  # every 5th is a pass → 4 passes, 16 fails
            stab._history.append(passed)
            stab._update_state(passed)

    assert stab.is_read_only


# ---------------------------------------------------------------------------
# 3. Minimum 15-min duration prevents early exit
# ---------------------------------------------------------------------------

def test_min_duration_prevents_early_exit():
    stab = JohansenStabilizer(min_readonly_minutes=15.0, pass_streak=2)
    t0 = 1000.0
    # Enter Read-Only
    with patch("quant_system.strategies.stabilizer.time.monotonic", return_value=t0):
        stab.is_read_only = True
        stab._readonly_since = t0
        stab._readonly_warn_logged = True

    # Attempt exit 5 minutes later — too soon
    with patch("quant_system.strategies.stabilizer.time.monotonic", return_value=t0 + 300):
        stab._update_state(True)
        stab._update_state(True)  # two passes but duration not met

    assert stab.is_read_only


def test_exits_readonly_after_min_duration_and_passes():
    stab = JohansenStabilizer(min_readonly_minutes=15.0, pass_streak=2)
    t0 = 1000.0
    # Enter Read-Only
    with patch("quant_system.strategies.stabilizer.time.monotonic", return_value=t0):
        stab.is_read_only = True
        stab._readonly_since = t0
        stab._readonly_warn_logged = True

    # Exit 20 minutes later with enough passes
    with patch("quant_system.strategies.stabilizer.time.monotonic", return_value=t0 + 1200):
        stab._update_state(True)
        stab._update_state(True)

    assert not stab.is_read_only


# ---------------------------------------------------------------------------
# 4. Exponential backoff schedule
# ---------------------------------------------------------------------------

def test_backoff_first_attempt_is_15_min():
    stab = JohansenStabilizer(fail_streak=1)
    t0 = 5000.0
    with patch("quant_system.strategies.stabilizer.time.monotonic", return_value=t0):
        stab._history.append(False)
        stab._update_state(False)

    assert stab.is_read_only
    expected = t0 + _BACKOFF_SCHEDULE_MINUTES[0] * 60.0
    assert stab._next_eval_at == pytest.approx(expected)


def test_backoff_second_attempt_is_30_min():
    stab = JohansenStabilizer(fail_streak=1)
    t0 = 5000.0
    # First failure → 15 min
    with patch("quant_system.strategies.stabilizer.time.monotonic", return_value=t0):
        stab._history.append(False)
        stab._update_state(False)

    # Second failure (re-eval failed) → 30 min
    t1 = t0 + _BACKOFF_SCHEDULE_MINUTES[0] * 60.0 + 1  # just past backoff
    with patch("quant_system.strategies.stabilizer.time.monotonic", return_value=t1):
        stab._history.append(False)
        stab._update_state(False)

    expected = t1 + _BACKOFF_SCHEDULE_MINUTES[1] * 60.0
    assert stab._next_eval_at == pytest.approx(expected)


def test_backoff_third_attempt_is_60_min():
    stab = JohansenStabilizer(fail_streak=1)
    t0 = 5000.0
    with patch("quant_system.strategies.stabilizer.time.monotonic", return_value=t0):
        stab._history.append(False)
        stab._update_state(False)

    t1 = t0 + 1
    with patch("quant_system.strategies.stabilizer.time.monotonic", return_value=t1):
        stab._history.append(False)
        stab._update_state(False)

    t2 = t1 + 1
    with patch("quant_system.strategies.stabilizer.time.monotonic", return_value=t2):
        stab._history.append(False)
        stab._update_state(False)

    expected = t2 + _BACKOFF_SCHEDULE_MINUTES[2] * 60.0
    assert stab._next_eval_at == pytest.approx(expected)


# ---------------------------------------------------------------------------
# 5. WARN logged only once per Read-Only period
# ---------------------------------------------------------------------------

def test_warn_logged_only_once_per_readonly_period(caplog):
    stab = JohansenStabilizer(fail_streak=1)
    t0 = 1000.0

    with caplog.at_level(logging.WARNING, logger="quant_system.strategies.stabilizer"):
        with patch("quant_system.strategies.stabilizer.time.monotonic", return_value=t0):
            for _ in range(5):
                stab._history.append(False)
                stab._update_state(False)

    warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert len(warnings) == 1
    assert "Read-Only" in warnings[0].message
