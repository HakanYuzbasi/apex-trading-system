from __future__ import annotations

from datetime import date, timedelta

import pytest

from quant_system.events import CorporateAction, FundingSnapshot, GreeksSnapshot, QuoteTick, TradeTick, utc_now


def test_quote_tick_accepts_valid_l1_payload() -> None:
    now = utc_now()
    event = QuoteTick(
        instrument_id="AAPL",
        exchange_ts=now,
        received_ts=now,
        processed_ts=now,
        sequence_id=1,
        source="feed.test",
        bid=199.95,
        ask=200.05,
        bid_size=100.0,
        ask_size=125.0,
    )

    assert event.event_type == "quote_tick"


def test_trade_tick_requires_positive_print_values() -> None:
    now = utc_now()

    with pytest.raises(ValueError, match="last_size must be positive"):
        TradeTick(
            instrument_id="AAPL",
            exchange_ts=now,
            received_ts=now,
            processed_ts=now,
            sequence_id=2,
            source="feed.test",
            last_price=200.0,
            last_size=0.0,
            aggressor_side="buy",
            trade_id="t-1",
        )


def test_greeks_snapshot_validates_delta_range() -> None:
    now = utc_now()

    with pytest.raises(ValueError, match="delta must be between -1.0 and 1.0"):
        GreeksSnapshot(
            instrument_id="AAPL-20261218-200-C",
            exchange_ts=now,
            received_ts=now,
            processed_ts=now,
            sequence_id=3,
            source="options.test",
            iv=0.25,
            delta=1.2,
            gamma=0.01,
            theta=-0.04,
            vega=0.12,
            rho=0.02,
            underlying_price=200.0,
        )


def test_funding_snapshot_requires_future_timezone_aware_timestamp() -> None:
    now = utc_now()
    event = FundingSnapshot(
        instrument_id="BTC-PERP",
        exchange_ts=now,
        received_ts=now,
        processed_ts=now,
        sequence_id=4,
        source="funding.test",
        funding_rate=0.0001,
        next_funding_ts=now + timedelta(hours=8),
    )

    assert event.event_type == "funding_snapshot"


def test_corporate_action_requires_action_payload() -> None:
    now = utc_now()

    with pytest.raises(ValueError, match="at least one of split_ratio or dividend_cash"):
        CorporateAction(
            instrument_id="MSFT",
            exchange_ts=now,
            received_ts=now,
            processed_ts=now,
            sequence_id=5,
            source="corp-actions.test",
            effective_date=date(2026, 4, 6),
        )
