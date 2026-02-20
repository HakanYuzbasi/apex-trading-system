"""Tests for dashboard equity response shape."""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from api.auth import User
from services.broker import router as broker_router


@pytest.mark.asyncio
async def test_portfolio_balance_breakdown_shape(monkeypatch: pytest.MonkeyPatch) -> None:
    async def fake_snapshot(user_id: str):
        return {
            "tenant_id": user_id,
            "total_equity": 3000.0,
            "as_of": datetime.now(timezone.utc).isoformat(),
            "breakdown": [
                {
                    "value": 1000.0,
                    "broker": "alpaca",
                    "stale": False,
                    "as_of": datetime.now(timezone.utc).isoformat(),
                    "source": "A",
                    "source_id": "a1",
                },
                {
                    "value": 2000.0,
                    "broker": "ibkr",
                    "stale": True,
                    "as_of": datetime.now(timezone.utc).isoformat(),
                    "source": "I",
                    "source_id": "i1",
                },
            ],
        }

    monkeypatch.setattr(broker_router.broker_service, "get_tenant_equity_snapshot", fake_snapshot)

    payload = await broker_router.get_portfolio_balance(user=User(user_id="tenant-1", username="u"))

    assert payload["total_equity"] == 3000.0
    assert len(payload["breakdown"]) == 2
    for row in payload["breakdown"]:
        assert set(["value", "broker", "stale", "as_of"]).issubset(row.keys())
