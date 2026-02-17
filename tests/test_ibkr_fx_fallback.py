from unittest.mock import AsyncMock, MagicMock

import pytest

from execution.ibkr_connector import IBKRConnector


@pytest.mark.asyncio
async def test_get_contract_fx_fallback_qualifies_when_primary_fails():
    connector = IBKRConnector()
    connector.offline_mode = False
    connector.ib = MagicMock()
    connector.ib.isConnected.return_value = True

    qualified_contract = MagicMock()
    qualify_calls = []

    async def fake_qualify(contract):
        qualify_calls.append(contract)
        if len(qualify_calls) == 1:
            return []
        return [qualified_contract]

    connector.ib.qualifyContractsAsync = AsyncMock(side_effect=fake_qualify)
    connector._throttle_ibkr = AsyncMock(return_value=None)

    contract = await connector.get_contract("FX:EUR/USD")

    assert contract is qualified_contract
    assert len(qualify_calls) >= 2
