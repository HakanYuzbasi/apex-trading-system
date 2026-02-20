"""Broker adapter dispatching extracted from trading entrypoint."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any, Dict, Optional

from core.exceptions import ApexBrokerError


@dataclass
class BrokerDispatchResult:
    """Result for broker adapter execution."""

    broker: str
    success: bool
    payload: Dict[str, Any]


class BrokerDispatch:
    """Route broker operations to IBKR/Alpaca adapters with consistent errors."""

    def __init__(self, ibkr: Optional[Any], alpaca: Optional[Any]) -> None:
        self._ibkr = ibkr
        self._alpaca = alpaca

    async def place_order(self, broker: str, **order_kwargs: Any) -> BrokerDispatchResult:
        """Place an order on the selected broker adapter."""
        if broker == "ibkr":
            if self._ibkr is None:
                raise ApexBrokerError(
                    code="BROKER_NOT_CONFIGURED",
                    message="IBKR adapter is not configured",
                    context={"broker": broker},
                )
            result = await self._execute(self._ibkr.place_order, **order_kwargs)
            return BrokerDispatchResult(broker="ibkr", success=True, payload={"result": result})

        if broker == "alpaca":
            if self._alpaca is None:
                raise ApexBrokerError(
                    code="BROKER_NOT_CONFIGURED",
                    message="Alpaca adapter is not configured",
                    context={"broker": broker},
                )
            result = await self._execute(self._alpaca.place_order, **order_kwargs)
            return BrokerDispatchResult(broker="alpaca", success=True, payload={"result": result})

        raise ApexBrokerError(
            code="BROKER_UNKNOWN",
            message=f"Unsupported broker dispatch target: {broker}",
            context={"broker": broker},
        )

    @staticmethod
    async def _execute(fn: Any, **kwargs: Any) -> Any:
        """Execute adapter call in async-safe manner."""
        if asyncio.iscoroutinefunction(fn):
            return await fn(**kwargs)
        return await asyncio.to_thread(fn, **kwargs)
