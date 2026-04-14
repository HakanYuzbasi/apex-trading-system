from __future__ import annotations

import asyncio
import inspect
from collections import defaultdict
from dataclasses import dataclass
from typing import Awaitable, Callable, DefaultDict, TypeAlias

from quant_system.events.base import BaseEvent

EventHandler: TypeAlias = Callable[[BaseEvent], None]
AsyncEventHandler: TypeAlias = Callable[[BaseEvent], Awaitable[None]]
AnyEventHandler: TypeAlias = EventHandler | AsyncEventHandler


@dataclass(frozen=True, slots=True)
class Subscription:
    token: str
    event_type: str
    handler_name: str
    is_async: bool
    sequence: int


@dataclass(frozen=True, slots=True)
class DispatchResult:
    event_id: str
    event_type: str
    subscriber_count: int
    sync_handler_count: int
    async_handler_count: int


@dataclass(frozen=True, slots=True)
class _RegisteredHandler:
    token: str
    event_type: str
    handler: AnyEventHandler
    is_async: bool
    sequence: int


class InMemoryEventBus:
    """
    In-memory event dispatcher for both deterministic replay and live fan-out.

    Design rules:
    - `publish()` is fully synchronous and deterministic. It rejects async handlers.
    - `publish_async()` preserves deterministic dispatch order while allowing async
      handlers to run concurrently after they have been scheduled in registration order.
    - Subscribers can bind to a concrete `event_type` or to the wildcard `*`.
    """

    def __init__(self, max_log_size: int = 1000) -> None:
        self._subscriptions: DefaultDict[str, list[_RegisteredHandler]] = defaultdict(list)
        self._dispatch_sequence = 0
        self._subscription_sequence = 0
        from collections import deque
        self._event_log: deque[BaseEvent] = deque(maxlen=max_log_size)

    def subscribe(
        self,
        event_type: str,
        handler: AnyEventHandler,
        *,
        is_async: bool | None = None,
    ) -> Subscription:
        if not event_type.strip():
            raise ValueError("event_type must be a non-empty string")

        inferred_is_async = inspect.iscoroutinefunction(handler)
        run_async = inferred_is_async if is_async is None else is_async
        if run_async and not inferred_is_async:
            raise TypeError("is_async=True requires an async handler")

        token = f"sub-{self._subscription_sequence}"
        subscription = _RegisteredHandler(
            token=token,
            event_type=event_type,
            handler=handler,
            is_async=run_async,
            sequence=self._subscription_sequence,
        )
        self._subscription_sequence += 1
        self._subscriptions[event_type].append(subscription)
        return Subscription(
            token=subscription.token,
            event_type=subscription.event_type,
            handler_name=self._handler_name(handler),
            is_async=subscription.is_async,
            sequence=subscription.sequence,
        )

    def unsubscribe(self, token: str) -> bool:
        for event_type, handlers in self._subscriptions.items():
            for idx, registered in enumerate(handlers):
                if registered.token == token:
                    del handlers[idx]
                    if not handlers:
                        del self._subscriptions[event_type]
                    return True
        return False

    def publish(self, event: BaseEvent) -> DispatchResult:
        handlers = self._resolve_handlers(event.event_type)
        async_handlers = [handler for handler in handlers if handler.is_async]
        if async_handlers:
            raise RuntimeError(
                "Synchronous publish cannot dispatch async subscribers. "
                "Use publish_async() for live or concurrent handlers."
            )

        self._record_event(event)
        for registered in handlers:
            sync_handler = registered.handler
            sync_handler(event)

        return self._build_result(event, handlers)

    async def publish_async(self, event: BaseEvent) -> DispatchResult:
        handlers = self._resolve_handlers(event.event_type)
        self._record_event(event)

        pending_tasks: list[asyncio.Task[None]] = []
        for registered in handlers:
            if registered.is_async:
                coroutine = registered.handler(event)
                pending_tasks.append(asyncio.create_task(coroutine))
            else:
                sync_handler = registered.handler
                sync_handler(event)

        if pending_tasks:
            results = await asyncio.gather(*pending_tasks, return_exceptions=True)
            for result in results:
                if isinstance(result, Exception):
                    import logging
                    logging.getLogger(__name__).error(
                        "Async handler raised an exception (event_type=%s): %s",
                        event.event_type, result, exc_info=result,
                    )

        return self._build_result(event, handlers)

    def subscriptions_for(self, event_type: str) -> list[Subscription]:
        handlers = self._resolve_handlers(event_type)
        return [
            Subscription(
                token=handler.token,
                event_type=handler.event_type,
                handler_name=self._handler_name(handler.handler),
                is_async=handler.is_async,
                sequence=handler.sequence,
            )
            for handler in handlers
        ]

    @property
    def published_events(self) -> tuple[BaseEvent, ...]:
        return tuple(self._event_log)

    def _resolve_handlers(self, event_type: str) -> list[_RegisteredHandler]:
        exact = self._subscriptions.get(event_type, [])
        wildcard = self._subscriptions.get("*", [])
        merged = exact + wildcard
        seen_tokens: set[str] = set()
        ordered: list[_RegisteredHandler] = []
        for handler in sorted(merged, key=lambda entry: entry.sequence):
            if handler.token in seen_tokens:
                continue
            seen_tokens.add(handler.token)
            ordered.append(handler)
        return ordered

    def _record_event(self, event: BaseEvent) -> None:
        self._dispatch_sequence += 1
        self._event_log.append(event)

    @staticmethod
    def _handler_name(handler: AnyEventHandler) -> str:
        return getattr(handler, "__name__", handler.__class__.__name__)

    @staticmethod
    def _build_result(event: BaseEvent, handlers: list[_RegisteredHandler]) -> DispatchResult:
        async_count = sum(1 for handler in handlers if handler.is_async)
        sync_count = len(handlers) - async_count
        return DispatchResult(
            event_id=event.event_id,
            event_type=event.event_type,
            subscriber_count=len(handlers),
            sync_handler_count=sync_count,
            async_handler_count=async_count,
        )
