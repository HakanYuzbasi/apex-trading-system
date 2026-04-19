"""
execution/shutdown_manager.py — SIGTERM-driven graceful shutdown
=================================================================

Container orchestrators (Kubernetes, systemd, Docker) send ``SIGTERM`` when
they want a process to stop and then ``SIGKILL`` after a grace period. A
trading system that ignores ``SIGTERM`` will either leave positions exposed
(if killed mid-cycle) or corrupt persisted state. This module installs a
signal-driven shutdown controller that:

1. Flips a shared ``asyncio.Event`` so the main loop exits its next iteration.
2. Optionally flattens every open position via the registered broker
   connectors (controlled by ``SHUTDOWN_FLATTEN_POSITIONS``).
3. Persists state and disconnects brokers within
   ``SHUTDOWN_TIMEOUT_SECONDS``.

Usage:
    manager = ShutdownManager()
    manager.install_signal_handlers(asyncio.get_event_loop())
    manager.register_flattener(broker.flatten_all)
    manager.register_cleanup(broker.disconnect)
    ...
    await manager.stop_event.wait()
    await manager.run_shutdown()
"""
from __future__ import annotations

import asyncio
import logging
import signal
from typing import Awaitable, Callable, List, Optional, Union

from config import ApexConfig

logger = logging.getLogger(__name__)


# Callables can be sync or async; both forms are honoured.
_SyncOrAsync = Callable[[], Union[None, Awaitable[None]]]


class ShutdownManager:
    """
    Coordinator for signal-driven graceful shutdown.

    Attributes:
        stop_event: Set once a shutdown signal is received. Main loops should
            break when this event fires.
        timeout_seconds: Hard ceiling for the entire shutdown sequence.
        flatten_positions: When ``True``, every registered flattener runs
            before cleanups.
    """

    def __init__(
        self,
        timeout_seconds: Optional[float] = None,
        flatten_positions: Optional[bool] = None,
    ) -> None:
        self.stop_event: asyncio.Event = asyncio.Event()
        self.timeout_seconds: float = (
            float(timeout_seconds)
            if timeout_seconds is not None
            else float(getattr(ApexConfig, "SHUTDOWN_TIMEOUT_SECONDS", 30.0))
        )
        self.flatten_positions: bool = (
            bool(flatten_positions)
            if flatten_positions is not None
            else bool(getattr(ApexConfig, "SHUTDOWN_FLATTEN_POSITIONS", True))
        )
        self._flatteners: List[_SyncOrAsync] = []
        self._cleanups: List[_SyncOrAsync] = []
        self._installed: bool = False
        self._triggered: bool = False

    # ── Registration ─────────────────────────────────────────────────────────

    def register_flattener(self, fn: _SyncOrAsync) -> None:
        """
        Register a callable responsible for flattening positions (e.g.
        ``close_all_positions``). Called before cleanups when
        ``flatten_positions`` is true.

        Args:
            fn: Zero-arg sync or async callable.

        Raises:
            TypeError: If ``fn`` is not callable.
        """
        if not callable(fn):
            raise TypeError(f"flattener must be callable, got {type(fn).__name__}")
        self._flatteners.append(fn)

    def register_cleanup(self, fn: _SyncOrAsync) -> None:
        """
        Register a cleanup callable (broker disconnect, state save, etc.)
        that runs unconditionally after any flatten phase.

        Args:
            fn: Zero-arg sync or async callable.

        Raises:
            TypeError: If ``fn`` is not callable.
        """
        if not callable(fn):
            raise TypeError(f"cleanup must be callable, got {type(fn).__name__}")
        self._cleanups.append(fn)

    # ── Signal handling ──────────────────────────────────────────────────────

    def install_signal_handlers(
        self, loop: Optional[asyncio.AbstractEventLoop] = None
    ) -> None:
        """
        Register ``SIGTERM`` / ``SIGINT`` handlers on the given event loop. The
        handlers merely flip :attr:`stop_event`; the caller is responsible for
        invoking :meth:`run_shutdown` once the main loop observes the event.

        Args:
            loop: Event loop to install handlers on. Defaults to the running
                loop when ``None``.

        Notes:
            On platforms where :meth:`loop.add_signal_handler` is not
            implemented (e.g. Windows), we fall back to ``signal.signal`` and
            schedule :meth:`stop_event.set` thread-safely.
        """
        if self._installed:
            return
        loop = loop or asyncio.get_event_loop()
        for sig in (signal.SIGTERM, signal.SIGINT):
            try:
                loop.add_signal_handler(sig, self._on_signal, sig.name)
            except NotImplementedError:
                # Windows — can't use add_signal_handler
                signal.signal(sig, lambda signum, _frame, _s=sig.name: self._on_signal(_s))
            except RuntimeError as exc:
                logger.warning(
                    "ShutdownManager: could not install %s handler (%s)",
                    sig.name, exc,
                )
        self._installed = True

    def _on_signal(self, sig_name: str) -> None:
        """Idempotent signal-reception callback."""
        if self._triggered:
            logger.warning(
                "ShutdownManager: received %s while already shutting down — "
                "ignoring duplicate", sig_name,
            )
            return
        self._triggered = True
        logger.warning(
            "ShutdownManager: %s received — initiating graceful shutdown (timeout=%.1fs)",
            sig_name, self.timeout_seconds,
        )
        self.stop_event.set()

    # ── Run shutdown ────────────────────────────────────────────────────────

    async def run_shutdown(self) -> None:
        """
        Execute registered flatteners (when enabled) then cleanups, guarded
        by :attr:`timeout_seconds`. Exceptions in any callback are logged and
        *do not* abort the remainder of the sequence.
        """
        try:
            await asyncio.wait_for(self._shutdown_sequence(), timeout=self.timeout_seconds)
        except asyncio.TimeoutError:
            logger.error(
                "ShutdownManager: shutdown exceeded %.1fs budget — "
                "remaining cleanups skipped",
                self.timeout_seconds,
            )

    async def _shutdown_sequence(self) -> None:
        if self.flatten_positions and self._flatteners:
            logger.info(
                "ShutdownManager: flattening positions across %d registered broker(s)",
                len(self._flatteners),
            )
            for fn in list(self._flatteners):
                await self._invoke("flatten", fn)

        for fn in list(self._cleanups):
            await self._invoke("cleanup", fn)
        logger.info("ShutdownManager: shutdown sequence complete")

    @staticmethod
    async def _invoke(tag: str, fn: _SyncOrAsync) -> None:
        try:
            result = fn()
            if asyncio.iscoroutine(result):
                await result
        except Exception as exc:
            logger.error(
                "ShutdownManager: %s callback %s raised: %s",
                tag, getattr(fn, "__name__", repr(fn)), exc,
                exc_info=True,
            )


_GLOBAL_MANAGER: Optional[ShutdownManager] = None


def get_shutdown_manager() -> ShutdownManager:
    """Return the process-wide :class:`ShutdownManager` singleton."""
    global _GLOBAL_MANAGER
    if _GLOBAL_MANAGER is None:
        _GLOBAL_MANAGER = ShutdownManager()
    return _GLOBAL_MANAGER
