"""APEX startup orchestrator.

This entrypoint enforces startup hardening checks, then delegates runtime execution
into the extracted execution loop module.
"""

from __future__ import annotations

import asyncio
import logging
import os
from typing import Optional
from pathlib import Path

try:
    import fcntl  # Unix-only; available on macOS/Linux
except ImportError:  # pragma: no cover
    fcntl = None  # type: ignore[assignment]

from config import ApexConfig, assert_live_trading_confirmation
from scripts.check_secrets import validate_secrets

logger = logging.getLogger(__name__)
_LOCK_FH = None


def _acquire_singleton_lock() -> None:
    """Prevent multiple concurrent main.py runtime instances."""
    global _LOCK_FH
    lock_dir = Path(ApexConfig.DATA_DIR)
    lock_dir.mkdir(parents=True, exist_ok=True)
    lock_path = lock_dir / "apex_engine.lock"
    _LOCK_FH = open(lock_path, "a+", encoding="utf-8")
    _LOCK_FH.seek(0)
    if fcntl is not None:
        try:
            fcntl.flock(_LOCK_FH.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            raise RuntimeError(
                "APEX runtime already running (lock busy). Stop existing main.py processes first."
            )
    _LOCK_FH.truncate(0)
    _LOCK_FH.write(str(os.getpid()))
    _LOCK_FH.flush()


def run_startup_guards() -> None:
    """Run hard startup guards before any trading runtime initialization."""
    validate_secrets()
    assert_live_trading_confirmation()


async def main() -> None:
    """Main async entrypoint for the APEX trading runtime."""
    run_startup_guards()
    
    from core.orchestrator import execution_manager
    await execution_manager.start()
    
    try:
        # Keep the main process alive while orchestrator manages background tasks
        while True:
            await asyncio.sleep(3600)
    except asyncio.CancelledError:
        logger.info("Main execution loop cancelled, gracefully shutting down orchestrator...")
    finally:
        await execution_manager.stop()


def _run() -> Optional[int]:
    """Synchronous entrypoint wrapper for CLI execution."""
    try:
        _acquire_singleton_lock()
        asyncio.run(main())
        return 0
    except KeyboardInterrupt:
        print("\n\n👋 Goodbye!")
        return 0
    except Exception:
        logger.exception("Fatal startup/runtime failure")
        return 1


if __name__ == "__main__":
    raise SystemExit(_run())
