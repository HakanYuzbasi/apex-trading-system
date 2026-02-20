"""APEX startup orchestrator.

This entrypoint enforces startup hardening checks, then delegates runtime execution
into the extracted execution loop module.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Optional

from config import ApexConfig, assert_live_trading_confirmation
from core.execution_loop import ApexTradingSystem
from scripts.check_secrets import validate_secrets

logger = logging.getLogger(__name__)


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
