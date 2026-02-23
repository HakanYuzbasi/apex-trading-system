"""
core/orchestrator.py - Multi-Tenant Execution Manager

Spawns and manages isolated Execution Loops (ApexTradingSystem) for each active tenant.
Monitors the BrokerService for newly mapped tenant connections and gracefully provisions 
or shuts down loops dynamically.
"""

import asyncio
import logging
from typing import Dict, Optional
from datetime import datetime

from core.execution_loop import ApexTradingSystem
from services.broker.service import broker_service

logger = logging.getLogger(__name__)

class TenantProcess:
    """Tracks the state and task of a single tenant's trading loop."""
    def __init__(self, tenant_id: str, system: ApexTradingSystem, task: asyncio.Task):
        self.tenant_id = tenant_id
        self.system = system
        self.task = task
        self.started_at = datetime.now()
        
    def stop(self):
        """Signal the system to stop and wait for loop termination."""
        if self.system:
            self.system.is_running = False
        if self.task and not self.task.done():
            self.task.cancel()

class ExecutionManager:
    """
    God-level orchestrator that scales `ApexTradingSystem` execution loops dynamically
    based on active broker connections in the database.
    """
    def __init__(self):
        self.tenants: Dict[str, TenantProcess] = {}
        self._running = False
        self._monitor_task: Optional[asyncio.Task] = None
        
        # Backoff tracking: tenant_id -> (crash_count, next_allowed_spawn_time)
        self._crash_backoff: Dict[str, tuple[int, datetime]] = {}
        
    async def start(self):
        """Start the orchestrator monitoring."""
        if self._running:
            return
            
        self._running = True
        logger.info("🪐 Starting ExecutionManager Orchestrator...")
        
        # The monitor loop will lazily load credentials for active tenants using list_tenant_ids()
        
        # Start background monitor
        self._monitor_task = asyncio.create_task(self._monitor_tenants_loop())
        
    async def stop(self):
        """Shutdown all tenant loops gracefully."""
        self._running = False
        if self._monitor_task:
            self._monitor_task.cancel()
            
        logger.info("🪐 ExecutionManager shutting down. Stopping all tenant loops...")
        
        # Stop all running loops
        for tenant_id, process in list(self.tenants.items()):
            logger.info(f"   Stopping loop for tenant {tenant_id}...")
            process.stop()
            
        # Wait for all tasks to cleanly exit
        tasks = [p.task for p in self.tenants.values() if p.task and not p.task.done()]
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
            
        self.tenants.clear()
        logger.info("🪐 All tenant loops gracefully terminated.")
        
    async def _monitor_tenants_loop(self):
        """
        Background task that periodically pulls active tenants from the broker_service 
        and reconciles running execution loops.
        """
        import datetime as dt

        while self._running:
            try:
                now = dt.datetime.now()
                # 1. Fetch the latest active subset of tenants from the database/service
                active_tenant_ids = set(await broker_service.list_tenant_ids())
                
                # A tenant is active if they have at least one active connection
                for tenant_id in active_tenant_ids:
                    await broker_service._load_user(tenant_id)
                
                # 2. Check for missing tenants (Spawn)
                for tenant_id in active_tenant_ids:
                    if tenant_id not in self.tenants:
                        # Check exponential backoff
                        if tenant_id in self._crash_backoff:
                            _, next_allowed = self._crash_backoff[tenant_id]
                            if now < next_allowed:
                                continue  # Still in backoff penalty box
                                
                        self._spawn_tenant_loop(tenant_id)
                        
                # 3. Check for stale tenants (Kill)
                for tenant_id in list(self.tenants.keys()):
                    if tenant_id not in active_tenant_ids:
                        self._kill_tenant_loop(tenant_id)
                        
                # 4. Check for crashed tenant tasks (Restart)
                for tenant_id, process in list(self.tenants.items()):
                    if process.task.done():
                        exc = process.task.exception()
                        uptime = (now - process.started_at).total_seconds()
                        
                        crash_count, _ = self._crash_backoff.get(tenant_id, (0, now))
                        # Reset crash count if it ran successfully for a long time
                        if uptime > 3600:
                            crash_count = 0
                            
                        crash_count += 1
                        # Backoff: 60s, 120s, 240s... max 1 hour
                        backoff_seconds = min(60 * (2 ** (crash_count - 1)), 3600)
                        next_allowed = now + dt.timedelta(seconds=backoff_seconds)
                        
                        self._crash_backoff[tenant_id] = (crash_count, next_allowed)
                        
                        if exc:
                            logger.error(f"🚨 Tenant loop {tenant_id} crashed with exception: {exc}")
                        else:
                            logger.warning(f"⚠️  Tenant loop {tenant_id} exited unexpectedly.")
                            
                        logger.warning(f"   Respawning scheduled in {backoff_seconds}s (Attempt #{crash_count})")
                            
                        # Cleanup and self-heal by removing it; it'll respawn next tick
                        del self.tenants[tenant_id]
                        
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"🪐 ExecutionManager monitor error: {e}")
                
            await asyncio.sleep(10)  # Reconcile every 10 seconds
            
    def _spawn_tenant_loop(self, tenant_id: str):
        """Spawn a new execution loop for a tenant."""
        logger.info(f"🪐 Spawning new execution loop for tenant {tenant_id}...")
        try:
            # Instantiate an isolated ApexTradingSystem for this tenant
            system = ApexTradingSystem(tenant_id=tenant_id, broker_service=broker_service)
            
            # Start the loop dynamically
            loop_task = asyncio.create_task(system.run())
            
            self.tenants[tenant_id] = TenantProcess(tenant_id, system, loop_task)
            logger.info(f"✅ Loop spawned successfully for {tenant_id}")
            
        except Exception as e:
            logger.error(f"❌ Failed to spawn loop for tenant {tenant_id}: {e}")
            
    def _kill_tenant_loop(self, tenant_id: str):
        """Gracefully kill an existing execution loop."""
        logger.info(f"🪐 Gracefully killing execution loop for tenant {tenant_id}...")
        if tenant_id in self.tenants:
            process = self.tenants.pop(tenant_id)
            process.stop()
            
        # Also clean up any backoffs if they are intentionally killed
        if tenant_id in self._crash_backoff:
            del self._crash_backoff[tenant_id]

# Global singleton orchestrator
execution_manager = ExecutionManager()
