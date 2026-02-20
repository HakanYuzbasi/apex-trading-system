import asyncio
import logging
import time
from typing import Dict, Optional
from services.common.redis_client import get_redis

logger = logging.getLogger(__name__)

class IBKRLeaseManager:
    """
    Manages IBKR client_id allocations to prevent collisions.
    Supports local asyncio locks and Redis-distributed TTL leasing
    so dropped processes release their IDs gracefully.
    """
    _instance = None

    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            cls._instance = super(IBKRLeaseManager, cls).__new__(cls)
            cls._instance._init()
        return cls._instance

    def _init(self):
        self._lock = asyncio.Lock()
        self._local_leases: Dict[int, float] = {}
        self.default_ttl = 300.0  # 5 minutes
        self.min_id = 100
        self.max_id = 999
        self.redis_prefix = "ibkr:lease:"
        
    async def allocate(self, preferred_id: Optional[int] = None, ttl: Optional[float] = None) -> int:
        """Allocate a client ID with TTL."""
        lease_ttl = float(ttl if ttl is not None else self.default_ttl)
        redis = await get_redis()
        
        async with self._lock:
            now = time.time()
            self._prune_local(now)
            
            # Check preferred
            if preferred_id is not None and preferred_id >= self.min_id and preferred_id <= self.max_id:
                acquired = await self._try_acquire(redis, preferred_id, lease_ttl, now)
                if acquired:
                    return preferred_id
                    
            # Find any available
            for client_id in range(self.min_id, self.max_id + 1):
                acquired = await self._try_acquire(redis, client_id, lease_ttl, now)
                if acquired:
                    return client_id
                    
            raise RuntimeError("No available IBKR client IDs to lease")
            
    async def heartbeat(self, client_id: int, ttl: Optional[float] = None) -> bool:
        """Renew a lease."""
        lease_ttl = float(ttl if ttl is not None else self.default_ttl)
        redis = await get_redis()
        
        async with self._lock:
            if redis:
                key = f"{self.redis_prefix}{client_id}"
                # Extend if it exists
                if await redis.exists(key):
                    await redis.pexpire(key, int(lease_ttl * 1000))
                    return True
                return False
            else:
                if client_id in self._local_leases:
                    self._local_leases[client_id] = time.time() + lease_ttl
                    return True
                return False
                
    async def release(self, client_id: int):
        """Release a client ID."""
        redis = await get_redis()
        async with self._lock:
            if redis:
                await redis.delete(f"{self.redis_prefix}{client_id}")
            if client_id in self._local_leases:
                del self._local_leases[client_id]
            logger.debug(f"Released IBKR client_id {client_id}")
            
    async def _try_acquire(self, redis, client_id: int, lease_ttl: float, now: float) -> bool:
        if redis:
            key = f"{self.redis_prefix}{client_id}"
            # SETNX equivalent logic with millisecond precision
            acquired = await redis.set(key, "locked", px=int(lease_ttl * 1000), nx=True)
            return bool(acquired)
        else:
            if client_id not in self._local_leases:
                self._local_leases[client_id] = now + lease_ttl
                return True
            return False
            
    def _prune_local(self, now: float):
        expired = [cid for cid, xt in self._local_leases.items() if now > xt]
        for cid in expired:
            del self._local_leases[cid]

lease_manager = IBKRLeaseManager()
