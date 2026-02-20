import pytest
import asyncio
import time
from execution.ibkr_lease_manager import IBKRLeaseManager

@pytest.fixture
async def lease_manager(monkeypatch):
    """Provides a fresh lease manager forcing local non-redis mode for deterministic tests."""
    from services.common import redis_client
    # Mock get_redis to return None so we test the local dict fallback logic predictably
    monkeypatch.setattr(redis_client, "get_redis", _mock_get_redis)
    
    IBKRLeaseManager._instance = None
    manager = IBKRLeaseManager()
    
    yield manager
    manager._local_leases.clear()

async def _mock_get_redis():
    return None

@pytest.mark.asyncio
async def test_lease_allocation_local(lease_manager):
    client_id = await lease_manager.allocate(preferred_id=105, ttl=1.0)
    assert client_id == 105
    assert 105 in lease_manager._local_leases
    
@pytest.mark.asyncio
async def test_lease_collision_fallback(lease_manager):
    await lease_manager.allocate(preferred_id=105, ttl=10.0)
    
    # Second request for 105 should fall back to next available (100)
    second_id = await lease_manager.allocate(preferred_id=105, ttl=10.0)
    assert second_id != 105
    assert second_id >= lease_manager.min_id
    
@pytest.mark.asyncio
async def test_lease_expiration_and_reallocation(lease_manager):
    # Allocate with 0.1s TTL
    cid = await lease_manager.allocate(preferred_id=999, ttl=0.1)
    assert cid == 999
    
    # Immediately try to allocate 999, should fail/fallback
    cid2 = await lease_manager.allocate(preferred_id=999, ttl=10.0)
    assert cid2 != 999
    
    # Wait for expiration
    await asyncio.sleep(0.15)
    
    # Should be able to get 999 again because _prune_local clears it
    cid3 = await lease_manager.allocate(preferred_id=999, ttl=10.0)
    assert cid3 == 999

@pytest.mark.asyncio
async def test_lease_heartbeat(lease_manager):
    cid = await lease_manager.allocate(preferred_id=888, ttl=0.1)
    
    # Heartbeat to extend ttl
    success = await lease_manager.heartbeat(cid, ttl=0.5)
    assert success is True
    
    await asyncio.sleep(0.15)
    # Lease should still be active because of heartbeat
    cid2 = await lease_manager.allocate(preferred_id=888, ttl=10.0)
    assert cid2 != 888

@pytest.mark.asyncio
async def test_lease_release(lease_manager):
    cid = await lease_manager.allocate(preferred_id=777, ttl=10.0)
    
    await lease_manager.release(cid)
    
    # Should be available immediately
    cid2 = await lease_manager.allocate(preferred_id=777, ttl=10.0)
    assert cid2 == 777
