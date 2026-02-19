import pytest
import asyncio
from unittest.mock import MagicMock, patch
from services.broker.service import BrokerService
from models.broker import BrokerType

@pytest.mark.asyncio
async def test_encryption():
    service = BrokerService()
    creds = {"api_key": "test", "secret": "test"}
    encrypted = service._encrypt_credentials(creds)
    decrypted = service._decrypt_credentials(encrypted)
    assert decrypted == creds

@pytest.mark.asyncio
async def test_create_connection_alpaca():
    service = BrokerService()
    
    # Mock validation
    with patch.object(service, 'validate_credentials', return_value=True) as mock_validate:
        conn = await service.create_connection(
            user_id="user123",
            broker_type=BrokerType.ALPACA,
            name="Test Alpaca",
            credentials={"api_key": "PROCESS", "secret_key": "SECRET"},
            environment="paper"
        )
        
        assert conn.user_id == "user123"
        assert conn.broker_type == BrokerType.ALPACA
        assert conn.environment == "paper"
        assert "data" in conn.credentials
        
        # Verify stored
        stored = await service.get_connection(conn.id)
        assert stored == conn

@pytest.mark.asyncio
async def test_ibkr_validation_failure():
    service = BrokerService()
    # Mock connect_ibkr to raise error
    with patch.object(service, 'connect_ibkr', side_effect=ValueError("Connection refused")):
        with pytest.raises(ValueError):
            await service.validate_credentials(
                BrokerType.IBKR, 
                {"host": "127.0.0.1", "port": 4001, "client_id": 1}
            )

if __name__ == "__main__":
    # Manually run if executed as script
    async def main():
        try:
            await test_encryption()
            print("Encryption test passed")
            await test_create_connection_alpaca()
            print("Create connection test passed")
            await test_ibkr_validation_failure()
            print("Validation failure test passed")
        except Exception as e:
            print(f"Tests failed: {e}")
            raise
    
    asyncio.run(main())
