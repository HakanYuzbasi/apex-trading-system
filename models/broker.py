from enum import Enum
from typing import Optional, Dict, Any, Literal
from pydantic import BaseModel, Field, validator
from datetime import datetime
import uuid

class BrokerType(str, Enum):
    IBKR = "ibkr"
    ALPACA = "alpaca"

class BrokerConnection(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    user_id: str
    broker_type: BrokerType
    name: str
    environment: Literal["paper", "live"] = "paper"
    client_id: Optional[int] = None  # Arbitrary int (1-32) for IBKR, None for Alpaca
    credentials: Dict[str, Any]  # This will be encrypted at rest in the DB layer
    is_active: bool = True
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)

    @validator("client_id")
    def validate_client_id(cls, v, values):
        if values.get("broker_type") == BrokerType.IBKR:
            if v is None:
                raise ValueError("client_id is required for IBKR")
            if not (0 <= v <= 999999): # IBKR client IDs are typically positive integers
                raise ValueError("client_id must be a positive integer")
        return v
