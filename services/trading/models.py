
import uuid
import enum
from datetime import datetime
from sqlalchemy import (
    Column, String, Boolean, DateTime, Float, ForeignKey, Enum, JSON, Numeric
)
from sqlalchemy.orm import relationship
from services.common.db import Base

# Enums
class OrderSide(str, enum.Enum):
    BUY = "buy"
    SELL = "sell"

class OrderType(str, enum.Enum):
    MARKET = "market"
    LIMIT = "limit"
    STOP = "stop"

class OrderStatus(str, enum.Enum):
    PENDING = "pending"
    FILLED = "filled"
    CANCELLED = "cancelled"
    REJECTED = "rejected"
    FAILED = "failed"

def _uuid() -> str:
    return str(uuid.uuid4())

class PortfolioModel(Base):
    __tablename__ = "portfolios"

    id = Column(String(36), primary_key=True, default=_uuid)
    user_id = Column(String(36), ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True)
    name = Column(String(100), nullable=False)
    currency = Column(String(10), default="USD", nullable=False)
    balance = Column(Numeric(18, 4), default=0.0)
    initial_balance = Column(Numeric(18, 4), nullable=True)  # Starting capital
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # Relationships
    # Note: 'user' relationship defined in auth/models.py needs back_populates matching this
    # but we can unidirectionally link here or update User model later.
    # For now, we rely on the FK.
    positions = relationship("PositionModel", back_populates="portfolio", cascade="all, delete-orphan")
    orders = relationship("OrderModel", back_populates="portfolio", cascade="all, delete-orphan")


class PositionModel(Base):
    __tablename__ = "positions"

    id = Column(String(36), primary_key=True, default=_uuid)
    portfolio_id = Column(String(36), ForeignKey("portfolios.id", ondelete="CASCADE"), nullable=False, index=True)
    symbol = Column(String(20), nullable=False, index=True)
    quantity = Column(Numeric(18, 4), default=0.0)
    side = Column(String(10), nullable=True)  # LONG/SHORT from trading_state.json
    average_entry_price = Column(Numeric(18, 4), default=0.0)
    current_price = Column(Numeric(18, 4), nullable=True)
    unrealized_pnl = Column(Numeric(18, 4), default=0.0)
    pnl_pct = Column(Numeric(10, 6), nullable=True)  # P&L percentage
    entry_time = Column(DateTime, nullable=True)  # When position was opened
    last_updated = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    portfolio = relationship("PortfolioModel", back_populates="positions")


class OrderModel(Base):
    __tablename__ = "orders"

    id = Column(String(36), primary_key=True, default=_uuid)
    portfolio_id = Column(String(36), ForeignKey("portfolios.id", ondelete="CASCADE"), nullable=False, index=True)
    symbol = Column(String(20), nullable=False, index=True)
    side = Column(Enum(OrderSide), nullable=False)
    order_type = Column(Enum(OrderType), nullable=False)
    status = Column(Enum(OrderStatus), default=OrderStatus.PENDING, index=True)
    quantity = Column(Numeric(18, 4), nullable=False)
    filled_quantity = Column(Numeric(18, 4), default=0.0)
    price = Column(Numeric(18, 4), nullable=True) # Limit price or execution price
    commission = Column(Numeric(18, 4), default=0.0)
    
    # Metadata
    client_order_id = Column(String(100), nullable=True, unique=True)
    exchange_order_id = Column(String(100), nullable=True)
    error_message = Column(String(500), nullable=True)
    
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    portfolio = relationship("PortfolioModel", back_populates="orders")


class SignalModel(Base):
    __tablename__ = "signals"

    id = Column(String(36), primary_key=True, default=_uuid)
    symbol = Column(String(20), nullable=False, index=True)
    signal_value = Column(Float, nullable=False)
    confidence = Column(Float, nullable=False)
    direction = Column(String(20), nullable=True)  # STRONG BUY, BUY, NEUTRAL, SELL, etc.
    strength_pct = Column(Numeric(10, 6), nullable=True)  # Signal strength percentage
    regime = Column(String(20), nullable=True)
    model_version = Column(String(50), nullable=True)
    generated_at = Column(DateTime, default=datetime.utcnow, index=True)
    metadata_json = Column(JSON, nullable=True)

    # Optional linkage to an order if this signal triggered it
    triggered_order_id = Column(String(36), ForeignKey("orders.id"), nullable=True)
