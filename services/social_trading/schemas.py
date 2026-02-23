"""World-Class Pydantic Schemas for Social Trading.

Provides comprehensive data validation and serialization for trader profiles,
social feeds, and copy trading signals with strict type safety.
"""

from datetime import datetime
from decimal import Decimal
from enum import Enum
from typing import Optional
from uuid import UUID, uuid4

from pydantic import BaseModel, Field, condecimal, conint, constr


class ProfilePrivacy(str, Enum):
    """Privacy settings for trader profiles."""
    PUBLIC = "public"
    PRIVATE = "private"
    FOLLOWERS_ONLY = "followers_only"


class SignalAction(str, Enum):
    """Actions for copy trading signals."""
    BUY = "buy"
    SELL = "sell"
    CLOSE = "close"
    UPDATE = "update"


class TraderProfile(BaseModel):
    """Comprehensive trader profile with performance stats."""
    
    user_id: UUID
    username: constr(min_length=3, max_length=50)
    bio: Optional[constr(max_length=500)] = None
    avatar_url: Optional[str] = None
    
    # Performance Stats
    total_return_pct: Decimal = Field(default=Decimal("0"))
    sharpe_ratio: Decimal = Field(default=Decimal("0"))
    win_rate: Decimal = Field(default=Decimal("0"))
    max_drawdown: Decimal = Field(default=Decimal("0"))
    
    # Social Stats
    followers_count: conint(ge=0) = 0
    following_count: conint(ge=0) = 0
    subscribers_count: conint(ge=0) = 0
    
    # Settings
    privacy: ProfilePrivacy = ProfilePrivacy.PUBLIC
    is_verified: bool = False
    joined_at: datetime = Field(default_factory=datetime.utcnow)


class SocialFeedPost(BaseModel):
    """Social feed post with engagement metrics."""
    
    post_id: UUID = Field(default_factory=uuid4)
    author_id: UUID
    author_username: str
    
    content: constr(min_length=1, max_length=2000)
    related_strategy_id: Optional[UUID] = None
    
    # Engagement
    likes_count: conint(ge=0) = 0
    comments_count: conint(ge=0) = 0
    shares_count: conint(ge=0) = 0
    
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)


class CopyTradingSignal(BaseModel):
    """High-performance copy trading signal."""
    
    signal_id: UUID = Field(default_factory=uuid4)
    provider_id: UUID
    strategy_id: UUID
    
    symbol: str
    action: SignalAction
    price: Decimal
    quantity_pct: condecimal(gt=0, le=1) = Field(..., description="Percentage of portfolio to allocate")
    
    stop_loss: Optional[Decimal] = None
    take_profit: Optional[Decimal] = None
    
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    expires_at: Optional[datetime] = None


class SubscriberSettings(BaseModel):
    """Individual subscriber settings for copy trading."""
    
    subscriber_id: UUID
    provider_id: UUID
    
    is_active: bool = True
    allocation_limit: Decimal = Field(..., gt=0)
    max_slippage_bps: conint(ge=0) = 50
    
    # Risk Management
    stop_copying_drawdown_pct: Optional[Decimal] = Field(None, ge=0, le=1)
    auto_close_on_provider_exit: bool = True
    
    subscribed_at: datetime = Field(default_factory=datetime.utcnow)
