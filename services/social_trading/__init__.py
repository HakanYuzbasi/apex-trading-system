"""World-Class Social Trading - Advanced Copy Trading and Community Insights.

Enables traders to share insights, follow top-performing portfolios, and
automatically replicate strategies with enterprise-grade synchronization and
risk management.

Features:
    - High-performance copy trading with sub-millisecond latency
    - Social feeds with real-time trading updates
    - Advanced following/subscriber management
    - Performance leaderboards and trader profiles
    - Risk-aware synchronization and portfolio weighting
    - Community engagement tools (comments, likes, shares)
    - Revenue sharing for top-performing strategy providers

Architecture:
    - Real-time event streaming with RabbitMQ
    - Distributed caching with Redis for low-latency copy signals
    - Relational data in PostgreSQL for trader relationships
    - Analytics and time-series data in MongoDB
    - Secure signal distribution with end-to-end encryption

Author: ApexTrader Development Team
Version: 1.0.0
"""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .service import SocialTradingService
    from .schemas import (
        TraderProfile,
        SocialFeedPost,
        CopyTradingSignal,
        SubscriberSettings
    )

__version__ = "1.0.0"
__author__ = "ApexTrader Development Team"
__all__ = [
    "SocialTradingService",
    "TraderProfile",
    "SocialFeedPost",
    "CopyTradingSignal",
    "SubscriberSettings",
]
