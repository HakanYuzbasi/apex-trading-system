"""Strategy Marketplace - World-Class Trading Strategy Sharing Platform.

A production-grade marketplace for discovering, sharing, and monetizing
quantitative trading strategies with enterprise-level security, performance,
and analytics.

Features:
    - Advanced strategy discovery with ML-powered recommendations
    - Comprehensive performance analytics and backtesting
    - Secure strategy sharing with encryption and access control
    - Real-time leaderboards and rankings
    - Community reviews, ratings, and social features
    - Strategy versioning and deployment tracking
    - Revenue sharing and licensing management
    - API access for programmatic integration

Architecture:
    - Microservices-based design with async processing
    - Redis caching for sub-millisecond response times
    - PostgreSQL for relational data with full-text search
    - MongoDB for strategy metadata and time-series data
    - Elasticsearch for advanced search capabilities
    - RabbitMQ for event-driven architecture
    - Prometheus metrics and OpenTelemetry tracing

Author: ApexTrader Development Team
Version: 1.0.0
License: Proprietary
"""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .service import StrategyMarketplaceService
    from .schemas import (
        StrategyListing,
        StrategyMetadata,
        PerformanceMetrics,
        UserReview,
        MarketplaceSearch
    )

__version__ = "1.0.0"
__author__ = "ApexTrader Development Team"
__all__ = [
    "StrategyMarketplaceService",
    "StrategyListing",
    "StrategyMetadata",
    "PerformanceMetrics",
    "UserReview",
    "MarketplaceSearch",
]
