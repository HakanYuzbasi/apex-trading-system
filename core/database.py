"""
core/database.py - Database Persistence Layer

Provides persistent storage for trades, positions, and performance metrics.
Supports SQLite for local development and PostgreSQL for production.

Features:
- Trade history recording with full audit trail
- Position snapshot persistence
- Performance metrics storage
- Automatic backup functionality
- Migration support
"""

import asyncio
import aiosqlite
import json
import logging
import os
from abc import ABC, abstractmethod
from contextlib import asynccontextmanager
from dataclasses import dataclass, asdict
from datetime import datetime, date, timedelta
from pathlib import Path
from typing import Optional, List, Dict, Any, AsyncIterator
from enum import Enum

from config import ApexConfig

logger = logging.getLogger(__name__)


class DatabaseType(Enum):
    """Supported database types."""
    SQLITE = "sqlite"
    POSTGRESQL = "postgresql"


@dataclass
class TradeRecord:
    """A recorded trade."""
    id: Optional[int] = None
    symbol: str = ""
    side: str = ""  # BUY or SELL
    quantity: int = 0
    price: float = 0.0
    total_value: float = 0.0
    commission: float = 0.0
    order_id: str = ""
    order_type: str = ""
    fill_time: Optional[datetime] = None
    signal_strength: float = 0.0
    regime: str = ""
    notes: str = ""
    created_at: Optional[datetime] = None

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        # Convert datetime to ISO format
        for key in ['fill_time', 'created_at']:
            if data[key]:
                data[key] = data[key].isoformat()
        return data


@dataclass
class PositionSnapshot:
    """A snapshot of a position."""
    id: Optional[int] = None
    symbol: str = ""
    quantity: int = 0
    avg_cost: float = 0.0
    current_price: float = 0.0
    market_value: float = 0.0
    unrealized_pnl: float = 0.0
    realized_pnl: float = 0.0
    snapshot_time: Optional[datetime] = None


@dataclass
class DailyMetrics:
    """Daily performance metrics."""
    id: Optional[int] = None
    date: Optional[date] = None
    starting_capital: float = 0.0
    ending_capital: float = 0.0
    daily_pnl: float = 0.0
    daily_return_pct: float = 0.0
    trades_count: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    max_drawdown: float = 0.0
    sharpe_ratio: Optional[float] = None
    notes: str = ""


class DatabaseBackend(ABC):
    """Abstract database backend."""

    @abstractmethod
    async def connect(self):
        """Establish database connection."""
        pass

    @abstractmethod
    async def disconnect(self):
        """Close database connection."""
        pass

    @abstractmethod
    async def initialize_schema(self):
        """Create database tables if they don't exist."""
        pass

    @abstractmethod
    async def record_trade(self, trade: TradeRecord) -> int:
        """Record a trade and return its ID."""
        pass

    @abstractmethod
    async def get_trades(
        self,
        symbol: Optional[str] = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        limit: int = 100
    ) -> List[TradeRecord]:
        """Get trade history."""
        pass

    @abstractmethod
    async def save_position_snapshot(self, snapshot: PositionSnapshot) -> int:
        """Save a position snapshot."""
        pass

    @abstractmethod
    async def get_position_snapshots(
        self,
        symbol: Optional[str] = None,
        limit: int = 100
    ) -> List[PositionSnapshot]:
        """Get position snapshots."""
        pass

    @abstractmethod
    async def save_daily_metrics(self, metrics: DailyMetrics) -> int:
        """Save daily metrics."""
        pass

    @abstractmethod
    async def get_daily_metrics(
        self,
        start_date: Optional[date] = None,
        end_date: Optional[date] = None
    ) -> List[DailyMetrics]:
        """Get daily metrics history."""
        pass


class SQLiteBackend(DatabaseBackend):
    """SQLite database backend."""

    def __init__(self, db_path: str = None):
        self.db_path = db_path or str(ApexConfig.DATA_DIR / "apex_trading.db")
        self.connection: Optional[aiosqlite.Connection] = None

        # Ensure directory exists
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)

    async def connect(self):
        """Connect to SQLite database."""
        self.connection = await aiosqlite.connect(self.db_path)
        self.connection.row_factory = aiosqlite.Row
        await self.connection.execute("PRAGMA journal_mode=WAL")
        await self.connection.execute("PRAGMA foreign_keys=ON")
        logger.info(f"Connected to SQLite database: {self.db_path}")

    async def disconnect(self):
        """Close SQLite connection."""
        if self.connection:
            await self.connection.close()
            logger.info("Disconnected from SQLite database")

    async def initialize_schema(self):
        """Create tables if they don't exist."""
        if not self.connection:
            await self.connect()

        await self.connection.executescript("""
            -- Trades table
            CREATE TABLE IF NOT EXISTS trades (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol TEXT NOT NULL,
                side TEXT NOT NULL,
                quantity INTEGER NOT NULL,
                price REAL NOT NULL,
                total_value REAL NOT NULL,
                commission REAL DEFAULT 0,
                order_id TEXT,
                order_type TEXT,
                fill_time TIMESTAMP,
                signal_strength REAL,
                regime TEXT,
                notes TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );

            -- Position snapshots table
            CREATE TABLE IF NOT EXISTS position_snapshots (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol TEXT NOT NULL,
                quantity INTEGER NOT NULL,
                avg_cost REAL NOT NULL,
                current_price REAL NOT NULL,
                market_value REAL NOT NULL,
                unrealized_pnl REAL,
                realized_pnl REAL,
                snapshot_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );

            -- Daily metrics table
            CREATE TABLE IF NOT EXISTS daily_metrics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                date DATE UNIQUE NOT NULL,
                starting_capital REAL NOT NULL,
                ending_capital REAL NOT NULL,
                daily_pnl REAL NOT NULL,
                daily_return_pct REAL NOT NULL,
                trades_count INTEGER DEFAULT 0,
                winning_trades INTEGER DEFAULT 0,
                losing_trades INTEGER DEFAULT 0,
                max_drawdown REAL,
                sharpe_ratio REAL,
                notes TEXT
            );

            -- System events table
            CREATE TABLE IF NOT EXISTS system_events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                event_type TEXT NOT NULL,
                event_data TEXT,
                severity TEXT DEFAULT 'info',
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );

            -- Create indexes
            CREATE INDEX IF NOT EXISTS idx_trades_symbol ON trades(symbol);
            CREATE INDEX IF NOT EXISTS idx_trades_fill_time ON trades(fill_time);
            CREATE INDEX IF NOT EXISTS idx_trades_created_at ON trades(created_at);
            CREATE INDEX IF NOT EXISTS idx_position_snapshots_symbol ON position_snapshots(symbol);
            CREATE INDEX IF NOT EXISTS idx_daily_metrics_date ON daily_metrics(date);
        """)
        await self.connection.commit()
        logger.info("Database schema initialized")

    async def record_trade(self, trade: TradeRecord) -> int:
        """Record a trade."""
        if not self.connection:
            await self.connect()

        cursor = await self.connection.execute("""
            INSERT INTO trades (
                symbol, side, quantity, price, total_value, commission,
                order_id, order_type, fill_time, signal_strength, regime, notes
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            trade.symbol, trade.side, trade.quantity, trade.price,
            trade.total_value, trade.commission, trade.order_id,
            trade.order_type, trade.fill_time, trade.signal_strength,
            trade.regime, trade.notes
        ))
        await self.connection.commit()

        trade_id = cursor.lastrowid
        logger.debug(f"Recorded trade {trade_id}: {trade.side} {trade.quantity} {trade.symbol}")
        return trade_id

    async def get_trades(
        self,
        symbol: Optional[str] = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        limit: int = 100
    ) -> List[TradeRecord]:
        """Get trade history."""
        if not self.connection:
            await self.connect()

        query = "SELECT * FROM trades WHERE 1=1"
        params = []

        if symbol:
            query += " AND symbol = ?"
            params.append(symbol)

        if start_date:
            query += " AND fill_time >= ?"
            params.append(start_date.isoformat())

        if end_date:
            query += " AND fill_time <= ?"
            params.append(end_date.isoformat())

        query += " ORDER BY fill_time DESC LIMIT ?"
        params.append(limit)

        cursor = await self.connection.execute(query, params)
        rows = await cursor.fetchall()

        trades = []
        for row in rows:
            trade = TradeRecord(
                id=row['id'],
                symbol=row['symbol'],
                side=row['side'],
                quantity=row['quantity'],
                price=row['price'],
                total_value=row['total_value'],
                commission=row['commission'] or 0,
                order_id=row['order_id'] or '',
                order_type=row['order_type'] or '',
                fill_time=datetime.fromisoformat(row['fill_time']) if row['fill_time'] else None,
                signal_strength=row['signal_strength'] or 0,
                regime=row['regime'] or '',
                notes=row['notes'] or '',
                created_at=datetime.fromisoformat(row['created_at']) if row['created_at'] else None
            )
            trades.append(trade)

        return trades

    async def save_position_snapshot(self, snapshot: PositionSnapshot) -> int:
        """Save a position snapshot."""
        if not self.connection:
            await self.connect()

        cursor = await self.connection.execute("""
            INSERT INTO position_snapshots (
                symbol, quantity, avg_cost, current_price,
                market_value, unrealized_pnl, realized_pnl, snapshot_time
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            snapshot.symbol, snapshot.quantity, snapshot.avg_cost,
            snapshot.current_price, snapshot.market_value,
            snapshot.unrealized_pnl, snapshot.realized_pnl,
            snapshot.snapshot_time or datetime.now()
        ))
        await self.connection.commit()
        return cursor.lastrowid

    async def get_position_snapshots(
        self,
        symbol: Optional[str] = None,
        limit: int = 100
    ) -> List[PositionSnapshot]:
        """Get position snapshots."""
        if not self.connection:
            await self.connect()

        query = "SELECT * FROM position_snapshots"
        params = []

        if symbol:
            query += " WHERE symbol = ?"
            params.append(symbol)

        query += " ORDER BY snapshot_time DESC LIMIT ?"
        params.append(limit)

        cursor = await self.connection.execute(query, params)
        rows = await cursor.fetchall()

        snapshots = []
        for row in rows:
            snapshot = PositionSnapshot(
                id=row['id'],
                symbol=row['symbol'],
                quantity=row['quantity'],
                avg_cost=row['avg_cost'],
                current_price=row['current_price'],
                market_value=row['market_value'],
                unrealized_pnl=row['unrealized_pnl'] or 0,
                realized_pnl=row['realized_pnl'] or 0,
                snapshot_time=datetime.fromisoformat(row['snapshot_time']) if row['snapshot_time'] else None
            )
            snapshots.append(snapshot)

        return snapshots

    async def save_daily_metrics(self, metrics: DailyMetrics) -> int:
        """Save daily metrics."""
        if not self.connection:
            await self.connect()

        cursor = await self.connection.execute("""
            INSERT OR REPLACE INTO daily_metrics (
                date, starting_capital, ending_capital, daily_pnl,
                daily_return_pct, trades_count, winning_trades,
                losing_trades, max_drawdown, sharpe_ratio, notes
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            metrics.date or date.today(),
            metrics.starting_capital, metrics.ending_capital,
            metrics.daily_pnl, metrics.daily_return_pct,
            metrics.trades_count, metrics.winning_trades,
            metrics.losing_trades, metrics.max_drawdown,
            metrics.sharpe_ratio, metrics.notes
        ))
        await self.connection.commit()
        return cursor.lastrowid

    async def get_daily_metrics(
        self,
        start_date: Optional[date] = None,
        end_date: Optional[date] = None
    ) -> List[DailyMetrics]:
        """Get daily metrics history."""
        if not self.connection:
            await self.connect()

        query = "SELECT * FROM daily_metrics WHERE 1=1"
        params = []

        if start_date:
            query += " AND date >= ?"
            params.append(start_date.isoformat())

        if end_date:
            query += " AND date <= ?"
            params.append(end_date.isoformat())

        query += " ORDER BY date DESC"

        cursor = await self.connection.execute(query, params)
        rows = await cursor.fetchall()

        metrics_list = []
        for row in rows:
            metrics = DailyMetrics(
                id=row['id'],
                date=date.fromisoformat(row['date']) if row['date'] else None,
                starting_capital=row['starting_capital'],
                ending_capital=row['ending_capital'],
                daily_pnl=row['daily_pnl'],
                daily_return_pct=row['daily_return_pct'],
                trades_count=row['trades_count'] or 0,
                winning_trades=row['winning_trades'] or 0,
                losing_trades=row['losing_trades'] or 0,
                max_drawdown=row['max_drawdown'] or 0,
                sharpe_ratio=row['sharpe_ratio'],
                notes=row['notes'] or ''
            )
            metrics_list.append(metrics)

        return metrics_list

    async def record_event(self, event_type: str, data: Dict[str, Any], severity: str = "info"):
        """Record a system event."""
        if not self.connection:
            await self.connect()

        await self.connection.execute("""
            INSERT INTO system_events (event_type, event_data, severity)
            VALUES (?, ?, ?)
        """, (event_type, json.dumps(data), severity))
        await self.connection.commit()

    async def backup(self, backup_path: Optional[str] = None) -> Path:
        """Create a database backup."""
        if backup_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_path = str(ApexConfig.DATA_DIR / "backups" / f"apex_trading_{timestamp}.db")

        Path(backup_path).parent.mkdir(parents=True, exist_ok=True)

        if self.connection:
            await self.connection.execute(f"VACUUM INTO '{backup_path}'")
            logger.info(f"Database backed up to: {backup_path}")

        return Path(backup_path)


class Database:
    """
    Main database interface.

    Example usage:
        db = Database()
        await db.connect()

        # Record a trade
        trade = TradeRecord(
            symbol="AAPL",
            side="BUY",
            quantity=100,
            price=185.50,
            total_value=18550.0
        )
        trade_id = await db.record_trade(trade)

        # Get trade history
        trades = await db.get_trades(symbol="AAPL", limit=10)
    """

    def __init__(
        self,
        db_type: DatabaseType = DatabaseType.SQLITE,
        db_path: str = None
    ):
        self.db_type = db_type

        if db_type == DatabaseType.SQLITE:
            self.backend = SQLiteBackend(db_path)
        else:
            raise NotImplementedError(f"Database type {db_type} not implemented")

    async def connect(self):
        """Connect to database."""
        await self.backend.connect()
        await self.backend.initialize_schema()

    async def disconnect(self):
        """Disconnect from database."""
        await self.backend.disconnect()

    @asynccontextmanager
    async def session(self) -> AsyncIterator['Database']:
        """Context manager for database session."""
        await self.connect()
        try:
            yield self
        finally:
            await self.disconnect()

    # Delegate methods to backend
    async def record_trade(self, trade: TradeRecord) -> int:
        return await self.backend.record_trade(trade)

    async def get_trades(self, **kwargs) -> List[TradeRecord]:
        return await self.backend.get_trades(**kwargs)

    async def save_position_snapshot(self, snapshot: PositionSnapshot) -> int:
        return await self.backend.save_position_snapshot(snapshot)

    async def get_position_snapshots(self, **kwargs) -> List[PositionSnapshot]:
        return await self.backend.get_position_snapshots(**kwargs)

    async def save_daily_metrics(self, metrics: DailyMetrics) -> int:
        return await self.backend.save_daily_metrics(metrics)

    async def get_daily_metrics(self, **kwargs) -> List[DailyMetrics]:
        return await self.backend.get_daily_metrics(**kwargs)

    async def backup(self, path: str = None) -> Path:
        if isinstance(self.backend, SQLiteBackend):
            return await self.backend.backup(path)
        raise NotImplementedError("Backup not available for this backend")


# Global database instance
_database: Optional[Database] = None


def get_database() -> Database:
    """Get the global database instance."""
    global _database
    if _database is None:
        _database = Database()
    return _database


async def init_database(db_type: DatabaseType = DatabaseType.SQLITE, db_path: str = None):
    """Initialize the global database."""
    global _database
    _database = Database(db_type, db_path)
    await _database.connect()
    return _database
