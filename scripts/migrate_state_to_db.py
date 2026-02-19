#!/usr/bin/env python3
"""
Migrate trading_state.json to PostgreSQL database.

This script imports current positions, signals, and portfolio data
from the JSON file into the database.
"""

import asyncio
import json
import sys
from datetime import datetime, timedelta
from pathlib import Path
from uuid import uuid4

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from sqlalchemy import select, delete
from services.common.db import db_session
from services.auth.models import UserModel
from services.trading.models import PortfolioModel, PositionModel, SignalModel
from config import ApexConfig


async def migrate_state_to_db(dry_run: bool = False):
    """
    Main migration function.
    
    Args:
        dry_run: If True, only print what would be done without making changes
    """
    
    # 1. Load trading_state.json
    state_file = ApexConfig.DATA_DIR / "trading_state.json"
    if not state_file.exists():
        print(f"❌ Error: {state_file} not found!")
        return False
    
    with open(state_file) as f:
        state = json.load(f)
    
    print(f"📄 Loaded state from: {state_file}")
    print(f"   Timestamp: {state.get('timestamp')}")
    print(f"   Positions: {len(state.get('positions', {}))}")
    print(f"   Signals: {len(state.get('signals', {}))}")
    print(f"   Capital: ${state.get('capital', 0):,.2f}")
    print()
    
    if dry_run:
        print("🔍 DRY RUN MODE - No changes will be made")
        print()
        return True
    
    async with db_session() as session:
        # 2. Create/Get system user
        result = await session.execute(
            select(UserModel).where(UserModel.email == "system@apex-trading.local")
        )
        user = result.scalar_one_or_none()
        
        if not user:
            print("👤 Creating system user...")
            user = UserModel(
                id=str(uuid4()),
                username="apex-system",
                email="system@apex-trading.local",
                password_hash=None,  # System user, no login
                is_active=True
            )
            session.add(user)
            await session.flush()
            print(f"   ✅ Created user: {user.email}")
        else:
            print(f"   ℹ️  Using existing user: {user.email}")
        
        # 3. Create/Update default portfolio
        result = await session.execute(
            select(PortfolioModel).where(
                PortfolioModel.user_id == user.id,
                PortfolioModel.name == "Main Portfolio"
            )
        )
        portfolio = result.scalar_one_or_none()
        
        if not portfolio:
            print("💼 Creating portfolio...")
            portfolio = PortfolioModel(
                id=str(uuid4()),
                user_id=user.id,
                name="Main Portfolio",
                currency="USD",
                balance=state.get("capital", 0),
                initial_balance=state.get("initial_capital", state.get("capital", 0)),
                created_at=datetime.utcnow()
            )
            session.add(portfolio)
            await session.flush()
            print(f"   ✅ Created portfolio: {portfolio.name}")
        else:
            print(f"   ℹ️  Updating portfolio: {portfolio.name}")
            portfolio.balance = state.get("capital", 0)
            portfolio.initial_balance = state.get("initial_capital", state.get("capital", 0))
        
        print(f"   💰 Balance: ${portfolio.balance:,.2f}")
        print(f"   💵 Initial: ${portfolio.initial_balance:,.2f}")
        print()
        
        # 4. Clear old positions (for re-import)
        print("🧹 Clearing existing positions...")
        result = await session.execute(
            delete(PositionModel).where(PositionModel.portfolio_id == portfolio.id)
        )
        deleted_count = result.rowcount
        print(f"   Deleted {deleted_count} old positions")
        print()
        
        # 5. Import positions
        print("📊 Importing positions...")
        positions_data = state.get("positions", {})
        imported_positions = 0
        
        for symbol, pos_data in positions_data.items():
            try:
                entry_time = None
                if "entry_time" in pos_data and pos_data["entry_time"]:
                    try:
                        entry_time = datetime.fromisoformat(pos_data["entry_time"])
                    except (ValueError, TypeError):
                        pass
                
                position = PositionModel(
                    id=str(uuid4()),
                    portfolio_id=portfolio.id,
                    symbol=symbol,
                    quantity=float(pos_data.get("qty", 0)),
                    side=pos_data.get("side"),
                    average_entry_price=float(pos_data.get("avg_price", 0)),
                    current_price=float(pos_data.get("current_price", 0)),
                    unrealized_pnl=float(pos_data.get("pnl", 0)),
                    pnl_pct=float(pos_data.get("pnl_pct", 0)),
                    entry_time=entry_time,
                    last_updated=datetime.fromisoformat(state["timestamp"])
                )
                session.add(position)
                imported_positions += 1
            except Exception as e:
                print(f"   ⚠️  Error importing position {symbol}: {e}")
        
        print(f"   ✅ Imported {imported_positions} positions")
        print()
        
        # 6. Clear recent signals (last 2 hours)
        print("🧹 Clearing recent signals (last 2 hours)...")
        two_hours_ago = datetime.utcnow() - timedelta(hours=2)
        result = await session.execute(
            delete(SignalModel).where(SignalModel.generated_at > two_hours_ago)
        )
        deleted_signals = result.rowcount
        print(f"   Deleted {deleted_signals} stale signals")
        print()
        
        # 7. Import signals
        print("🎯 Importing signals...")
        signals_data = state.get("signals", {})
        imported_signals = 0
        
        for symbol, signal_data in signals_data.items():
            try:
                generated_at = None
                if "timestamp" in signal_data and signal_data["timestamp"]:
                    try:
                        generated_at = datetime.fromisoformat(signal_data["timestamp"])
                    except (ValueError, TypeError):
                        generated_at = datetime.utcnow()
                else:
                    generated_at = datetime.utcnow()
                
                signal = SignalModel(
                    id=str(uuid4()),
                    symbol=symbol,
                    signal_value=float(signal_data.get("signal", 0)),
                    confidence=float(signal_data.get("confidence", 0)),
                    direction=signal_data.get("direction"),
                    strength_pct=float(signal_data.get("strength_pct", 0)),
                    generated_at=generated_at
                )
                session.add(signal)
                imported_signals += 1
            except Exception as e:
                print(f"   ⚠️  Error importing signal {symbol}: {e}")
        
        print(f"   ✅ Imported {imported_signals} signals")
        print()
        
        # 8. Commit transaction
        await session.commit()
        
    print("=" * 60)
    print("✅ MIGRATION COMPLETE!")
    print("=" * 60)
    print(f"📊 Summary:")
    print(f"   User: {user.email}")
    print(f"   Portfolio: {portfolio.name}")
    print(f"   Balance: ${portfolio.balance:,.2f}")
    print(f"   Positions: {imported_positions}")
    print(f"   Signals: {imported_signals}")
    print()
    
    return True


async def verify_migration():
    """Verify the migration was successful."""
    print("\n🔍 Verifying migration...")
    print("=" * 60)
    
    async with db_session() as session:
        # Check portfolio
        result = await session.execute(
            select(PortfolioModel).where(PortfolioModel.name == "Main Portfolio")
        )
        portfolio = result.scalar_one_or_none()
        
        if not portfolio:
            print("❌ Portfolio not found!")
            return False
        
        print(f"✅ Portfolio: {portfolio.name} (${portfolio.balance:,.2f})")
        
        # Count positions
        result = await session.execute(
            select(PositionModel).where(PositionModel.portfolio_id == portfolio.id)
        )
        positions = result.scalars().all()
        print(f"✅ Positions: {len(positions)}")
        
        # Show sample positions
        if positions:
            print("\n📊 Sample Positions:")
            for pos in positions[:5]:
                print(f"   {pos.symbol}: {pos.side} {pos.quantity} @ ${pos.average_entry_price:.2f} "
                      f"(P&L: ${pos.unrealized_pnl:.2f}, {pos.pnl_pct:.2f}%)")
        
        # Count recent signals
        one_hour_ago = datetime.utcnow() - timedelta(hours=1)
        result = await session.execute(
            select(SignalModel).where(SignalModel.generated_at > one_hour_ago)
        )
        signals = result.scalars().all()
        print(f"\n✅ Recent Signals (last hour): {len(signals)}")
        
        # Show sample signals
        if signals:
            print("\n🎯 Sample Signals:")
            for sig in signals[:5]:
                print(f"   {sig.symbol}: {sig.direction} (signal={sig.signal_value:.3f}, "
                      f"conf={sig.confidence:.3f})")
    
    print("\n" + "=" * 60)
    print("✅ VERIFICATION COMPLETE!")
    print("=" * 60)
    return True


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Migrate trading_state.json to PostgreSQL")
    parser.add_argument("--dry-run", action="store_true", help="Preview changes without applying")
    parser.add_argument("--verify", action="store_true", help="Verify migration results")
    args = parser.parse_args()
    
    if args.verify:
        success = asyncio.run(verify_migration())
    else:
        success = asyncio.run(migrate_state_to_db(dry_run=args.dry_run))
    
    sys.exit(0 if success else 1)
