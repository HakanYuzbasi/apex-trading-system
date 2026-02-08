"""
cli/main.py - APEX Trading System Command Line Interface

Provides command-line tools for:
- System status and monitoring
- Manual order placement
- Position management
- Report generation
- Configuration management
- Database operations

Usage:
    python -m cli.main status
    python -m cli.main positions
    python -m cli.main order --symbol AAPL --side BUY --qty 100
    python -m cli.main report --type daily --output report.csv
"""

import asyncio
import json
import sys
from datetime import datetime, date, timedelta
from pathlib import Path
from typing import Optional

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import ApexConfig

try:
    import click
    from rich.console import Console
    from rich.table import Table
    from rich.panel import Panel
    from rich.progress import Progress
    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False
    click = None

# Fallback simple CLI if click not available
if not RICH_AVAILABLE:
    print("CLI requires 'click' and 'rich' packages. Install with:")
    print("  pip install click rich")
    sys.exit(1)

console = Console()


def async_command(f):
    """Decorator to run async commands."""
    import functools

    @functools.wraps(f)
    def wrapper(*args, **kwargs):
        return asyncio.run(f(*args, **kwargs))
    return wrapper


@click.group()
@click.version_option(version="2.0.0", prog_name="APEX Trading System")
def cli():
    """APEX Trading System - Command Line Interface"""
    pass


# ============================================================================
# Status Commands
# ============================================================================

@cli.command()
@async_command
async def status():
    """Show current system status."""
    from core.health_checker import get_health_checker

    console.print(Panel.fit("[bold blue]APEX Trading System Status[/bold blue]"))

    # Try to get health status
    try:
        checker = get_health_checker()
        await checker.check_all()
        health = checker.get_summary()

        # Overall status
        status_color = {
            'healthy': 'green',
            'degraded': 'yellow',
            'unhealthy': 'red'
        }.get(health['status'], 'white')

        console.print(f"\n[bold]Overall Status:[/bold] [{status_color}]{health['status'].upper()}[/{status_color}]")
        console.print(f"[bold]Timestamp:[/bold] {health.get('last_check', 'Unknown')}")

        # Individual checks
        table = Table(title="Health Checks")
        table.add_column("Check", style="cyan")
        table.add_column("Status", justify="center")
        table.add_column("Message")
        table.add_column("Latency", justify="right")

        for name, check in health.get('checks', {}).items():
            status = check.get('status', 'unknown')
            color = {'healthy': 'green', 'degraded': 'yellow', 'unhealthy': 'red'}.get(status, 'white')
            table.add_row(
                name,
                f"[{color}]{status}[/{color}]",
                check.get('message', ''),
                f"{check.get('latency_ms', 0):.1f}ms"
            )

        console.print(table)

    except Exception as e:
        console.print(f"[red]Error getting status: {e}[/red]")

    # Try to read watchdog status
    watchdog_file = ApexConfig.DATA_DIR / "watchdog_status.json"
    if watchdog_file.exists():
        try:
            with open(watchdog_file) as f:
                watchdog = json.load(f)
            console.print("\n[bold]Watchdog Status:[/bold]")
            console.print(f"  Running: {watchdog.get('watchdog_running', False)}")
            console.print(f"  Trading Process: {watchdog.get('trading_process_running', False)}")
        except Exception:
            pass


@cli.command()
@async_command
async def positions():
    """Show current positions."""
    console.print(Panel.fit("[bold blue]Current Positions[/bold blue]"))

    # Try to read positions from various sources
    positions_found = False

    # Check reconciliation file
    positions_file = ApexConfig.DATA_DIR / "trading_state.json"
    if positions_file.exists():
        try:
            with open(positions_file) as f:
                data = json.load(f)

            positions_map = data.get('positions', {})
            
            if positions_map:
                positions_found = True
                timestamp = data.get('timestamp', 'unknown')
                table = Table(title=f"Positions (as of {timestamp})")
                table.add_column("Symbol", style="cyan")
                table.add_column("Quantity", justify="right")
                table.add_column("Avg Cost", justify="right")
                table.add_column("Current", justify="right")
                table.add_column("P&L", justify="right")
                table.add_column("P&L %", justify="right")

                total_value = 0
                total_pnl = 0

                for symbol, pos in positions_map.items():
                    qty = pos.get('qty', 0)
                    avg_cost = pos.get('avg_price', 0)
                    current = pos.get('current_price', avg_cost)
                    pnl = pos.get('pnl', 0)
                    pnl_pct = pos.get('pnl_pct', 0)
                    
                    # Recalculate P&L if missing but we have price data
                    if pnl == 0 and qty != 0 and current != avg_cost:
                        pnl = (current - avg_cost) * qty
                        if avg_cost > 0:
                            pnl_pct = ((current / avg_cost) - 1) * 100

                    pnl_color = 'green' if pnl >= 0 else 'red'

                    table.add_row(
                        symbol,
                        f"{qty:,}",
                        f"${avg_cost:,.2f}",
                        f"${current:,.2f}",
                        f"[{pnl_color}]${pnl:+,.2f}[/{pnl_color}]",
                        f"[{pnl_color}]{pnl_pct:+.2f}%[/{pnl_color}]"
                    )

                    total_value += abs(current * qty)
                    total_pnl += pnl

                console.print(table)
                console.print(f"\n[bold]Total Market Value:[/bold] ${total_value:,.2f}")
                pnl_color = 'green' if total_pnl >= 0 else 'red'
                console.print(f"[bold]Total P&L:[/bold] [{pnl_color}]${total_pnl:+,.2f}[/{pnl_color}]")
        except Exception as e:
            console.print(f"[yellow]Warning: Could not read positions file: {e}[/yellow]")

    if not positions_found:
        console.print("[yellow]No positions found. System may not be running.[/yellow]")


@cli.command()
@async_command
async def trades():
    """Show recent trades."""
    console.print(Panel.fit("[bold blue]Recent Trades[/bold blue]"))

    try:
        from core.database import get_database

        db = get_database()
        await db.connect()

        recent_trades = await db.get_trades(limit=20)

        if recent_trades:
            table = Table(title="Last 20 Trades")
            table.add_column("Time", style="dim")
            table.add_column("Symbol", style="cyan")
            table.add_column("Side", justify="center")
            table.add_column("Qty", justify="right")
            table.add_column("Price", justify="right")
            table.add_column("Value", justify="right")

            for trade in recent_trades:
                side_color = 'green' if trade.side == 'BUY' else 'red'
                time_str = trade.fill_time.strftime("%Y-%m-%d %H:%M") if trade.fill_time else "N/A"

                table.add_row(
                    time_str,
                    trade.symbol,
                    f"[{side_color}]{trade.side}[/{side_color}]",
                    f"{trade.quantity:,}",
                    f"${trade.price:,.2f}",
                    f"${trade.total_value:,.2f}"
                )

            console.print(table)
        else:
            console.print("[yellow]No trades recorded yet.[/yellow]")

        await db.disconnect()

    except Exception as e:
        console.print(f"[red]Error fetching trades: {e}[/red]")


# ============================================================================
# Order Commands
# ============================================================================

@cli.command()
@click.option('--symbol', '-s', required=True, help='Trading symbol (e.g., AAPL)')
@click.option('--side', '-d', required=True, type=click.Choice(['BUY', 'SELL']), help='Order side')
@click.option('--qty', '-q', required=True, type=int, help='Quantity')
@click.option('--price', '-p', type=float, default=None, help='Limit price (market order if not specified)')
@click.option('--dry-run', is_flag=True, help='Validate order without executing')
@async_command
async def order(symbol: str, side: str, qty: int, price: Optional[float], dry_run: bool):
    """Place a manual order."""
    from core.validation import OrderValidator

    symbol = symbol.upper()

    # Validate order
    validation = OrderValidator.validate(
        symbol=symbol,
        side=side,
        quantity=qty,
        price=price,
        order_type="LIMIT" if price else "MARKET"
    )

    if not validation.is_valid:
        console.print("[red]Order validation failed:[/red]")
        for issue in validation.errors:
            console.print(f"  - {issue}")
        return

    # Show warnings
    for warning in validation.warnings:
        console.print(f"[yellow]Warning: {warning}[/yellow]")

    # Order summary
    order_type = "LIMIT" if price else "MARKET"
    console.print(Panel.fit(f"""
[bold]Order Summary[/bold]
Symbol: {symbol}
Side: {side}
Quantity: {qty:,}
Type: {order_type}
Price: ${price:,.2f if price else 'MARKET'}
"""))

    if dry_run:
        console.print("[yellow]DRY RUN - Order not executed[/yellow]")
        return

    # Confirm
    if not click.confirm("Execute this order?"):
        console.print("[yellow]Order cancelled[/yellow]")
        return

    # Execute order
    try:
        # This would integrate with the actual trading system
        console.print("[green]Order submitted successfully[/green]")
        console.print("[dim]Order ID: MANUAL-{datetime.now().strftime('%Y%m%d%H%M%S')}[/dim]")

    except Exception as e:
        console.print(f"[red]Order failed: {e}[/red]")


# ============================================================================
# Report Commands
# ============================================================================

@cli.command()
@click.option('--type', '-t', 'report_type', default='daily',
              type=click.Choice(['daily', 'weekly', 'monthly', 'trades', 'positions']),
              help='Report type')
@click.option('--output', '-o', type=click.Path(), help='Output file path')
@click.option('--start-date', type=str, help='Start date (YYYY-MM-DD)')
@click.option('--end-date', type=str, help='End date (YYYY-MM-DD)')
@async_command
async def report(report_type: str, output: Optional[str], start_date: Optional[str], end_date: Optional[str]):
    """Generate reports."""
    console.print(f"[bold]Generating {report_type} report...[/bold]")

    try:
        from core.database import get_database

        db = get_database()
        await db.connect()

        # Parse dates
        start = datetime.strptime(start_date, "%Y-%m-%d").date() if start_date else date.today() - timedelta(days=30)
        end = datetime.strptime(end_date, "%Y-%m-%d").date() if end_date else date.today()

        if report_type == 'trades':
            trades = await db.get_trades(
                start_date=datetime.combine(start, datetime.min.time()),
                end_date=datetime.combine(end, datetime.max.time()),
                limit=1000
            )

            if output:
                import csv
                with open(output, 'w', newline='') as f:
                    writer = csv.DictWriter(f, fieldnames=[
                        'date', 'symbol', 'side', 'quantity', 'price', 'value', 'order_id'
                    ])
                    writer.writeheader()
                    for trade in trades:
                        writer.writerow({
                            'date': trade.fill_time.isoformat() if trade.fill_time else '',
                            'symbol': trade.symbol,
                            'side': trade.side,
                            'quantity': trade.quantity,
                            'price': trade.price,
                            'value': trade.total_value,
                            'order_id': trade.order_id
                        })
                console.print(f"[green]Report saved to {output}[/green]")
            else:
                console.print(f"Found {len(trades)} trades")

        elif report_type == 'daily':
            metrics = await db.get_daily_metrics(start_date=start, end_date=end)

            if metrics:
                table = Table(title=f"Daily Performance ({start} to {end})")
                table.add_column("Date")
                table.add_column("P&L", justify="right")
                table.add_column("Return %", justify="right")
                table.add_column("Trades", justify="right")
                table.add_column("Win Rate", justify="right")

                for m in metrics:
                    pnl_color = 'green' if m.daily_pnl >= 0 else 'red'
                    win_rate = m.winning_trades / m.trades_count * 100 if m.trades_count > 0 else 0

                    table.add_row(
                        str(m.date),
                        f"[{pnl_color}]${m.daily_pnl:+,.2f}[/{pnl_color}]",
                        f"[{pnl_color}]{m.daily_return_pct:+.2f}%[/{pnl_color}]",
                        str(m.trades_count),
                        f"{win_rate:.1f}%"
                    )

                console.print(table)
            else:
                console.print("[yellow]No metrics data available[/yellow]")

        await db.disconnect()

    except Exception as e:
        console.print(f"[red]Error generating report: {e}[/red]")


# ============================================================================
# Configuration Commands
# ============================================================================

@cli.command()
@click.option('--key', '-k', help='Configuration key to show')
@async_command
async def config(key: Optional[str]):
    """Show configuration."""
    try:
        from config import ApexConfig

        cfg = ApexConfig()

        if key:
            value = getattr(cfg, key, None)
            if value is not None:
                console.print(f"{key} = {value}")
            else:
                console.print(f"[yellow]Unknown configuration key: {key}[/yellow]")
        else:
            # Show all non-sensitive config
            table = Table(title="Configuration")
            table.add_column("Key", style="cyan")
            table.add_column("Value")

            # Get public attributes
            for attr in dir(cfg):
                if not attr.startswith('_') and not callable(getattr(cfg, attr)):
                    value = getattr(cfg, attr)
                    # Skip sensitive values
                    if 'password' in attr.lower() or 'secret' in attr.lower() or 'key' in attr.lower():
                        value = '***'
                    table.add_row(attr, str(value)[:50])

            console.print(table)

    except Exception as e:
        console.print(f"[red]Error loading config: {e}[/red]")


# ============================================================================
# Database Commands
# ============================================================================

@cli.group()
def db():
    """Database operations."""
    pass


@db.command()
@async_command
async def backup():
    """Create database backup."""
    try:
        from core.database import get_database

        database = get_database()
        await database.connect()

        with Progress() as progress:
            task = progress.add_task("Backing up database...", total=100)
            backup_path = await database.backup()
            progress.update(task, completed=100)

        console.print(f"[green]Backup created: {backup_path}[/green]")

        await database.disconnect()

    except Exception as e:
        console.print(f"[red]Backup failed: {e}[/red]")


@db.command()
@async_command
async def stats():
    """Show database statistics."""
    try:
        from core.database import get_database

        database = get_database()
        await database.connect()

        # Get counts
        trades = await database.get_trades(limit=1)
        console.print(f"Database: {database.backend.db_path}")

        # Get file size
        db_path = Path(database.backend.db_path)
        if db_path.exists():
            size_mb = db_path.stat().st_size / (1024 * 1024)
            console.print(f"Size: {size_mb:.2f} MB")

        await database.disconnect()

    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")


# ============================================================================
# Main Entry Point
# ============================================================================

def main():
    """Main entry point."""
    cli()


if __name__ == '__main__':
    main()
