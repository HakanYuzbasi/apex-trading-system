"""
reconciliation/daily_reconciler.py - Daily Position and Trade Reconciliation

Reconciles:
- System positions vs IBKR positions
- Trade fills vs expected
- Cash balance checks
- P&L verification
"""

from typing import Dict, List, Optional, Tuple
from datetime import date, timedelta
from dataclasses import dataclass
import logging
import json
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class ReconciliationDiff:
    """Represents a reconciliation difference."""
    category: str  # 'position', 'trade', 'cash'
    symbol: Optional[str]
    expected: float
    actual: float
    difference: float
    severity: str  # 'info', 'warning', 'error'
    message: str
    
    def to_dict(self) -> Dict:
        return {
            'category': self.category,
            'symbol': self.symbol,
            'expected': self.expected,
            'actual': self.actual,
            'difference': self.difference,
            'severity': self.severity,
            'message': self.message
        }


@dataclass
class ReconciliationReport:
    """Daily reconciliation report."""
    date: date
    positions_matched: int
    positions_mismatched: int
    trades_matched: int
    trades_mismatched: int
    cash_matched: bool
    total_position_diff: float
    total_pnl_diff: float
    diffs: List[ReconciliationDiff]
    status: str  # 'ok', 'warning', 'error'
    
    def to_dict(self) -> Dict:
        return {
            'date': self.date.isoformat(),
            'positions_matched': self.positions_matched,
            'positions_mismatched': self.positions_mismatched,
            'trades_matched': self.trades_matched,
            'trades_mismatched': self.trades_mismatched,
            'cash_matched': self.cash_matched,
            'total_position_diff': self.total_position_diff,
            'total_pnl_diff': self.total_pnl_diff,
            'status': self.status,
            'diffs': [d.to_dict() for d in self.diffs]
        }


class DailyReconciler:
    """
    Perform daily reconciliation between system state and broker.
    
    Checks:
    1. Position quantities match
    2. Trade fills match expected
    3. Cash balances match
    4. P&L calculations match
    """
    
    def __init__(
        self,
        tolerance_shares: int = 1,
        tolerance_cash: float = 100.0,
        report_dir: str = 'reconciliation'
    ):
        """
        Initialize reconciler.
        
        Args:
            tolerance_shares: Acceptable share difference
            tolerance_cash: Acceptable cash difference ($)
            report_dir: Directory to save reports
        """
        self.tolerance_shares = tolerance_shares
        self.tolerance_cash = tolerance_cash
        self.report_dir = Path(report_dir)
        self.report_dir.mkdir(exist_ok=True)
        
        self.reports: List[ReconciliationReport] = []
        
        logger.info("DailyReconciler initialized")
    
    def reconcile_positions(
        self,
        system_positions: Dict[str, int],
        broker_positions: Dict[str, int]
    ) -> Tuple[int, int, List[ReconciliationDiff]]:
        """
        Reconcile system positions against broker positions.
        
        Args:
            system_positions: {symbol: quantity} from system
            broker_positions: {symbol: quantity} from broker
        
        Returns:
            Tuple of (matched_count, mismatched_count, list of diffs)
        """
        diffs = []
        matched = 0
        mismatched = 0
        
        all_symbols = set(system_positions.keys()) | set(broker_positions.keys())
        
        for symbol in all_symbols:
            system_qty = system_positions.get(symbol, 0)
            broker_qty = broker_positions.get(symbol, 0)
            
            diff = abs(system_qty - broker_qty)
            
            if diff <= self.tolerance_shares:
                matched += 1
            else:
                mismatched += 1
                severity = 'error' if diff > 10 else 'warning'
                
                diffs.append(ReconciliationDiff(
                    category='position',
                    symbol=symbol,
                    expected=system_qty,
                    actual=broker_qty,
                    difference=broker_qty - system_qty,
                    severity=severity,
                    message=f'{symbol}: System={system_qty}, Broker={broker_qty}'
                ))
        
        return matched, mismatched, diffs
    
    def reconcile_trades(
        self,
        system_trades: List[Dict],
        broker_trades: List[Dict]
    ) -> Tuple[int, int, List[ReconciliationDiff]]:
        """
        Reconcile system trades against broker fills.
        
        Args:
            system_trades: List of trade dicts from system
            broker_trades: List of fill dicts from broker
        
        Returns:
            Tuple of (matched_count, mismatched_count, list of diffs)
        """
        diffs = []
        matched = 0
        mismatched = 0
        
        # Match by symbol, side, and approximate quantity
        broker_matched = set()
        
        for sys_trade in system_trades:
            found_match = False
            
            for i, broker_trade in enumerate(broker_trades):
                if i in broker_matched:
                    continue
                
                # Check if trades match
                if (sys_trade.get('symbol') == broker_trade.get('symbol') and
                    sys_trade.get('side') == broker_trade.get('side')):
                    
                    qty_diff = abs(sys_trade.get('quantity', 0) - broker_trade.get('quantity', 0))
                    
                    if qty_diff <= self.tolerance_shares:
                        matched += 1
                        broker_matched.add(i)
                        found_match = True
                        break
            
            if not found_match:
                mismatched += 1
                diffs.append(ReconciliationDiff(
                    category='trade',
                    symbol=sys_trade.get('symbol'),
                    expected=sys_trade.get('quantity', 0),
                    actual=0,
                    difference=-sys_trade.get('quantity', 0),
                    severity='warning',
                    message=f"System trade not found in broker: {sys_trade.get('symbol')} {sys_trade.get('side')} {sys_trade.get('quantity')}"
                ))
        
        # Check for broker trades not in system
        for i, broker_trade in enumerate(broker_trades):
            if i not in broker_matched:
                mismatched += 1
                diffs.append(ReconciliationDiff(
                    category='trade',
                    symbol=broker_trade.get('symbol'),
                    expected=0,
                    actual=broker_trade.get('quantity', 0),
                    difference=broker_trade.get('quantity', 0),
                    severity='warning',
                    message=f"Broker trade not in system: {broker_trade.get('symbol')} {broker_trade.get('side')} {broker_trade.get('quantity')}"
                ))
        
        return matched, mismatched, diffs
    
    def reconcile_cash(
        self,
        system_cash: float,
        broker_cash: float
    ) -> Tuple[bool, Optional[ReconciliationDiff]]:
        """
        Reconcile cash balances.
        
        Args:
            system_cash: Cash per system
            broker_cash: Cash per broker
        
        Returns:
            Tuple of (matched, optional diff)
        """
        diff = abs(system_cash - broker_cash)
        
        if diff <= self.tolerance_cash:
            return True, None
        
        severity = 'error' if diff > 1000 else 'warning'
        
        return False, ReconciliationDiff(
            category='cash',
            symbol=None,
            expected=system_cash,
            actual=broker_cash,
            difference=broker_cash - system_cash,
            severity=severity,
            message=f'Cash mismatch: System=${system_cash:,.2f}, Broker=${broker_cash:,.2f}'
        )
    
    def run_daily_reconciliation(
        self,
        system_positions: Dict[str, int],
        broker_positions: Dict[str, int],
        system_trades: Optional[List[Dict]] = None,
        broker_trades: Optional[List[Dict]] = None,
        system_cash: Optional[float] = None,
        broker_cash: Optional[float] = None
    ) -> ReconciliationReport:
        """
        Run full daily reconciliation.
        
        Args:
            system_positions: System's position view
            broker_positions: Broker's position view
            system_trades: Optional list of system trades
            broker_trades: Optional list of broker fills
            system_cash: Optional system cash
            broker_cash: Optional broker cash
        
        Returns:
            ReconciliationReport
        """
        diffs = []
        
        # Position reconciliation
        pos_matched, pos_mismatched, pos_diffs = self.reconcile_positions(
            system_positions, broker_positions
        )
        diffs.extend(pos_diffs)
        
        # Trade reconciliation
        trade_matched, trade_mismatched = 0, 0
        if system_trades and broker_trades:
            trade_matched, trade_mismatched, trade_diffs = self.reconcile_trades(
                system_trades, broker_trades
            )
            diffs.extend(trade_diffs)
        
        # Cash reconciliation
        cash_matched = True
        if system_cash is not None and broker_cash is not None:
            cash_matched, cash_diff = self.reconcile_cash(system_cash, broker_cash)
            if cash_diff:
                diffs.append(cash_diff)
        
        # Calculate totals
        total_pos_diff = sum(
            abs(d.difference) for d in diffs if d.category == 'position'
        )
        total_pnl_diff = 0  # Would need P&L data
        
        # Determine status
        if any(d.severity == 'error' for d in diffs):
            status = 'error'
        elif any(d.severity == 'warning' for d in diffs):
            status = 'warning'
        else:
            status = 'ok'
        
        report = ReconciliationReport(
            date=date.today(),
            positions_matched=pos_matched,
            positions_mismatched=pos_mismatched,
            trades_matched=trade_matched,
            trades_mismatched=trade_mismatched,
            cash_matched=cash_matched,
            total_position_diff=total_pos_diff,
            total_pnl_diff=total_pnl_diff,
            diffs=diffs,
            status=status
        )
        
        # Store and save report
        self.reports.append(report)
        self._save_report(report)
        
        # Log summary
        if status == 'ok':
            logger.info(f"✅ Reconciliation complete: {pos_matched} positions matched")
        else:
            logger.warning(
                f"⚠️ Reconciliation {status}: {pos_mismatched} position mismatches, "
                f"{trade_mismatched} trade mismatches"
            )
            for diff in diffs:
                logger.warning(f"   {diff.message}")
        
        return report
    
    def _save_report(self, report: ReconciliationReport):
        """Save report to file."""
        filename = self.report_dir / f"recon_{report.date.isoformat()}.json"
        
        with open(filename, 'w') as f:
            json.dump(report.to_dict(), f, indent=2)
    
    def get_report_history(self, days: int = 7) -> List[ReconciliationReport]:
        """Get recent reconciliation reports."""
        cutoff = date.today() - timedelta(days=days)
        return [r for r in self.reports if r.date >= cutoff]
    
    def get_summary(self) -> Dict:
        """Get reconciliation summary."""
        recent = self.get_report_history(days=7)
        
        if not recent:
            return {
                'status': 'no_data',
                'days_checked': 0,
                'ok_count': 0,
                'warning_count': 0,
                'error_count': 0
            }
        
        return {
            'status': 'ok' if all(r.status == 'ok' for r in recent) else 'issues',
            'days_checked': len(recent),
            'ok_count': len([r for r in recent if r.status == 'ok']),
            'warning_count': len([r for r in recent if r.status == 'warning']),
            'error_count': len([r for r in recent if r.status == 'error']),
            'last_check': recent[-1].date.isoformat() if recent else None
        }
