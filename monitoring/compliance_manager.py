"""
monitoring/compliance_manager.py
COMPLIANCE & AUDIT TRAIL
- Trade logging
- Compliance checks
- Audit reports
- Regulatory compliance
"""

import json
import logging
import hashlib
from typing import Dict, Optional
from datetime import datetime
from pathlib import Path

from core.logging_config import setup_logging

logger = logging.getLogger(__name__)


class ComplianceManager:
    """
    Manage compliance and audit trail.
    
    Features:
    - Trade audit logging
    - Pre-trade compliance checks
    - Daily reports
    - Regulatory compliance
    - Immutable audit trail
    """
    
    def __init__(self, audit_dir: str = "audit_logs"):
        self.audit_dir = Path(audit_dir)
        self.audit_dir.mkdir(exist_ok=True, parents=True)
        
        self.trade_log = []
        self.compliance_checks = []
        self.violations = []
        
        logger.info("✅ Compliance Manager initialized")
        logger.info(f"   Audit logs: {self.audit_dir}")
    
    def pre_trade_check(
        self,
        symbol: str,
        side: str,
        quantity: int,
        price: float,
        portfolio_value: float,
        current_positions: Dict[str, int],
        config: Dict
    ) -> Dict:
        """
        Pre-trade compliance checks.
        
        Checks:
        1. Position size limits
        2. Concentration limits
        3. Risk limits
        4. Regulatory limits
        
        Args:
            symbol: Trading symbol
            side: 'BUY' or 'SELL'
            quantity: Order quantity
            price: Order price
            portfolio_value: Current portfolio value
            current_positions: Current positions
            config: Trading configuration
        
        Returns:
            {
                'approved': bool,
                'violations': list,
                'warnings': list,
                'check_id': str
            }
        """
        violations = []
        warnings = []
        
        notional = quantity * price
        
        # Check 1: Position size limit
        max_position_pct = config.get('max_position_pct', 0.02)
        if notional > portfolio_value * max_position_pct:
            violations.append(
                f"Position size ${notional:,.0f} exceeds limit "
                f"(${portfolio_value * max_position_pct:,.0f})"
            )
        
        # Check 2: Concentration limit
        # Calculate total exposure after this trade
        total_exposure = sum(abs(qty) * price for qty in current_positions.values())
        
        if side == 'BUY':
            new_exposure = total_exposure + notional
        else:
            new_exposure = total_exposure - notional
        
        max_exposure_pct = config.get('max_exposure_pct', 0.95)
        if new_exposure > portfolio_value * max_exposure_pct:
            violations.append(
                f"Total exposure ${new_exposure:,.0f} exceeds limit "
                f"(${portfolio_value * max_exposure_pct:,.0f})"
            )
        
        # Check 3: Symbol limits
        if symbol in current_positions:
            current_qty = current_positions[symbol]
            
            if side == 'BUY':
                new_qty = current_qty + quantity
            else:
                new_qty = current_qty - quantity
            
            max_shares = config.get('max_shares_per_symbol', 200)
            if abs(new_qty) > max_shares:
                violations.append(
                    f"Position in {symbol} would be {new_qty} shares "
                    f"(limit: {max_shares})"
                )
        
        # Check 4: Short selling restrictions
        if side == 'SELL' and symbol not in current_positions:
            if not config.get('allow_short_selling', False):
                violations.append(f"Short selling not allowed for {symbol}")
        
        # Check 5: Minimum price check
        min_price = config.get('min_stock_price', 5.0)
        if price < min_price:
            warnings.append(f"Stock price ${price:.2f} below minimum ${min_price:.2f}")
        
        # Determine approval
        approved = len(violations) == 0
        
        # Create check record
        check_record = {
            'timestamp': datetime.now().isoformat(),
            'symbol': symbol,
            'side': side,
            'quantity': quantity,
            'price': price,
            'notional': notional,
            'approved': approved,
            'violations': violations,
            'warnings': warnings,
            'check_id': self._generate_check_id()
        }
        
        self.compliance_checks.append(check_record)
        
        if not approved:
            self.violations.append(check_record)
            logger.warning(f"⚠️  Pre-trade check FAILED for {symbol}:")
            for v in violations:
                logger.warning(f"   - {v}")
        
        elif warnings:
            logger.info(f"⚠️  Pre-trade check PASSED with warnings for {symbol}:")
            for w in warnings:
                logger.info(f"   - {w}")
        
        return check_record
    
    def log_trade(
        self,
        trade: Dict,
        check_id: Optional[str] = None
    ) -> str:
        """
        Log trade to immutable audit trail.
        
        Args:
            trade: Trade details
            check_id: Optional compliance check ID
        
        Returns:
            Trade log ID
        """
        # Create audit record
        audit_record = {
            'log_id': self._generate_trade_id(),
            'timestamp': datetime.now().isoformat(),
            'check_id': check_id,
            'trade': trade,
            'hash': None  # Will be calculated
        }
        
        # Calculate hash for immutability
        audit_record['hash'] = self._calculate_hash(audit_record)
        
        self.trade_log.append(audit_record)
        
        # Write to file
        self._write_audit_log(audit_record)
        
        logger.debug(f"📝 Trade logged: {audit_record['log_id']}")
        
        return audit_record['log_id']
    
    def _generate_check_id(self) -> str:
        """Generate unique compliance check ID."""
        timestamp = datetime.now().strftime('%Y%m%d%H%M%S%f')
        return f"CHK-{timestamp}"
    
    def _generate_trade_id(self) -> str:
        """Generate unique trade log ID."""
        timestamp = datetime.now().strftime('%Y%m%d%H%M%S%f')
        return f"TRD-{timestamp}"
    
    def _calculate_hash(self, record: Dict) -> str:
        """Calculate hash for audit record."""
        # Create string representation (excluding hash field)
        record_copy = record.copy()
        record_copy.pop('hash', None)
        
        record_str = json.dumps(record_copy, sort_keys=True)
        
        # Calculate SHA-256 hash
        return hashlib.sha256(record_str.encode()).hexdigest()
    
    def _write_audit_log(self, record: Dict):
        """Write audit record to file."""
        date_str = datetime.now().strftime('%Y%m%d')
        log_file = self.audit_dir / f"audit_{date_str}.jsonl"
        
        with open(log_file, 'a') as f:
            f.write(json.dumps(record) + '\n')
    
    def verify_audit_trail(self, date: Optional[str] = None) -> Dict:
        """
        Verify integrity of audit trail.
        
        Args:
            date: Date to verify (YYYYMMDD), or None for today
        
        Returns:
            Verification results
        """
        if date is None:
            date = datetime.now().strftime('%Y%m%d')
        
        log_file = self.audit_dir / f"audit_{date}.jsonl"
        
        if not log_file.exists():
            return {
                'verified': True,
                'message': f"No audit log for {date}"
            }
        
        verified = True
        tampered_records = []
        
        with open(log_file, 'r') as f:
            for line_num, line in enumerate(f, 1):
                try:
                    record = json.loads(line)
                    
                    # Verify hash
                    original_hash = record.get('hash')
                    record_copy = record.copy()
                    record_copy.pop('hash', None)
                    
                    calculated_hash = self._calculate_hash(record_copy)
                    
                    if original_hash != calculated_hash:
                        verified = False
                        tampered_records.append({
                            'line': line_num,
                            'log_id': record.get('log_id'),
                            'original_hash': original_hash,
                            'calculated_hash': calculated_hash
                        })
                
                except json.JSONDecodeError:
                    verified = False
                    tampered_records.append({
                        'line': line_num,
                        'error': 'Invalid JSON'
                    })
        
        result = {
            'date': date,
            'verified': verified,
            'tampered_records': tampered_records
        }
        
        if verified:
            logger.info(f"✅ Audit trail verified for {date}")
        else:
            logger.error(f"❌ Audit trail verification FAILED for {date}")
            logger.error(f"   Tampered records: {len(tampered_records)}")
        
        return result
    
    def generate_daily_report(self, date: Optional[str] = None) -> str:
        """
        Generate daily compliance report.
        
        Args:
            date: Date for report (YYYY-MM-DD), or None for today
        
        Returns:
            Report text
        """
        if date is None:
            date = datetime.now().strftime('%Y-%m-%d')
        
        # Filter records for date
        date_checks = [
            c for c in self.compliance_checks
            if c['timestamp'].startswith(date)
        ]
        
        date_trades = [
            t for t in self.trade_log
            if t['timestamp'].startswith(date)
        ]
        
        date_violations = [
            v for v in self.violations
            if v['timestamp'].startswith(date)
        ]
        
        # Generate report
        report = f"""
{'='*80}
APEX TRADING SYSTEM - DAILY COMPLIANCE REPORT
{'='*80}
Date: {date}
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

SUMMARY
-------
Total Compliance Checks: {len(date_checks)}
Approved Checks: {sum(1 for c in date_checks if c['approved'])}
Rejected Checks: {sum(1 for c in date_checks if not c['approved'])}
Total Trades Executed: {len(date_trades)}
Compliance Violations: {len(date_violations)}

"""
        
        if date_violations:
            report += "\nCOMPLIANCE VIOLATIONS\n"
            report += "-" * 80 + "\n"
            for v in date_violations:
                report += f"\nTime: {v['timestamp']}\n"
                report += f"Symbol: {v['symbol']}\n"
                report += f"Side: {v['side']}, Quantity: {v['quantity']}\n"
                report += "Violations:\n"
                for violation in v['violations']:
                    report += f"  - {violation}\n"
        
        else:
            report += "\n✅ No compliance violations detected.\n"
        
        report += "\n" + "="*80 + "\n"
        report += "End of Report\n"
        report += "="*80 + "\n"
        
        # Save report
        report_file = self.audit_dir / f"compliance_report_{date}.txt"
        with open(report_file, 'w') as f:
            f.write(report)
        
        logger.info(f"📄 Compliance report generated: {report_file}")
        
        return report
    
    def get_statistics(self) -> Dict:
        """Get compliance statistics."""
        if not self.compliance_checks:
            return {}
        
        total_checks = len(self.compliance_checks)
        approved = sum(1 for c in self.compliance_checks if c['approved'])
        rejected = total_checks - approved
        
        stats = {
            'total_checks': total_checks,
            'approved': approved,
            'rejected': rejected,
            'approval_rate': approved / total_checks if total_checks > 0 else 0,
            'total_violations': len(self.violations),
            'total_trades_logged': len(self.trade_log)
        }
        
        logger.info("📊 Compliance Statistics:")
        logger.info(f"   Total Checks: {stats['total_checks']}")
        logger.info(f"   Approval Rate: {stats['approval_rate']*100:.1f}%")
        logger.info(f"   Violations: {stats['total_violations']}")
        
        return stats


if __name__ == "__main__":
    # Test compliance manager
    setup_logging(level="INFO", log_file=None, json_format=False, console_output=True)
    
    manager = ComplianceManager()
    
    print("\n" + "="*60)
    print("TEST 1: PRE-TRADE CHECKS")
    print("="*60)
    
    config = {
        'max_position_pct': 0.02,
        'max_exposure_pct': 0.95,
        'max_shares_per_symbol': 200,
        'allow_short_selling': False,
        'min_stock_price': 5.0
    }
    
    # Test 1: Valid trade
    check1 = manager.pre_trade_check(
        symbol='AAPL',
        side='BUY',
        quantity=100,
        price=180.0,
        portfolio_value=1_000_000,
        current_positions={},
        config=config
    )
    print(f"Valid trade: {'✅ APPROVED' if check1['approved'] else '❌ REJECTED'}")
    
    # Test 2: Position too large
    check2 = manager.pre_trade_check(
        symbol='AAPL',
        side='BUY',
        quantity=1000,
        price=180.0,
        portfolio_value=100_000,
        current_positions={},
        config=config
    )
    print(f"Large position: {'✅ APPROVED' if check2['approved'] else '❌ REJECTED'}")
    
    # Test 3: Illegal short
    check3 = manager.pre_trade_check(
        symbol='AAPL',
        side='SELL',
        quantity=100,
        price=180.0,
        portfolio_value=1_000_000,
        current_positions={},
        config=config
    )
    print(f"Illegal short: {'✅ APPROVED' if check3['approved'] else '❌ REJECTED'}")
    
    print("\n" + "="*60)
    print("TEST 2: TRADE LOGGING")
    print("="*60)
    
    trade = {
        'symbol': 'AAPL',
        'side': 'BUY',
        'quantity': 100,
        'price': 180.0,
        'timestamp': datetime.now().isoformat()
    }
    
    log_id = manager.log_trade(trade, check_id=check1['check_id'])
    print(f"Trade logged: {log_id}")
    
    print("\n" + "="*60)
    print("TEST 3: AUDIT TRAIL VERIFICATION")
    print("="*60)
    
    verification = manager.verify_audit_trail()
    print(f"Audit trail: {'✅ VERIFIED' if verification['verified'] else '❌ TAMPERED'}")
    
    print("\n" + "="*60)
    print("TEST 4: DAILY REPORT")
    print("="*60)
    
    report = manager.generate_daily_report()
    print(report[:500] + "...")
    
    print("\n" + "="*60)
    print("TEST 5: STATISTICS")
    print("="*60)
    
    stats = manager.get_statistics()
    
    print("\n✅ Compliance manager tests complete!")
