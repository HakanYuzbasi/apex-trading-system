"""
monitoring/data_quality.py - Data Quality Monitoring

Systematic checks for data quality issues:
- Stale data detection
- Missing values
- Outlier detection
- Corporate action adjustments
- Survivorship bias checks
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class DataQualityIssue:
    """Represents a data quality issue."""
    symbol: str
    issue_type: str
    severity: str  # 'warning', 'error', 'critical'
    message: str
    detected_at: datetime
    
    def to_dict(self) -> Dict:
        return {
            'symbol': self.symbol,
            'issue_type': self.issue_type,
            'severity': self.severity,
            'message': self.message,
            'detected_at': self.detected_at.isoformat()
        }


class DataQualityMonitor:
    """
    Monitor data quality for trading system.
    
    Checks:
    - Stale prices (no updates)
    - Missing data
    - Price gaps (potential splits/dividends)
    - Outlier returns
    - Data completeness
    """
    
    def __init__(
        self,
        stale_threshold_minutes: int = 30,
        outlier_std: float = 5.0,
        min_history_days: int = 60
    ):
        """
        Initialize data quality monitor.
        
        Args:
            stale_threshold_minutes: Minutes before price is considered stale
            outlier_std: Standard deviations for outlier detection
            min_history_days: Minimum days of history required
        """
        self.stale_threshold_minutes = stale_threshold_minutes
        self.outlier_std = outlier_std
        self.min_history_days = min_history_days
        
        self.issues: List[DataQualityIssue] = []
        self.last_prices: Dict[str, Tuple[float, datetime]] = {}
        
        logger.info("DataQualityMonitor initialized")
    
    def update_price(self, symbol: str, price: float):
        """Update last known price for a symbol."""
        self.last_prices[symbol] = (price, datetime.now())
    
    def check_staleness(self, symbol: str) -> Optional[DataQualityIssue]:
        """Check if price data is stale."""
        if symbol not in self.last_prices:
            return DataQualityIssue(
                symbol=symbol,
                issue_type='no_data',
                severity='warning',
                message='No price data available',
                detected_at=datetime.now()
            )
        
        price, last_update = self.last_prices[symbol]
        minutes_since = (datetime.now() - last_update).total_seconds() / 60
        
        if minutes_since > self.stale_threshold_minutes:
            return DataQualityIssue(
                symbol=symbol,
                issue_type='stale_data',
                severity='warning',
                message=f'Price is {minutes_since:.0f} minutes old',
                detected_at=datetime.now()
            )
        
        return None
    
    def check_price_gap(
        self,
        symbol: str,
        prices: pd.Series,
        gap_threshold: float = 0.20
    ) -> Optional[DataQualityIssue]:
        """
        Check for suspicious price gaps (potential splits/dividends).
        
        Args:
            symbol: Stock ticker
            prices: Price series
            gap_threshold: Threshold for suspicious gap (20% default)
        
        Returns:
            DataQualityIssue if gap detected
        """
        if len(prices) < 2:
            return None
        
        returns = prices.pct_change().dropna()
        
        # Check last return for large gap
        if len(returns) > 0:
            last_return = abs(returns.iloc[-1])
            if last_return > gap_threshold:
                return DataQualityIssue(
                    symbol=symbol,
                    issue_type='price_gap',
                    severity='warning',
                    message=f'Large price change: {last_return*100:.1f}% (possible split/dividend)',
                    detected_at=datetime.now()
                )
        
        return None
    
    def check_outliers(
        self,
        symbol: str,
        prices: pd.Series
    ) -> Optional[DataQualityIssue]:
        """
        Check for outlier prices.
        
        Args:
            symbol: Stock ticker
            prices: Price series
        
        Returns:
            DataQualityIssue if outlier detected
        """
        if len(prices) < 20:
            return None
        
        returns = prices.pct_change().dropna()
        
        mean_ret = returns.mean()
        std_ret = returns.std()
        
        if std_ret == 0:
            return None
        
        # Check last return
        last_return = returns.iloc[-1]
        z_score = abs((last_return - mean_ret) / std_ret)
        
        if z_score > self.outlier_std:
            return DataQualityIssue(
                symbol=symbol,
                issue_type='outlier',
                severity='warning',
                message=f'Outlier return: {last_return*100:.1f}% (z-score: {z_score:.1f})',
                detected_at=datetime.now()
            )
        
        return None
    
    def check_data_completeness(
        self,
        symbol: str,
        data: pd.DataFrame
    ) -> Optional[DataQualityIssue]:
        """
        Check if data is complete.
        
        Args:
            symbol: Stock ticker
            data: DataFrame with price data
        
        Returns:
            DataQualityIssue if data incomplete
        """
        if len(data) < self.min_history_days:
            return DataQualityIssue(
                symbol=symbol,
                issue_type='insufficient_history',
                severity='warning',
                message=f'Only {len(data)} days of history (need {self.min_history_days})',
                detected_at=datetime.now()
            )
        
        # Check for NaN values
        if 'Close' in data.columns:
            nan_count = data['Close'].isna().sum()
            if nan_count > 0:
                nan_pct = nan_count / len(data) * 100
                return DataQualityIssue(
                    symbol=symbol,
                    issue_type='missing_values',
                    severity='warning' if nan_pct < 5 else 'error',
                    message=f'{nan_count} missing values ({nan_pct:.1f}%)',
                    detected_at=datetime.now()
                )
        
        return None
    
    def run_all_checks(
        self,
        symbol: str,
        data: Optional[pd.DataFrame] = None,
        prices: Optional[pd.Series] = None
    ) -> List[DataQualityIssue]:
        """
        Run all data quality checks for a symbol.
        
        Args:
            symbol: Stock ticker
            data: Full DataFrame (optional)
            prices: Price series (optional)
        
        Returns:
            List of issues found
        """
        issues = []
        
        # Staleness check
        issue = self.check_staleness(symbol)
        if issue:
            issues.append(issue)
        
        # Price gap check
        if prices is not None:
            issue = self.check_price_gap(symbol, prices)
            if issue:
                issues.append(issue)
            
            issue = self.check_outliers(symbol, prices)
            if issue:
                issues.append(issue)
        
        # Completeness check
        if data is not None:
            issue = self.check_data_completeness(symbol, data)
            if issue:
                issues.append(issue)
        
        # Store issues
        self.issues.extend(issues)
        
        # Log errors/criticals
        for issue in issues:
            if issue.severity in ['error', 'critical']:
                logger.error(f"Data quality {issue.severity}: {issue.symbol} - {issue.message}")
            else:
                logger.debug(f"Data quality {issue.severity}: {issue.symbol} - {issue.message}")
        
        return issues
    
    def get_recent_issues(
        self,
        minutes: int = 60,
        severity: Optional[str] = None
    ) -> List[DataQualityIssue]:
        """
        Get recent data quality issues.
        
        Args:
            minutes: How far back to look
            severity: Filter by severity (optional)
        
        Returns:
            List of issues
        """
        cutoff = datetime.now() - timedelta(minutes=minutes)
        filtered = [i for i in self.issues if i.detected_at >= cutoff]
        
        if severity:
            filtered = [i for i in filtered if i.severity == severity]
        
        return filtered
    
    def clear_old_issues(self, hours: int = 24):
        """Clear issues older than specified hours."""
        cutoff = datetime.now() - timedelta(hours=hours)
        self.issues = [i for i in self.issues if i.detected_at >= cutoff]
    
    def get_summary(self) -> Dict:
        """Get summary of data quality status."""
        recent = self.get_recent_issues(minutes=60)
        
        return {
            'total_issues_1h': len(recent),
            'warnings': len([i for i in recent if i.severity == 'warning']),
            'errors': len([i for i in recent if i.severity == 'error']),
            'criticals': len([i for i in recent if i.severity == 'critical']),
            'symbols_affected': len(set(i.symbol for i in recent)),
            'issue_types': list(set(i.issue_type for i in recent))
        }
