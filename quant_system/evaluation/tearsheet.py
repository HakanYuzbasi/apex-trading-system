"""
quant_system/evaluation/tearsheet.py
================================================================================
INSTITUTIONAL TEARSHEET GENERATOR
================================================================================
"""

import numpy as np
import pandas as pd
import logging

logger = logging.getLogger(__name__)

class InstitutionalTearsheet:
    def __init__(self, risk_free_rate: float = 0.0, periods_per_year: int = 252 * 288):
        """Standard 5-min frequency expected: 252 days * 288 bars"""
        self.rf = risk_free_rate
        self.N = periods_per_year

    def generate_tearsheet(self, returns: np.ndarray, actions: np.ndarray, slippage_bps: float = 2.0):
        """
        Computes Vectorized Risk Adjusted Returns and Capacity constraints.
        Applies a realistic physical slippage penalty strictly on top of gross PnL.
        """
        if len(returns) == 0:
            logger.warning("Empty return array. Tearsheet aborted.")
            return

        slippage_penalty = slippage_bps / 10000.0
        net_returns = returns - (actions > 0.0) * slippage_penalty
        
        cum_ret = np.cumprod(1 + net_returns)
        peak = np.maximum.accumulate(cum_ret)
        drawdown = (cum_ret - peak) / peak
        mdd = np.min(drawdown)
        
        underwater = drawdown < 0
        underwater_duration = 0
        max_uw_duration = 0
        
        for uw in underwater:
            if uw:
                underwater_duration += 1
                if underwater_duration > max_uw_duration:
                    max_uw_duration = underwater_duration
            else:
                underwater_duration = 0
                
        ann_ret = np.mean(net_returns) * self.N
        ann_vol = np.std(net_returns) * np.sqrt(self.N)
        
        sharpe = (ann_ret - self.rf) / ann_vol if ann_vol > 0 else 0.0
        
        downside_rets = net_returns[net_returns < 0]
        downside_vol = np.std(downside_rets) * np.sqrt(self.N) if len(downside_rets) > 0 else 0.0
        sortino = (ann_ret - self.rf) / downside_vol if downside_vol > 0 else 0.0
        
        calmar = ann_ret / abs(mdd) if mdd < 0 else 0.0
        
        # Estimate Daily Turnover footprint based on typical 288 bar windows
        turnover = np.mean(actions) * 288
        
        report = (
            "\n" + "="*55 + "\n"
            " INSTITUTIONAL EVALUATION TEARSHEET \n"
            + "="*55 + "\n"
            f" Annualized Return:       {ann_ret:8.2%}\n"
            f" Annualized Volatility:   {ann_vol:8.2%}\n"
            f" Sharpe Ratio:            {sharpe:8.2f}\n"
            f" Sortino Ratio:           {sortino:8.2f}\n"
            f" Calmar Ratio:            {calmar:8.2f}\n"
            f" Maximum Drawdown (MDD):  {mdd:8.2%}\n"
            f" Max Underwater Duration: {max_uw_duration} periods\n"
            f" Estimated Daily Turnover:{turnover:8.2%}\n"
            + "="*55 + "\n"
        )
        print(report)
        return {
            'ann_ret': ann_ret, 'ann_vol': ann_vol, 'sharpe': sharpe, 
            'sortino': sortino, 'calmar': calmar, 'mdd': mdd, 
            'max_uw': max_uw_duration, 'turnover': turnover
        }
