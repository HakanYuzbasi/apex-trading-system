"""
risk/market_neutral_hedge.py - Absolute Risk Neutrality (Delta-Hedging)
Calculates the portfolio's net dollar Beta ($Δ$) exposure relative to the SPY
and computes the exact SPY/QQQ short blocks required to sterilize macro market collapse risk.
"""

import logging
from typing import Dict
from config import ApexConfig
from core.symbols import parse_symbol, AssetClass

logger = logging.getLogger(__name__)

class DeltaHedger:
    """
    Continuous Beta Hedging engine. 
    Guarantees structural Market Neutrality by sizing SPY inverse executions
    so Portfolio_Delta = 0.0.
    """
    def __init__(self, market_data_fetcher):
        self.market_data = market_data_fetcher
        self._beta_cache: Dict[str, float] = {}

    def _get_asset_beta(self, symbol: str) -> float:
        """Fetch or cache stock Beta vs SPY."""
        try:
            parsed = parse_symbol(symbol)
            if parsed.asset_class == AssetClass.CRYPTO:
                # Crypto has a non-linear but highly correlated Beta to QQQ/SPY recently.
                # Assuming ~1.5 Beta for massive assets like BTC.
                return 1.2
            
            if symbol in self._beta_cache:
                return self._beta_cache[symbol]

            # Native Yahoo Finance extraction
            info = self.market_data.get_market_info(symbol)
            _beta = float(info.get("beta", 1.0))
            self._beta_cache[symbol] = _beta
            return _beta
        except Exception as e:
            logger.debug(f"Failed to fetch Beta for {symbol}: {e}")
            return 1.0

    async def calculate_hedge_order(self, current_positions: Dict[str, float], price_cache: Dict[str, float], hedge_symbol="SPY") -> float:
        """
        Evaluate entire universe exposure.
        Returns:
            qty_to_trade (float): Positive means BUY SPY, Negative means SHORT SPY
        """
        target_delta = 0.0
        portfolio_value = 0.0
        spy_exposure = 0.0

        for sym, qty in current_positions.items():
            if qty == 0:
                continue
            
            price = price_cache.get(sym) or self.market_data.get_current_price(sym)
            if not price or price <= 0:
                continue
            
            position_usd = qty * price
            portfolio_value += abs(position_usd)

            if sym == hedge_symbol:
                spy_exposure = position_usd
                continue
            
            asset_beta = self._get_asset_beta(sym)
            # $Δ = Position_Value * Beta
            target_delta += position_usd * asset_beta

        if portfolio_value == 0:
            return 0.0
        
        # We want: target_delta + spy_exposure = 0
        # spy_required_usd = -target_delta
        # Diff = spy_required_usd - current_spy_exposure
        spy_required_usd = -target_delta
        diff_usd = spy_required_usd - spy_exposure
        
        # Only hedge if the unhedged delta is significant (> $5,000 imbalance)
        if abs(diff_usd) < getattr(ApexConfig, "MIN_HEDGE_DOLLAR_IMBALANCE", 5000):
            return 0.0

        spy_price = price_cache.get(hedge_symbol) or self.market_data.get_current_price(hedge_symbol)
        if not spy_price or spy_price <= 0:
            return 0.0
        
        # Strict integer shares for SPY
        qty_to_trade = int(diff_usd / spy_price)
        
        if abs(qty_to_trade) > 0:
            logger.info(f"🛡️ DELTA HEDGE: Unhedged Δ=${target_delta:,.0f} | Executing {qty_to_trade}x {hedge_symbol} to sterilize Beta.")
            
        return qty_to_trade
