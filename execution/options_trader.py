"""
execution/options_trader.py - Options Trading Module

Implements options strategies for:
- Hedging existing positions (protective puts)
- Income generation (covered calls)
- Volatility trading (straddles, strangles)
- Directional bets (long calls/puts)

Integrates with IBKR for options execution.

Usage:
    trader = OptionsTrader(ibkr_connector)
    await trader.buy_protective_put("AAPL", 100, days_to_expiry=30)
    await trader.sell_covered_call("AAPL", 100, delta=0.30)
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from enum import Enum
import math

# Allow nested event loops for ib_insync compatibility
try:
    import nest_asyncio
    nest_asyncio.apply()
except ImportError:
    pass

logger = logging.getLogger(__name__)

# Check for scipy (for Black-Scholes)
try:
    from scipy.stats import norm
    from scipy.optimize import brentq
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    logger.warning("scipy not installed. Options pricing limited. Install with: pip install scipy")


class OptionType(Enum):
    """Option type."""
    CALL = "call"
    PUT = "put"


class OptionStrategy(Enum):
    """Common options strategies."""
    LONG_CALL = "long_call"
    LONG_PUT = "long_put"
    COVERED_CALL = "covered_call"
    PROTECTIVE_PUT = "protective_put"
    BULL_CALL_SPREAD = "bull_call_spread"
    BEAR_PUT_SPREAD = "bear_put_spread"
    STRADDLE = "straddle"
    STRANGLE = "strangle"
    IRON_CONDOR = "iron_condor"
    COLLAR = "collar"


@dataclass
class OptionContract:
    """Represents an options contract."""
    symbol: str
    option_type: OptionType
    strike: float
    expiry: datetime
    multiplier: int = 100
    trading_class: str = ""

    # Greeks (populated after pricing)
    delta: float = 0.0
    gamma: float = 0.0
    theta: float = 0.0
    vega: float = 0.0
    rho: float = 0.0

    # Pricing
    theoretical_price: float = 0.0
    bid: float = 0.0
    ask: float = 0.0
    last: float = 0.0
    implied_vol: float = 0.0

    @property
    def days_to_expiry(self) -> int:
        return (self.expiry - datetime.now()).days

    @property
    def time_to_expiry(self) -> float:
        """Time to expiry in years."""
        return max(0, (self.expiry - datetime.now()).days / 365.0)

    @property
    def mid_price(self) -> float:
        if self.bid > 0 and self.ask > 0:
            return (self.bid + self.ask) / 2
        return self.last or self.theoretical_price

    def to_dict(self) -> dict:
        return {
            'symbol': self.symbol,
            'type': self.option_type.value,
            'strike': self.strike,
            'expiry': self.expiry.isoformat(),
            'days_to_expiry': self.days_to_expiry,
            'delta': self.delta,
            'gamma': self.gamma,
            'theta': self.theta,
            'vega': self.vega,
            'implied_vol': self.implied_vol,
            'mid_price': self.mid_price,
            'trading_class': self.trading_class,
            'multiplier': self.multiplier
        }


@dataclass
class OptionPosition:
    """An open options position."""
    contract: OptionContract
    quantity: int  # Positive = long, negative = short
    entry_price: float
    entry_date: datetime

    @property
    def is_long(self) -> bool:
        return self.quantity > 0

    @property
    def notional_value(self) -> float:
        return abs(self.quantity) * self.contract.strike * self.contract.multiplier

    @property
    def current_value(self) -> float:
        return abs(self.quantity) * self.contract.mid_price * self.contract.multiplier

    @property
    def pnl(self) -> float:
        return (self.contract.mid_price - self.entry_price) * self.quantity * self.contract.multiplier


class BlackScholes:
    """Black-Scholes options pricing model."""

    @staticmethod
    def price(
        S: float,  # Spot price
        K: float,  # Strike price
        T: float,  # Time to expiry (years)
        r: float,  # Risk-free rate
        sigma: float,  # Volatility
        option_type: OptionType
    ) -> float:
        """
        Calculate option price using Black-Scholes.

        Args:
            S: Current stock price
            K: Strike price
            T: Time to expiration in years
            r: Risk-free interest rate
            sigma: Volatility (annualized)
            option_type: CALL or PUT

        Returns:
            Option price
        """
        if not SCIPY_AVAILABLE:
            # Simple approximation without scipy
            intrinsic = max(0, S - K) if option_type == OptionType.CALL else max(0, K - S)
            time_value = S * sigma * math.sqrt(T) * 0.4  # Rough approximation
            return intrinsic + time_value

        if T <= 0:
            # At expiration
            if option_type == OptionType.CALL:
                return max(0, S - K)
            else:
                return max(0, K - S)

        d1 = (math.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * math.sqrt(T))
        d2 = d1 - sigma * math.sqrt(T)

        if option_type == OptionType.CALL:
            price = S * norm.cdf(d1) - K * math.exp(-r * T) * norm.cdf(d2)
        else:
            price = K * math.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)

        return price

    @staticmethod
    def delta(
        S: float, K: float, T: float, r: float, sigma: float,
        option_type: OptionType
    ) -> float:
        """Calculate option delta."""
        if not SCIPY_AVAILABLE or T <= 0:
            return 1.0 if option_type == OptionType.CALL else -1.0

        d1 = (math.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * math.sqrt(T))

        if option_type == OptionType.CALL:
            return norm.cdf(d1)
        else:
            return norm.cdf(d1) - 1

    @staticmethod
    def gamma(S: float, K: float, T: float, r: float, sigma: float) -> float:
        """Calculate option gamma."""
        if not SCIPY_AVAILABLE or T <= 0:
            return 0.0

        d1 = (math.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * math.sqrt(T))
        return norm.pdf(d1) / (S * sigma * math.sqrt(T))

    @staticmethod
    def theta(
        S: float, K: float, T: float, r: float, sigma: float,
        option_type: OptionType
    ) -> float:
        """Calculate option theta (per day)."""
        if not SCIPY_AVAILABLE or T <= 0:
            return 0.0

        d1 = (math.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * math.sqrt(T))
        d2 = d1 - sigma * math.sqrt(T)

        term1 = -(S * norm.pdf(d1) * sigma) / (2 * math.sqrt(T))

        if option_type == OptionType.CALL:
            term2 = -r * K * math.exp(-r * T) * norm.cdf(d2)
        else:
            term2 = r * K * math.exp(-r * T) * norm.cdf(-d2)

        return (term1 + term2) / 365  # Per day

    @staticmethod
    def vega(S: float, K: float, T: float, r: float, sigma: float) -> float:
        """Calculate option vega (per 1% vol change)."""
        if not SCIPY_AVAILABLE or T <= 0:
            return 0.0

        d1 = (math.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * math.sqrt(T))
        return S * norm.pdf(d1) * math.sqrt(T) / 100  # Per 1% change

    @staticmethod
    def implied_volatility(
        market_price: float,
        S: float, K: float, T: float, r: float,
        option_type: OptionType
    ) -> float:
        """Calculate implied volatility from market price."""
        if not SCIPY_AVAILABLE:
            return 0.25  # Default 25%

        def objective(sigma):
            return BlackScholes.price(S, K, T, r, sigma, option_type) - market_price

        try:
            iv = brentq(objective, 0.01, 5.0)
            return iv
        except Exception:
            return 0.25  # Default if can't solve


class OptionsTrader:
    """
    Options trading interface integrated with IBKR.

    Features:
    - Options chain retrieval
    - Greeks calculation
    - Strategy execution
    - Position management
    - Hedging recommendations
    """

    def __init__(
        self,
        ibkr_connector,
        risk_free_rate: float = 0.05,
        default_vol: float = 0.25
    ):
        """
        Initialize options trader.

        Args:
            ibkr_connector: IBKR connector instance
            risk_free_rate: Risk-free rate for pricing
            default_vol: Default volatility assumption
        """
        self.ibkr = ibkr_connector
        self.risk_free_rate = risk_free_rate
        self.default_vol = default_vol

        # Track options positions
        self.positions: Dict[str, OptionPosition] = {}

        # Options chain cache
        self._chain_cache: Dict[str, List[OptionContract]] = {}
        self._cache_expiry: Dict[str, datetime] = {}

        logger.info("🎯 Options Trader initialized")
        logger.info(f"   Risk-free rate: {risk_free_rate:.1%}")
        logger.info(f"   scipy available: {SCIPY_AVAILABLE}")

    async def get_options_chain(
        self,
        symbol: str,
        expiry: datetime = None,
        min_days: int = 7,
        max_days: int = 60
    ) -> List[OptionContract]:
        """
        Get options chain for a symbol.

        Args:
            symbol: Underlying symbol
            expiry: Specific expiry (auto-select if None)
            min_days: Minimum days to expiry
            max_days: Maximum days to expiry

        Returns:
            List of OptionContract objects
        """
        cache_key = f"{symbol}_{min_days}_{max_days}"

        # Check cache
        if cache_key in self._chain_cache:
            if datetime.now() < self._cache_expiry.get(cache_key, datetime.min):
                return self._chain_cache[cache_key]

        try:
            # Get current price
            price = await self.ibkr.get_market_price(symbol)
            if price == 0:
                logger.warning(f"Cannot get price for {symbol}")
                return []

            # Get options chain from IBKR
            # Note: Requires ib_insync options support
            contract = await self.ibkr.get_contract(symbol)
            if not contract:
                return []

            # Request option chains
            chains = self.ibkr.ib.reqSecDefOptParams(
                contract.symbol,
                '',
                contract.secType,
                contract.conId
            )

            await asyncio.sleep(1)  # Wait for response

            if not chains:
                logger.warning(f"No options chains for {symbol}")
                return []

            # Build contracts list
            options = []
            now = datetime.now()

            # Prefer SMART-routable chains; fall back to all if none match
            smart_chains = [c for c in chains if getattr(c, 'exchange', '') == 'SMART']
            selected_chains = smart_chains if smart_chains else chains

            for chain in selected_chains:
                for exp_str in chain.expirations[:5]:  # Limit to 5 nearest expiries
                    try:
                        expiry_date = datetime.strptime(exp_str, '%Y%m%d')
                        days_to_exp = (expiry_date - now).days

                        if min_days <= days_to_exp <= max_days:
                            # Get strikes near the money
                            strikes = [s for s in chain.strikes
                                       if 0.85 * price <= s <= 1.15 * price]

                            for strike in strikes:
                                for opt_type in [OptionType.CALL, OptionType.PUT]:
                                    opt = OptionContract(
                                        symbol=symbol,
                                        option_type=opt_type,
                                        strike=strike,
                                        expiry=expiry_date,
                                        trading_class=getattr(chain, 'tradingClass', ''),
                                        multiplier=int(getattr(chain, 'multiplier', '100'))
                                    )

                                    # Calculate Greeks
                                    self._calculate_greeks(opt, price)
                                    options.append(opt)

                    except Exception as e:
                        logger.debug(f"Error parsing expiry {exp_str}: {e}")

            # Cache results
            self._chain_cache[cache_key] = options
            self._cache_expiry[cache_key] = datetime.now() + timedelta(minutes=5)

            logger.info(f"Retrieved {len(options)} options for {symbol}")
            return options

        except Exception as e:
            logger.error(f"Error getting options chain for {symbol}: {e}")
            return []

    def _calculate_greeks(self, option: OptionContract, spot_price: float):
        """Calculate Greeks for an option."""
        T = option.time_to_expiry
        if T <= 0:
            return

        vol = option.implied_vol if option.implied_vol > 0 else self.default_vol

        option.theoretical_price = BlackScholes.price(
            spot_price, option.strike, T, self.risk_free_rate, vol, option.option_type
        )
        option.delta = BlackScholes.delta(
            spot_price, option.strike, T, self.risk_free_rate, vol, option.option_type
        )
        option.gamma = BlackScholes.gamma(
            spot_price, option.strike, T, self.risk_free_rate, vol
        )
        option.theta = BlackScholes.theta(
            spot_price, option.strike, T, self.risk_free_rate, vol, option.option_type
        )
        option.vega = BlackScholes.vega(
            spot_price, option.strike, T, self.risk_free_rate, vol
        )

    def select_option_by_delta(
        self,
        chain: List[OptionContract],
        target_delta: float,
        option_type: OptionType
    ) -> Optional[OptionContract]:
        """
        Select option closest to target delta.

        Args:
            chain: Options chain
            target_delta: Target delta (e.g., 0.30 for 30-delta)
            option_type: CALL or PUT

        Returns:
            Best matching option or None
        """
        candidates = [o for o in chain if o.option_type == option_type]

        if not candidates:
            return None

        # For puts, delta is negative, so use absolute value
        if option_type == OptionType.PUT:
            target_delta = -abs(target_delta)

        # Find closest to target delta
        best = min(candidates, key=lambda o: abs(o.delta - target_delta))
        return best

    def select_option_by_strike(
        self,
        chain: List[OptionContract],
        target_strike: float,
        option_type: OptionType,
        expiry: datetime = None
    ) -> Optional[OptionContract]:
        """Select option by strike price."""
        candidates = [o for o in chain if o.option_type == option_type]

        if expiry:
            candidates = [o for o in candidates if o.expiry.date() == expiry.date()]

        if not candidates:
            return None

        best = min(candidates, key=lambda o: abs(o.strike - target_strike))
        return best

    async def buy_protective_put(
        self,
        symbol: str,
        shares: int,
        delta: float = -0.30,
        days_to_expiry: int = 30
    ) -> Optional[Dict]:
        """
        Buy protective puts for an existing long position.

        Args:
            symbol: Underlying symbol
            shares: Number of shares to protect
            delta: Target delta (negative, e.g., -0.30)
            days_to_expiry: Target days to expiration

        Returns:
            Trade result or None
        """
        logger.info(f"🛡️ Buying protective put for {shares} {symbol}")

        # Get options chain
        chain = await self.get_options_chain(
            symbol,
            min_days=days_to_expiry - 7,
            max_days=days_to_expiry + 14
        )

        if not chain:
            logger.error(f"No options available for {symbol}")
            return None

        # Select put by delta
        put = self.select_option_by_delta(chain, delta, OptionType.PUT)
        if not put:
            logger.error(f"No suitable put found for {symbol}")
            return None

        # Calculate contracts needed (1 contract = 100 shares)
        contracts = shares // put.multiplier

        logger.info(f"   Selected: {put.strike} put, exp {put.expiry.date()}")
        logger.info(f"   Delta: {put.delta:.2f}, Price: ${put.mid_price:.2f}")
        logger.info(f"   Contracts: {contracts}")

        # Execute actual order via IBKR
        expiry_str = put.expiry.strftime('%Y%m%d')
        right = 'P'
        
        trade_result = await self.ibkr.execute_option_order(
            symbol=symbol,
            expiry=expiry_str,
            strike=put.strike,
            right=right,
            side='BUY',
            quantity=contracts,
            order_type='MKT',
            trading_class=put.trading_class,
            multiplier=put.multiplier
        )
        
        if not trade_result:
            logger.error(f"Failed to execute protective put for {symbol}")
            return None

        # Record position
        pos_key = f"{symbol}_{expiry_str}_{put.strike}_{right}"
        self.positions[pos_key] = OptionPosition(
            contract=put,
            quantity=contracts,
            entry_price=trade_result.get('price', put.mid_price),
            entry_date=datetime.now()
        )

        return {
            'strategy': OptionStrategy.PROTECTIVE_PUT.value,
            'symbol': symbol,
            'contract': put.to_dict(),
            'quantity': contracts,
            'cost': trade_result.get('price', 0) * contracts * put.multiplier,
            'order_id': trade_result.get('order_id')
        }

    async def sell_covered_call(
        self,
        symbol: str,
        shares: int,
        delta: float = 0.30,
        days_to_expiry: int = 30
    ) -> Optional[Dict]:
        """
        Sell covered calls against an existing long position.

        Args:
            symbol: Underlying symbol
            shares: Number of shares owned
            delta: Target delta (e.g., 0.30 for 30-delta)
            days_to_expiry: Target days to expiration

        Returns:
            Trade result or None
        """
        logger.info(f"💰 Selling covered call on {shares} {symbol}")

        # Get options chain
        chain = await self.get_options_chain(
            symbol,
            min_days=days_to_expiry - 7,
            max_days=days_to_expiry + 14
        )

        if not chain:
            logger.error(f"No options available for {symbol}")
            return None

        # Select call by delta
        call = self.select_option_by_delta(chain, delta, OptionType.CALL)
        if not call:
            logger.error(f"No suitable call found for {symbol}")
            return None

        # Calculate contracts (1 contract per 100 shares)
        contracts = shares // call.multiplier

        logger.info(f"   Selected: {call.strike} call, exp {call.expiry.date()}")
        logger.info(f"   Delta: {call.delta:.2f}, Premium: ${call.mid_price:.2f}")
        logger.info(f"   Contracts: {contracts}")

        # Execute actual order via IBKR
        expiry_str = call.expiry.strftime('%Y%m%d')
        right = 'C'
        
        trade_result = await self.ibkr.execute_option_order(
            symbol=symbol,
            expiry=expiry_str,
            strike=call.strike,
            right=right,
            side='SELL',
            quantity=contracts,
            order_type='MKT',
            trading_class=call.trading_class,
            multiplier=call.multiplier
        )
        
        if not trade_result:
            logger.error(f"Failed to execute covered call for {symbol}")
            return None

        # Record position
        pos_key = f"{symbol}_{expiry_str}_{call.strike}_{right}"
        self.positions[pos_key] = OptionPosition(
            contract=call,
            quantity=-contracts,
            entry_price=trade_result.get('price', call.mid_price),
            entry_date=datetime.now()
        )

        return {
            'strategy': OptionStrategy.COVERED_CALL.value,
            'symbol': symbol,
            'contract': call.to_dict(),
            'quantity': -contracts,  # Negative = short
            'premium': trade_result.get('price', 0) * contracts * call.multiplier,
            'order_id': trade_result.get('order_id')
        }

    async def buy_straddle(
        self,
        symbol: str,
        contracts: int = 1,
        days_to_expiry: int = 30
    ) -> Optional[Dict]:
        """
        Buy a straddle (long call + long put at same strike).

        Used for volatility plays when direction is uncertain.

        Args:
            symbol: Underlying symbol
            contracts: Number of contracts
            days_to_expiry: Target days to expiration

        Returns:
            Trade result or None
        """
        logger.info(f"📈📉 Buying straddle on {symbol}")

        # Get current price for ATM strike
        price = await self.ibkr.get_market_price(symbol)
        if price == 0:
            return None

        # Get options chain
        chain = await self.get_options_chain(
            symbol,
            min_days=days_to_expiry - 7,
            max_days=days_to_expiry + 14
        )

        if not chain:
            return None

        # Find ATM options
        atm_call = self.select_option_by_strike(chain, price, OptionType.CALL)
        atm_put = self.select_option_by_strike(chain, price, OptionType.PUT)

        if not atm_call or not atm_put:
            logger.error(f"Cannot find ATM options for {symbol}")
            return None

        # Ensure same strike
        if atm_call.strike != atm_put.strike:
            # Adjust to same strike
            atm_put = self.select_option_by_strike(
                chain, atm_call.strike, OptionType.PUT
            )

        total_cost = (atm_call.mid_price + atm_put.mid_price) * contracts * 100

        logger.info(f"   Strike: {atm_call.strike}")
        logger.info(f"   Call: ${atm_call.mid_price:.2f}, Put: ${atm_put.mid_price:.2f}")
        logger.info(f"   Total cost: ${total_cost:,.2f}")
        logger.info(f"   Breakeven: ${atm_call.strike - atm_call.mid_price - atm_put.mid_price:.2f} / ${atm_call.strike + atm_call.mid_price + atm_put.mid_price:.2f}")

        # Execute actual orders via IBKR
        expiry_str = atm_call.expiry.strftime('%Y%m%d')
        
        # 1. Buy Call
        call_result = await self.ibkr.execute_option_order(
            symbol=symbol,
            expiry=expiry_str,
            strike=atm_call.strike,
            right='C',
            side='BUY',
            quantity=contracts,
            order_type='MKT'
        )
        
        # 2. Buy Put
        put_result = await self.ibkr.execute_option_order(
            symbol=symbol,
            expiry=expiry_str,
            strike=atm_put.strike,
            right='P',
            side='BUY',
            quantity=contracts,
            order_type='MKT'
        )
        
        if not call_result or not put_result:
            logger.error(f"Failed to execute straddle for {symbol}")
            return None

        # Record positions
        call_key = f"{symbol}_{expiry_str}_{atm_call.strike}_C"
        put_key = f"{symbol}_{expiry_str}_{atm_put.strike}_P"
        
        self.positions[call_key] = OptionPosition(
            contract=atm_call,
            quantity=contracts,
            entry_price=call_result.get('price', atm_call.mid_price),
            entry_date=datetime.now()
        )
        self.positions[put_key] = OptionPosition(
            contract=atm_put,
            quantity=contracts,
            entry_price=put_result.get('price', atm_put.mid_price),
            entry_date=datetime.now()
        )

        return {
            'strategy': OptionStrategy.STRADDLE.value,
            'symbol': symbol,
            'call': atm_call.to_dict(),
            'put': atm_put.to_dict(),
            'quantity': contracts,
            'cost': (call_result.get('price', 0) + put_result.get('price', 0)) * contracts * 100,
            'call_order_id': call_result.get('order_id'),
            'put_order_id': put_result.get('order_id')
        }

    def calculate_hedge_ratio(
        self,
        stock_shares: int,
        option_delta: float
    ) -> int:
        """
        Calculate number of option contracts to hedge a stock position.

        Args:
            stock_shares: Number of shares to hedge
            option_delta: Delta of the option

        Returns:
            Number of contracts needed
        """
        if option_delta == 0:
            return 0

        # Each contract covers 100 shares
        # Number of contracts = shares / (100 * |delta|)
        contracts = abs(stock_shares / (100 * option_delta))
        return int(round(contracts))

    def get_portfolio_greeks(self) -> Dict[str, float]:
        """Calculate portfolio-level Greeks."""
        total_delta = 0.0
        total_gamma = 0.0
        total_theta = 0.0
        total_vega = 0.0

        for pos in self.positions.values():
            multiplier = pos.quantity * pos.contract.multiplier
            total_delta += pos.contract.delta * multiplier
            total_gamma += pos.contract.gamma * multiplier
            total_theta += pos.contract.theta * multiplier
            total_vega += pos.contract.vega * multiplier

        return {
            'delta': total_delta,
            'gamma': total_gamma,
            'theta': total_theta,
            'vega': total_vega
        }

    def get_status(self) -> Dict:
        """Get options trader status."""
        return {
            'positions_count': len(self.positions),
            'portfolio_greeks': self.get_portfolio_greeks(),
            'scipy_available': SCIPY_AVAILABLE,
            'cached_chains': len(self._chain_cache)
        }
