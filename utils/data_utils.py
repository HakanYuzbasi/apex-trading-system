"""
utils/data_utils.py - Data Transformation Utilities

Common data manipulation functions for financial time series data.
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Optional, Union, Tuple


def calculate_returns(
    prices: pd.Series,
    periods: List[int] = None,
    log_returns: bool = False
) -> Union[pd.Series, Dict[int, pd.Series]]:
    """
    Calculate returns at different periods.

    Args:
        prices: Price series
        periods: List of periods (default: [1])
        log_returns: Calculate log returns instead of simple returns

    Returns:
        Single Series if one period, Dict of Series if multiple

    Example:
        returns = calculate_returns(prices)
        multi_returns = calculate_returns(prices, periods=[1, 5, 20])
    """
    if periods is None:
        periods = [1]

    results = {}
    for period in periods:
        if log_returns:
            returns = np.log(prices / prices.shift(period))
        else:
            returns = prices.pct_change(period)
        results[period] = returns

    if len(periods) == 1:
        return results[periods[0]]
    return results


def calculate_volatility(
    returns: pd.Series,
    window: int = 20,
    annualize: bool = True,
    trading_days: int = 252
) -> pd.Series:
    """
    Calculate rolling volatility.

    Args:
        returns: Return series
        window: Rolling window size
        annualize: Annualize the volatility
        trading_days: Trading days per year

    Returns:
        Volatility series

    Example:
        vol = calculate_volatility(returns, window=20)
    """
    vol = returns.rolling(window=window).std()

    if annualize:
        vol = vol * np.sqrt(trading_days)

    return vol


def zscore_normalize(
    series: pd.Series,
    window: int = 20,
    min_periods: int = None
) -> pd.Series:
    """
    Normalize series using rolling z-score.

    Args:
        series: Input series
        window: Rolling window size
        min_periods: Minimum periods for calculation

    Returns:
        Z-score normalized series

    Example:
        normalized = zscore_normalize(prices, window=20)
    """
    if min_periods is None:
        min_periods = window // 2

    rolling_mean = series.rolling(window=window, min_periods=min_periods).mean()
    rolling_std = series.rolling(window=window, min_periods=min_periods).std()

    # Avoid division by zero
    rolling_std = rolling_std.replace(0, np.nan)

    return (series - rolling_mean) / rolling_std


def fill_missing_data(
    df: pd.DataFrame,
    method: str = "forward",
    limit: int = None,
    fill_value: float = None
) -> pd.DataFrame:
    """
    Intelligently fill missing values.

    Args:
        df: DataFrame with missing values
        method: Fill method ('forward', 'backward', 'interpolate', 'value')
        limit: Maximum consecutive fills
        fill_value: Value to fill with if method='value'

    Returns:
        DataFrame with filled values

    Example:
        filled = fill_missing_data(df, method='forward', limit=5)
    """
    df = df.copy()

    if method == "forward":
        df = df.ffill(limit=limit)
    elif method == "backward":
        df = df.bfill(limit=limit)
    elif method == "interpolate":
        df = df.interpolate(method='linear', limit=limit)
    elif method == "value":
        df = df.fillna(fill_value if fill_value is not None else 0)

    return df


def resample_ohlcv(
    df: pd.DataFrame,
    timeframe: str,
    ohlc_columns: Dict[str, str] = None
) -> pd.DataFrame:
    """
    Resample OHLCV data to different timeframe.

    Args:
        df: DataFrame with OHLCV data
        timeframe: Target timeframe ('5T', '15T', '1H', '1D', etc.)
        ohlc_columns: Column name mapping

    Returns:
        Resampled DataFrame

    Example:
        daily = resample_ohlcv(minute_data, '1D')
    """
    if ohlc_columns is None:
        ohlc_columns = {
            'open': 'Open',
            'high': 'High',
            'low': 'Low',
            'close': 'Close',
            'volume': 'Volume'
        }

    agg_dict = {
        ohlc_columns['open']: 'first',
        ohlc_columns['high']: 'max',
        ohlc_columns['low']: 'min',
        ohlc_columns['close']: 'last',
    }

    if ohlc_columns['volume'] in df.columns:
        agg_dict[ohlc_columns['volume']] = 'sum'

    return df.resample(timeframe).agg(agg_dict).dropna()


def detect_outliers(
    series: pd.Series,
    method: str = "zscore",
    threshold: float = 3.0,
    window: int = None
) -> pd.Series:
    """
    Detect outliers in a series.

    Args:
        series: Input series
        method: Detection method ('zscore', 'iqr', 'mad')
        threshold: Threshold for outlier detection
        window: Rolling window (None for full series)

    Returns:
        Boolean series (True = outlier)

    Example:
        outliers = detect_outliers(returns, method='zscore', threshold=3)
    """
    if method == "zscore":
        if window:
            mean = series.rolling(window).mean()
            std = series.rolling(window).std()
        else:
            mean = series.mean()
            std = series.std()

        zscore = (series - mean) / std
        return abs(zscore) > threshold

    elif method == "iqr":
        if window:
            q1 = series.rolling(window).quantile(0.25)
            q3 = series.rolling(window).quantile(0.75)
        else:
            q1 = series.quantile(0.25)
            q3 = series.quantile(0.75)

        iqr = q3 - q1
        lower = q1 - threshold * iqr
        upper = q3 + threshold * iqr
        return (series < lower) | (series > upper)

    elif method == "mad":
        # Median Absolute Deviation
        if window:
            median = series.rolling(window).median()
            mad = (series - median).abs().rolling(window).median()
        else:
            median = series.median()
            mad = (series - median).abs().median()

        modified_zscore = 0.6745 * (series - median) / mad
        return abs(modified_zscore) > threshold

    else:
        raise ValueError(f"Unknown method: {method}")


def winsorize(
    series: pd.Series,
    lower_percentile: float = 0.01,
    upper_percentile: float = 0.99
) -> pd.Series:
    """
    Winsorize series to reduce impact of outliers.

    Args:
        series: Input series
        lower_percentile: Lower percentile limit
        upper_percentile: Upper percentile limit

    Returns:
        Winsorized series

    Example:
        clean = winsorize(returns, lower_percentile=0.01, upper_percentile=0.99)
    """
    lower = series.quantile(lower_percentile)
    upper = series.quantile(upper_percentile)

    return series.clip(lower=lower, upper=upper)


def calculate_drawdown(prices: pd.Series) -> Tuple[pd.Series, float, int]:
    """
    Calculate drawdown series and statistics.

    Args:
        prices: Price series

    Returns:
        Tuple of (drawdown series, max drawdown, max drawdown duration)

    Example:
        dd, max_dd, duration = calculate_drawdown(prices)
    """
    # Calculate running max
    running_max = prices.cummax()

    # Calculate drawdown
    drawdown = (prices - running_max) / running_max

    # Max drawdown
    max_drawdown = drawdown.min()

    # Drawdown duration (simplified)
    is_underwater = drawdown < 0
    duration = 0
    current_duration = 0

    for underwater in is_underwater:
        if underwater:
            current_duration += 1
            duration = max(duration, current_duration)
        else:
            current_duration = 0

    return drawdown, max_drawdown, duration


def calculate_sharpe_ratio(
    returns: pd.Series,
    risk_free_rate: float = 0.0,
    periods_per_year: int = 252
) -> float:
    """
    Calculate Sharpe ratio.

    Args:
        returns: Return series
        risk_free_rate: Annual risk-free rate
        periods_per_year: Number of periods per year

    Returns:
        Sharpe ratio

    Example:
        sharpe = calculate_sharpe_ratio(daily_returns)
    """
    excess_returns = returns - risk_free_rate / periods_per_year
    mean_return = excess_returns.mean()
    std_return = excess_returns.std()

    if std_return == 0:
        return 0.0

    return mean_return / std_return * np.sqrt(periods_per_year)


def calculate_sortino_ratio(
    returns: pd.Series,
    risk_free_rate: float = 0.0,
    periods_per_year: int = 252
) -> float:
    """
    Calculate Sortino ratio (using downside deviation).

    Args:
        returns: Return series
        risk_free_rate: Annual risk-free rate
        periods_per_year: Number of periods per year

    Returns:
        Sortino ratio

    Example:
        sortino = calculate_sortino_ratio(daily_returns)
    """
    excess_returns = returns - risk_free_rate / periods_per_year
    mean_return = excess_returns.mean()

    # Downside deviation (only negative returns)
    downside_returns = excess_returns[excess_returns < 0]
    downside_std = np.sqrt((downside_returns ** 2).mean())

    if downside_std == 0:
        return 0.0

    return mean_return / downside_std * np.sqrt(periods_per_year)


def calculate_calmar_ratio(
    returns: pd.Series,
    periods_per_year: int = 252
) -> float:
    """
    Calculate Calmar ratio (annual return / max drawdown).

    Args:
        returns: Return series
        periods_per_year: Number of periods per year

    Returns:
        Calmar ratio

    Example:
        calmar = calculate_calmar_ratio(daily_returns)
    """
    # Calculate cumulative returns
    cum_returns = (1 + returns).cumprod()

    # Annual return
    total_return = cum_returns.iloc[-1] - 1
    years = len(returns) / periods_per_year
    annual_return = (1 + total_return) ** (1 / years) - 1 if years > 0 else 0

    # Max drawdown
    _, max_dd, _ = calculate_drawdown(cum_returns)

    if max_dd == 0:
        return 0.0

    return annual_return / abs(max_dd)


def ewma_volatility(
    returns: pd.Series,
    span: int = 20,
    annualize: bool = True,
    trading_days: int = 252
) -> pd.Series:
    """
    Calculate exponentially weighted moving average volatility.

    Args:
        returns: Return series
        span: EWMA span
        annualize: Annualize the volatility
        trading_days: Trading days per year

    Returns:
        EWMA volatility series

    Example:
        vol = ewma_volatility(returns, span=20)
    """
    variance = returns.ewm(span=span).var()
    vol = np.sqrt(variance)

    if annualize:
        vol = vol * np.sqrt(trading_days)

    return vol


def align_series(*series: pd.Series) -> List[pd.Series]:
    """
    Align multiple series to common dates.

    Args:
        *series: Variable number of series to align

    Returns:
        List of aligned series

    Example:
        aligned = align_series(prices1, prices2, prices3)
    """
    # Get common index
    common_index = series[0].index
    for s in series[1:]:
        common_index = common_index.intersection(s.index)

    return [s.loc[common_index] for s in series]
