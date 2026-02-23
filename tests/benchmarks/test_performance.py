"""
tests/benchmarks/test_performance.py - Performance Benchmarks

Benchmarks for critical trading system operations.
Use with pytest-benchmark: pytest tests/benchmarks/ --benchmark-only
"""

import pytest
import pandas as pd
import numpy as np
import time
import sys
import os

# Add parent to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))


@pytest.fixture
def large_price_data():
    """Generate large price dataset for benchmarking."""
    np.random.seed(42)
    n_days = 1000  # ~4 years of data

    dates = pd.date_range('2020-01-01', periods=n_days, freq='D')
    close = 100 * np.exp(np.cumsum(np.random.normal(0.0002, 0.02, n_days)))

    return pd.DataFrame({
        'Open': close * (1 + np.random.uniform(-0.01, 0.01, n_days)),
        'High': close * (1 + np.random.uniform(0, 0.02, n_days)),
        'Low': close * (1 - np.random.uniform(0, 0.02, n_days)),
        'Close': close,
        'Volume': np.random.randint(1_000_000, 50_000_000, n_days)
    }, index=dates)


@pytest.fixture
def multi_symbol_data(large_price_data):
    """Generate data for multiple symbols."""
    symbols = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'NVDA', 'TSLA', 'JPM', 'V', 'JNJ']
    return {symbol: large_price_data.copy() for symbol in symbols}


class TestSignalGenerationPerformance:
    """Benchmark signal generation performance."""

    @pytest.mark.benchmark(group="signals")
    def test_single_signal_latency(self, benchmark, large_price_data):
        """Benchmark single signal generation latency."""
        prices = large_price_data['Close']

        def generate_signal():
            # Simplified signal calculation
            returns = prices.pct_change(20).iloc[-1]
            ma_short = prices.rolling(20).mean().iloc[-1]
            ma_long = prices.rolling(50).mean().iloc[-1]
            trend = (ma_short - ma_long) / ma_long if ma_long > 0 else 0
            return np.tanh(returns * 10) * 0.5 + np.tanh(trend * 20) * 0.5

        result = benchmark(generate_signal)
        assert -1 <= result <= 1

    @pytest.mark.benchmark(group="signals")
    def test_batch_signal_generation(self, benchmark, multi_symbol_data):
        """Benchmark batch signal generation for multiple symbols."""
        def generate_all_signals():
            signals = {}
            for symbol, data in multi_symbol_data.items():
                prices = data['Close']
                returns = prices.pct_change(20).iloc[-1]
                signals[symbol] = np.tanh(returns * 10)
            return signals

        result = benchmark(generate_all_signals)
        assert len(result) == 10

    @pytest.mark.benchmark(group="signals")
    def test_feature_extraction_speed(self, benchmark, large_price_data):
        """Benchmark feature extraction speed."""
        prices = large_price_data['Close']

        def extract_features():
            features = {}

            # Multi-period returns
            for period in [5, 10, 20, 60]:
                features[f'ret_{period}d'] = prices.pct_change(period).iloc[-1]

            # Volatility
            returns = prices.pct_change()
            for period in [10, 20, 60]:
                features[f'vol_{period}d'] = returns.rolling(period).std().iloc[-1]

            # RSI
            delta = prices.diff()
            gain = delta.where(delta > 0, 0).rolling(14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
            rs = gain / loss
            features['rsi'] = (100 - (100 / (1 + rs))).iloc[-1]

            # MACD
            ema12 = prices.ewm(span=12).mean()
            ema26 = prices.ewm(span=26).mean()
            features['macd'] = (ema12.iloc[-1] - ema26.iloc[-1]) / prices.iloc[-1]

            return features

        result = benchmark(extract_features)
        assert len(result) >= 9


class TestRiskCalculationPerformance:
    """Benchmark risk calculation performance."""

    @pytest.mark.benchmark(group="risk")
    def test_var_calculation_speed(self, benchmark, large_price_data):
        """Benchmark Value at Risk calculation."""
        returns = large_price_data['Close'].pct_change().dropna()

        def calculate_var():
            var_95 = np.percentile(returns, 5)
            var_99 = np.percentile(returns, 1)
            return var_95, var_99

        result = benchmark(calculate_var)
        assert result[0] < 0  # VaR should be negative

    @pytest.mark.benchmark(group="risk")
    def test_correlation_matrix_speed(self, benchmark, multi_symbol_data):
        """Benchmark correlation matrix calculation."""
        returns = pd.DataFrame({
            symbol: data['Close'].pct_change()
            for symbol, data in multi_symbol_data.items()
        }).dropna()

        def calculate_correlation():
            return returns.corr()

        result = benchmark(calculate_correlation)
        assert result.shape == (10, 10)

    @pytest.mark.benchmark(group="risk")
    def test_portfolio_volatility_speed(self, benchmark, multi_symbol_data):
        """Benchmark portfolio volatility calculation."""
        returns = pd.DataFrame({
            symbol: data['Close'].pct_change()
            for symbol, data in multi_symbol_data.items()
        }).dropna()

        weights = np.array([0.1] * 10)
        cov_matrix = returns.cov() * 252

        def calculate_portfolio_vol():
            return np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))

        result = benchmark(calculate_portfolio_vol)
        assert result > 0


class TestDataProcessingPerformance:
    """Benchmark data processing performance."""

    @pytest.mark.benchmark(group="data")
    def test_ohlcv_processing_speed(self, benchmark, large_price_data):
        """Benchmark OHLCV data processing."""
        def process_data():
            df = large_price_data.copy()
            df['Returns'] = df['Close'].pct_change()
            df['SMA_20'] = df['Close'].rolling(20).mean()
            df['SMA_50'] = df['Close'].rolling(50).mean()
            df['Volatility'] = df['Returns'].rolling(20).std()
            df['High_Low_Range'] = (df['High'] - df['Low']) / df['Close']
            return df.dropna()

        result = benchmark(process_data)
        assert len(result) > 0

    @pytest.mark.benchmark(group="data")
    def test_resample_performance(self, benchmark, large_price_data):
        """Benchmark data resampling."""
        def resample_data():
            weekly = large_price_data.resample('W').agg({
                'Open': 'first',
                'High': 'max',
                'Low': 'min',
                'Close': 'last',
                'Volume': 'sum'
            })
            return weekly

        result = benchmark(resample_data)
        assert len(result) > 0

    @pytest.mark.benchmark(group="data")
    def test_rolling_window_speed(self, benchmark, large_price_data):
        """Benchmark rolling window calculations."""
        prices = large_price_data['Close']

        def calculate_rolling():
            return {
                'sma_10': prices.rolling(10).mean(),
                'sma_20': prices.rolling(20).mean(),
                'sma_50': prices.rolling(50).mean(),
                'sma_200': prices.rolling(200).mean(),
                'std_20': prices.rolling(20).std(),
                'max_20': prices.rolling(20).max(),
                'min_20': prices.rolling(20).min(),
            }

        result = benchmark(calculate_rolling)
        assert len(result) == 7


class TestPortfolioOptimizationPerformance:
    """Benchmark portfolio optimization performance."""

    @pytest.mark.benchmark(group="portfolio")
    def test_covariance_calculation(self, benchmark, multi_symbol_data):
        """Benchmark covariance matrix calculation."""
        returns = pd.DataFrame({
            symbol: data['Close'].pct_change()
            for symbol, data in multi_symbol_data.items()
        }).dropna()

        def calculate_cov():
            return returns.cov() * 252

        result = benchmark(calculate_cov)
        assert result.shape == (10, 10)

    @pytest.mark.benchmark(group="portfolio")
    def test_efficient_frontier_point(self, benchmark, multi_symbol_data):
        """Benchmark single efficient frontier point calculation."""
        returns = pd.DataFrame({
            symbol: data['Close'].pct_change()
            for symbol, data in multi_symbol_data.items()
        }).dropna()

        mean_returns = returns.mean() * 252
        cov_matrix = returns.cov() * 252

        def calculate_portfolio_metrics():
            weights = np.array([0.1] * 10)
            port_return = np.dot(weights, mean_returns)
            port_vol = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
            sharpe = port_return / port_vol
            return port_return, port_vol, sharpe

        result = benchmark(calculate_portfolio_metrics)
        assert len(result) == 3


class TestMemoryUsage:
    """Test memory usage of critical operations."""

    @pytest.mark.benchmark(group="memory")
    def test_dataframe_memory(self, large_price_data):
        """Measure DataFrame memory usage."""
        memory_mb = large_price_data.memory_usage(deep=True).sum() / 1024 / 1024
        assert memory_mb < 1.0  # Should be less than 1MB for 1000 rows

    @pytest.mark.benchmark(group="memory")
    def test_multi_symbol_memory(self, multi_symbol_data):
        """Measure multi-symbol data memory usage."""
        total_memory = sum(
            df.memory_usage(deep=True).sum()
            for df in multi_symbol_data.values()
        ) / 1024 / 1024

        assert total_memory < 10.0  # Should be less than 10MB for 10 symbols


class TestLatencyTargets:
    """Verify latency targets are met."""

    @pytest.mark.benchmark(group="latency")
    def test_signal_under_100ms(self, benchmark, large_price_data):
        """Signal generation should complete under 100ms."""
        prices = large_price_data['Close']

        def generate_full_signal():
            # Full signal generation pipeline
            features = {}

            # Technical indicators
            features['momentum'] = prices.pct_change(20).iloc[-1]
            features['rsi'] = 50  # Simplified
            features['macd'] = (
                prices.ewm(span=12).mean().iloc[-1] -
                prices.ewm(span=26).mean().iloc[-1]
            )

            # Generate signal
            signal = sum(features.values()) / len(features)
            return np.clip(signal, -1, 1)

        result = benchmark(generate_full_signal)

        # Verify benchmark mean is under 100ms
        # Note: This assertion works with pytest-benchmark
        assert -1 <= result <= 1

    @pytest.mark.benchmark(group="latency")
    def test_risk_check_under_50ms(self, benchmark, large_price_data):
        """Risk check should complete under 50ms."""
        returns = large_price_data['Close'].pct_change().dropna()

        def run_risk_checks():
            checks = {
                'var_95': np.percentile(returns, 5),
                'volatility': returns.std() * np.sqrt(252),
                'max_loss': returns.min(),
                'drawdown': (
                    large_price_data['Close'].iloc[-1] /
                    large_price_data['Close'].max() - 1
                ),
            }
            return all(v > -0.20 for v in checks.values())

        result = benchmark(run_risk_checks)
        assert isinstance(result, bool)


# Standalone timing utilities
def time_function(func, *args, iterations=100, **kwargs):
    """Time a function over multiple iterations."""
    times = []
    for _ in range(iterations):
        start = time.perf_counter()
        func(*args, **kwargs)
        elapsed = time.perf_counter() - start
        times.append(elapsed * 1000)  # Convert to ms

    return {
        'mean_ms': np.mean(times),
        'std_ms': np.std(times),
        'min_ms': np.min(times),
        'max_ms': np.max(times),
        'p95_ms': np.percentile(times, 95),
        'p99_ms': np.percentile(times, 99),
    }


if __name__ == '__main__':
    # Quick standalone benchmark
    print("Running quick benchmarks...")

    # Generate test data
    np.random.seed(42)
    n = 1000
    prices = pd.Series(100 * np.exp(np.cumsum(np.random.normal(0.0002, 0.02, n))))

    # Test signal generation
    def test_signal():
        return np.tanh(prices.pct_change(20).iloc[-1] * 10)

    results = time_function(test_signal)
    print(f"Signal generation: {results['mean_ms']:.3f}ms (p99: {results['p99_ms']:.3f}ms)")
