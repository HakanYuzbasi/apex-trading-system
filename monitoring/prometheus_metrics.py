"""
monitoring/prometheus_metrics.py - Prometheus Metrics Exporter

Exposes trading system metrics for Grafana dashboards:
- Portfolio metrics (value, P&L, positions)
- Trading metrics (trades, win rate, Sharpe)
- System metrics (latency, errors, uptime)
- Risk metrics (exposure, drawdown, VaR)

Runs a lightweight HTTP server for Prometheus scraping.

Usage:
    metrics = PrometheusMetrics(port=8000)
    metrics.start()

    # Update metrics
    metrics.update_portfolio_value(1100000)
    metrics.record_trade("AAPL", "BUY", 100, 150.0)

Grafana Dashboard:
    Import the provided JSON dashboard or create custom panels.
"""

import threading
import time
from datetime import datetime
from typing import Dict, Optional, List
from http.server import HTTPServer, BaseHTTPRequestHandler
import logging
import json

logger = logging.getLogger(__name__)

# Check for prometheus_client
try:
    from prometheus_client import (
        Counter, Gauge, Histogram, Summary,
        generate_latest, CONTENT_TYPE_LATEST,
        CollectorRegistry, REGISTRY
    )
    PROMETHEUS_AVAILABLE = True
except ImportError:
    PROMETHEUS_AVAILABLE = False
    logger.warning("prometheus_client not installed. Install with: pip install prometheus_client")


class MetricsHandler(BaseHTTPRequestHandler):
    """HTTP handler for Prometheus metrics endpoint."""

    def do_GET(self):
        if self.path == '/metrics':
            self.send_response(200)
            self.send_header('Content-Type', CONTENT_TYPE_LATEST)
            self.end_headers()
            self.wfile.write(generate_latest(REGISTRY))
        elif self.path == '/health':
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.end_headers()
            self.wfile.write(json.dumps({'status': 'healthy'}).encode())
        else:
            self.send_response(404)
            self.end_headers()

    def log_message(self, format, *args):
        # Suppress default logging
        pass


class PrometheusMetrics:
    """
    Prometheus metrics exporter for APEX Trading System.

    Metrics categories:
    - apex_portfolio_*: Portfolio-level metrics
    - apex_trading_*: Trading activity metrics
    - apex_risk_*: Risk management metrics
    - apex_system_*: System health metrics
    - apex_model_*: ML model metrics
    """

    def __init__(self, port: int = 8000, registry: CollectorRegistry = None):
        """
        Initialize Prometheus metrics exporter.

        Args:
            port: Port for metrics HTTP server
            registry: Custom registry (uses default if None)
        """
        if not PROMETHEUS_AVAILABLE:
            logger.warning("Prometheus client not available. Metrics disabled.")
            self.enabled = False
            return

        self.enabled = True
        self.port = port
        self.registry = registry or REGISTRY
        self._server: Optional[HTTPServer] = None
        self._server_thread: Optional[threading.Thread] = None

        # Initialize metrics
        self._init_portfolio_metrics()
        self._init_trading_metrics()
        self._init_risk_metrics()
        self._init_system_metrics()
        self._init_model_metrics()
        self._init_governor_metrics()

        logger.info(f"📊 Prometheus Metrics initialized on port {port}")

    def _init_portfolio_metrics(self):
        """Initialize portfolio-related metrics."""
        # Gauges (current values)
        self.portfolio_value = Gauge(
            'apex_portfolio_value_dollars',
            'Current portfolio value in USD',
            registry=self.registry
        )
        self.portfolio_cash = Gauge(
            'apex_portfolio_cash_dollars',
            'Available cash in USD',
            registry=self.registry
        )
        self.portfolio_positions = Gauge(
            'apex_portfolio_positions_count',
            'Number of open positions',
            registry=self.registry
        )
        self.portfolio_daily_pnl = Gauge(
            'apex_portfolio_daily_pnl_dollars',
            'Daily P&L in USD',
            registry=self.registry
        )
        self.portfolio_daily_return = Gauge(
            'apex_portfolio_daily_return_percent',
            'Daily return percentage',
            registry=self.registry
        )
        self.portfolio_total_return = Gauge(
            'apex_portfolio_total_return_percent',
            'Total return since inception',
            registry=self.registry
        )

        # Per-position metrics
        self.position_value = Gauge(
            'apex_position_value_dollars',
            'Position value in USD',
            ['symbol'],
            registry=self.registry
        )
        self.position_pnl = Gauge(
            'apex_position_pnl_dollars',
            'Position P&L in USD',
            ['symbol'],
            registry=self.registry
        )
        self.position_pnl_percent = Gauge(
            'apex_position_pnl_percent',
            'Position P&L percentage',
            ['symbol'],
            registry=self.registry
        )

    def _init_trading_metrics(self):
        """Initialize trading activity metrics."""
        # Counters (cumulative)
        self.trades_total = Counter(
            'apex_trades_total',
            'Total number of trades',
            ['side', 'symbol'],
            registry=self.registry
        )
        self.trades_value_total = Counter(
            'apex_trades_value_dollars_total',
            'Total trade value in USD',
            ['side'],
            registry=self.registry
        )
        self.commissions_total = Counter(
            'apex_commissions_dollars_total',
            'Total commissions paid',
            registry=self.registry
        )

        # Gauges
        self.win_rate = Gauge(
            'apex_trading_win_rate',
            'Win rate (0-1)',
            registry=self.registry
        )
        self.profit_factor = Gauge(
            'apex_trading_profit_factor',
            'Profit factor (gross profit / gross loss)',
            registry=self.registry
        )
        self.avg_trade_pnl = Gauge(
            'apex_trading_avg_trade_pnl_dollars',
            'Average trade P&L',
            registry=self.registry
        )

        # Histograms
        self.trade_pnl_histogram = Histogram(
            'apex_trade_pnl_dollars',
            'Distribution of trade P&L',
            buckets=[-1000, -500, -200, -100, -50, 0, 50, 100, 200, 500, 1000],
            registry=self.registry
        )
        self.trade_duration_histogram = Histogram(
            'apex_trade_duration_hours',
            'Distribution of trade holding time',
            buckets=[0.5, 1, 2, 4, 8, 24, 48, 168],  # Up to 1 week
            registry=self.registry
        )

    def _init_risk_metrics(self):
        """Initialize risk management metrics."""
        self.gross_exposure = Gauge(
            'apex_risk_gross_exposure_dollars',
            'Total gross exposure',
            registry=self.registry
        )
        self.net_exposure = Gauge(
            'apex_risk_net_exposure_dollars',
            'Net exposure (long - short)',
            registry=self.registry
        )
        self.exposure_ratio = Gauge(
            'apex_risk_exposure_ratio',
            'Exposure as ratio of capital',
            registry=self.registry
        )
        self.max_drawdown = Gauge(
            'apex_risk_max_drawdown_percent',
            'Maximum drawdown percentage',
            registry=self.registry
        )
        self.current_drawdown = Gauge(
            'apex_risk_current_drawdown_percent',
            'Current drawdown from peak',
            registry=self.registry
        )
        self.sharpe_ratio = Gauge(
            'apex_risk_sharpe_ratio',
            'Sharpe ratio (annualized)',
            registry=self.registry
        )
        self.sortino_ratio = Gauge(
            'apex_risk_sortino_ratio',
            'Sortino ratio (annualized)',
            registry=self.registry
        )
        self.var_95 = Gauge(
            'apex_risk_var_95_dollars',
            'Value at Risk (95% confidence)',
            registry=self.registry
        )
        self.circuit_breaker_status = Gauge(
            'apex_risk_circuit_breaker_active',
            'Circuit breaker status (1=tripped, 0=normal)',
            registry=self.registry
        )

        # Sector exposure
        self.sector_exposure = Gauge(
            'apex_risk_sector_exposure_dollars',
            'Exposure by sector',
            ['sector'],
            registry=self.registry
        )

    def _init_system_metrics(self):
        """Initialize system health metrics."""
        self.system_uptime = Gauge(
            'apex_system_uptime_seconds',
            'System uptime in seconds',
            registry=self.registry
        )
        self.system_cycle_count = Counter(
            'apex_system_cycles_total',
            'Total trading cycles completed',
            registry=self.registry
        )
        self.system_errors = Counter(
            'apex_system_errors_total',
            'Total errors encountered',
            ['error_type'],
            registry=self.registry
        )
        self.system_ibkr_connected = Gauge(
            'apex_system_ibkr_connected',
            'IBKR connection status (1=connected)',
            registry=self.registry
        )
        self.system_last_heartbeat = Gauge(
            'apex_system_last_heartbeat_timestamp',
            'Unix timestamp of last heartbeat',
            registry=self.registry
        )

        # Latency metrics
        self.cycle_duration = Histogram(
            'apex_system_cycle_duration_seconds',
            'Trading cycle duration',
            buckets=[1, 5, 10, 30, 60, 120, 300],
            registry=self.registry
        )
        self.order_latency = Histogram(
            'apex_system_order_latency_seconds',
            'Order execution latency',
            buckets=[0.1, 0.5, 1, 2, 5, 10],
            registry=self.registry
        )
        self.data_fetch_latency = Summary(
            'apex_system_data_fetch_latency_seconds',
            'Data fetching latency',
            registry=self.registry
        )

    def _init_model_metrics(self):
        """Initialize ML model metrics."""
        self.model_signal_strength = Gauge(
            'apex_model_signal_strength',
            'Current signal strength',
            ['symbol'],
            registry=self.registry
        )
        self.model_confidence = Gauge(
            'apex_model_confidence',
            'Signal confidence',
            ['symbol'],
            registry=self.registry
        )
        self.model_regime = Gauge(
            'apex_model_regime',
            'Current market regime (0=unknown, 1=bull, 2=bear, 3=sideways, 4=high_vol)',
            registry=self.registry
        )
        self.model_feature_cache_hits = Counter(
            'apex_model_feature_cache_hits_total',
            'Feature cache hits',
            registry=self.registry
        )
        self.model_predictions_total = Counter(
            'apex_model_predictions_total',
            'Total model predictions made',
            registry=self.registry
        )

    def _init_governor_metrics(self):
        """Initialize performance governor and kill-switch metrics."""
        self.governor_tier_level = Gauge(
            'apex_governor_tier_level',
            'Governor tier level by asset class and regime (green=0,yellow=1,orange=2,red=3)',
            ['asset_class', 'regime'],
            registry=self.registry
        )
        self.governor_transition_total = Counter(
            'apex_governor_transition_total',
            'Governor tier transitions',
            ['asset_class', 'regime', 'from_tier', 'to_tier'],
            registry=self.registry
        )
        self.governor_blocked_entries_total = Counter(
            'apex_governor_blocked_entries_total',
            'Entries blocked by governor controls',
            ['asset_class', 'regime', 'reason'],
            registry=self.registry
        )
        self.governor_size_multiplier = Gauge(
            'apex_governor_size_multiplier',
            'Effective governor size multiplier',
            ['asset_class', 'regime'],
            registry=self.registry
        )
        self.governor_signal_threshold_boost = Gauge(
            'apex_governor_signal_threshold_boost',
            'Effective governor signal threshold boost',
            ['asset_class', 'regime'],
            registry=self.registry
        )
        self.governor_confidence_boost = Gauge(
            'apex_governor_confidence_boost',
            'Effective governor confidence boost',
            ['asset_class', 'regime'],
            registry=self.registry
        )
        self.governor_halt_entries = Gauge(
            'apex_governor_halt_entries',
            'Governor entry halt flag (1=halt)',
            ['asset_class', 'regime'],
            registry=self.registry
        )
        self.governor_policy_active_info = Gauge(
            'apex_governor_policy_active_info',
            'Active governor policy info label (always 1)',
            ['asset_class', 'regime', 'version'],
            registry=self.registry
        )
        self.kill_switch_active = Gauge(
            'apex_risk_kill_switch_active',
            'Kill-switch state (1=active)',
            registry=self.registry
        )
        self.kill_switch_triggers_total = Counter(
            'apex_risk_kill_switch_triggers_total',
            'Kill-switch trigger events',
            ['reason'],
            registry=self.registry
        )

    def start(self):
        """Start the metrics HTTP server."""
        if not self.enabled:
            return

        try:
            self._server = HTTPServer(('0.0.0.0', self.port), MetricsHandler)
            self._server_thread = threading.Thread(target=self._server.serve_forever)
            self._server_thread.daemon = True
            self._server_thread.start()

            logger.info(f"✅ Prometheus metrics server started on port {self.port}")
            logger.info(f"   Metrics endpoint: http://localhost:{self.port}/metrics")
            logger.info(f"   Health endpoint: http://localhost:{self.port}/health")

        except Exception as e:
            logger.error(f"Failed to start metrics server: {e}")

    def stop(self):
        """Stop the metrics HTTP server."""
        if self._server:
            self._server.shutdown()
            logger.info("Prometheus metrics server stopped")

    # ===================
    # Portfolio Updates
    # ===================

    def update_portfolio_value(self, value: float):
        """Update portfolio value."""
        if not self.enabled:
            return
        self.portfolio_value.set(value)

    def update_portfolio_cash(self, cash: float):
        """Update available cash."""
        if not self.enabled:
            return
        self.portfolio_cash.set(cash)

    def update_portfolio_positions(self, count: int):
        """Update position count."""
        if not self.enabled:
            return
        self.portfolio_positions.set(count)

    def update_daily_pnl(self, pnl: float, return_pct: float):
        """Update daily P&L."""
        if not self.enabled:
            return
        self.portfolio_daily_pnl.set(pnl)
        self.portfolio_daily_return.set(return_pct)

    def update_position(self, symbol: str, value: float, pnl: float, pnl_pct: float):
        """Update individual position metrics."""
        if not self.enabled:
            return
        self.position_value.labels(symbol=symbol).set(value)
        self.position_pnl.labels(symbol=symbol).set(pnl)
        self.position_pnl_percent.labels(symbol=symbol).set(pnl_pct)

    # ===================
    # Trading Updates
    # ===================

    def record_trade(
        self,
        symbol: str,
        side: str,
        quantity: int,
        price: float,
        pnl: float = 0,
        commission: float = 0
    ):
        """Record a trade."""
        if not self.enabled:
            return
        self.trades_total.labels(side=side, symbol=symbol).inc()
        self.trades_value_total.labels(side=side).inc(quantity * price)
        if commission > 0:
            self.commissions_total.inc(commission)
        if pnl != 0:
            self.trade_pnl_histogram.observe(pnl)

    def update_trading_stats(
        self,
        win_rate: float,
        profit_factor: float,
        avg_pnl: float
    ):
        """Update trading statistics."""
        if not self.enabled:
            return
        self.win_rate.set(win_rate)
        self.profit_factor.set(profit_factor)
        self.avg_trade_pnl.set(avg_pnl)

    # ===================
    # Risk Updates
    # ===================

    def update_exposure(
        self,
        gross: float,
        net: float,
        ratio: float
    ):
        """Update exposure metrics."""
        if not self.enabled:
            return
        self.gross_exposure.set(gross)
        self.net_exposure.set(net)
        self.exposure_ratio.set(ratio)

    def update_drawdown(self, current: float, maximum: float):
        """Update drawdown metrics."""
        if not self.enabled:
            return
        self.current_drawdown.set(current)
        self.max_drawdown.set(maximum)

    def update_risk_ratios(
        self,
        sharpe: float,
        sortino: float,
        var_95: float
    ):
        """Update risk-adjusted metrics."""
        if not self.enabled:
            return
        self.sharpe_ratio.set(sharpe)
        self.sortino_ratio.set(sortino)
        self.var_95.set(var_95)

    def update_circuit_breaker(self, is_tripped: bool):
        """Update circuit breaker status."""
        if not self.enabled:
            return
        self.circuit_breaker_status.set(1 if is_tripped else 0)

    def update_sector_exposure(self, exposures: Dict[str, float]):
        """Update sector exposures."""
        if not self.enabled:
            return
        for sector, exposure in exposures.items():
            self.sector_exposure.labels(sector=sector).set(exposure)

    # ===================
    # System Updates
    # ===================

    def update_uptime(self, seconds: float):
        """Update system uptime."""
        if not self.enabled:
            return
        self.system_uptime.set(seconds)

    def increment_cycle(self):
        """Increment cycle counter."""
        if not self.enabled:
            return
        self.system_cycle_count.inc()

    def record_error(self, error_type: str):
        """Record an error."""
        if not self.enabled:
            return
        self.system_errors.labels(error_type=error_type).inc()

    def update_ibkr_status(self, connected: bool):
        """Update IBKR connection status."""
        if not self.enabled:
            return
        self.system_ibkr_connected.set(1 if connected else 0)

    def record_heartbeat(self):
        """Record heartbeat timestamp."""
        if not self.enabled:
            return
        self.system_last_heartbeat.set(time.time())

    def record_cycle_duration(self, seconds: float):
        """Record cycle duration."""
        if not self.enabled:
            return
        self.cycle_duration.observe(seconds)

    def record_order_latency(self, seconds: float):
        """Record order execution latency."""
        if not self.enabled:
            return
        self.order_latency.observe(seconds)

    # ===================
    # Model Updates
    # ===================

    def update_signal(self, symbol: str, strength: float, confidence: float):
        """Update model signal for a symbol."""
        if not self.enabled:
            return
        self.model_signal_strength.labels(symbol=symbol).set(strength)
        self.model_confidence.labels(symbol=symbol).set(confidence)

    def update_regime(self, regime: int):
        """Update market regime (0=unknown, 1=bull, 2=bear, 3=sideways, 4=high_vol)."""
        if not self.enabled:
            return
        self.model_regime.set(regime)

    def record_cache_hit(self):
        """Record feature cache hit."""
        if not self.enabled:
            return
        self.model_feature_cache_hits.inc()

    def record_prediction(self):
        """Record model prediction."""
        if not self.enabled:
            return
        self.model_predictions_total.inc()

    # ===================
    # Governor/Kill-Switch
    # ===================

    def update_governor_state(
        self,
        asset_class: str,
        regime: str,
        tier_level: int,
        size_multiplier: float,
        signal_threshold_boost: float,
        confidence_boost: float,
        halt_entries: bool,
        policy_version: str,
    ):
        if not self.enabled:
            return
        labels = {"asset_class": asset_class, "regime": regime}
        self.governor_tier_level.labels(**labels).set(tier_level)
        self.governor_size_multiplier.labels(**labels).set(size_multiplier)
        self.governor_signal_threshold_boost.labels(**labels).set(signal_threshold_boost)
        self.governor_confidence_boost.labels(**labels).set(confidence_boost)
        self.governor_halt_entries.labels(**labels).set(1 if halt_entries else 0)
        self.governor_policy_active_info.labels(
            asset_class=asset_class,
            regime=regime,
            version=policy_version,
        ).set(1)

    def record_governor_transition(
        self,
        asset_class: str,
        regime: str,
        from_tier: str,
        to_tier: str,
    ):
        if not self.enabled:
            return
        self.governor_transition_total.labels(
            asset_class=asset_class,
            regime=regime,
            from_tier=from_tier,
            to_tier=to_tier,
        ).inc()

    def record_governor_blocked_entry(self, asset_class: str, regime: str, reason: str):
        if not self.enabled:
            return
        self.governor_blocked_entries_total.labels(
            asset_class=asset_class,
            regime=regime,
            reason=reason,
        ).inc()

    def update_kill_switch(self, active: bool, reason: str = ""):
        if not self.enabled:
            return
        self.kill_switch_active.set(1 if active else 0)
        if active and reason:
            self.kill_switch_triggers_total.labels(reason=reason).inc()


# Grafana Dashboard JSON (save to file)
GRAFANA_DASHBOARD = {
    "title": "APEX Trading System",
    "panels": [
        {
            "title": "Portfolio Value",
            "type": "stat",
            "targets": [{"expr": "apex_portfolio_value_dollars"}]
        },
        {
            "title": "Daily P&L",
            "type": "stat",
            "targets": [{"expr": "apex_portfolio_daily_pnl_dollars"}]
        },
        {
            "title": "Position Count",
            "type": "gauge",
            "targets": [{"expr": "apex_portfolio_positions_count"}]
        },
        {
            "title": "Sharpe Ratio",
            "type": "stat",
            "targets": [{"expr": "apex_risk_sharpe_ratio"}]
        },
        {
            "title": "Drawdown",
            "type": "gauge",
            "targets": [{"expr": "apex_risk_current_drawdown_percent"}]
        },
        {
            "title": "Trade P&L Distribution",
            "type": "histogram",
            "targets": [{"expr": "apex_trade_pnl_dollars_bucket"}]
        }
    ]
}
