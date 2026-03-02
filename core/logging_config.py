"""
core/logging_config.py - Production Logging Configuration

Two-channel logging:
  - MAIN log  (/private/tmp/apex_main.log): WARNING+ plus whitelisted
    operational INFO (trades, fills, signals, circuit breaker, kill-switch).
    Safe to tail -f for live monitoring.
  - DEBUG log (/private/tmp/apex_debug.log): everything (DEBUG+), for
    post-mortem investigation.

Console mirrors the main log (WARNING+) so stdout stays quiet in prod.
"""

import logging
import logging.handlers
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any

try:
    from core.request_context import get_context as get_request_context
    REQUEST_CONTEXT_AVAILABLE = True
except Exception:
    REQUEST_CONTEXT_AVAILABLE = False

# ---------------------------------------------------------------------------
# Whitelist: INFO-level messages that ARE important enough for the main log.
# Matched as a prefix/substring of record.getMessage().
# ---------------------------------------------------------------------------
_MAIN_LOG_INFO_KEYWORDS = (
    # Trade lifecycle
    "order placed", "order filled", "order cancelled", "order rejected",
    "trade recorded",
    # Risk events
    "circuit breaker", "kill switch", "kill-switch",
    "daily loss", "drawdown breached", "max drawdown",
    "position limit", "risk alert",
    # System lifecycle
    "apex trader started", "apex trader stopped", "shutting down",
    "connected to ibkr", "connected to alpaca", "disconnected from",
    "reconnect", "preflight",
    # Performance snapshots
    "performance summary", "equity recorded",
    # Signals (high-confidence only — checked separately via signal strength)
    "bullish signal", "bearish signal",
    # Mandate / governor
    "mandate", "governor policy",
    # Walk-forward / model
    "model retrained", "regime change",
)


def _safe_request_context() -> Dict[str, Any]:
    if not REQUEST_CONTEXT_AVAILABLE:
        return {}
    try:
        ctx = get_request_context().to_dict()
    except Exception:
        return {}
    return {k: v for k, v in ctx.items() if v is not None}


# ---------------------------------------------------------------------------
# Filters
# ---------------------------------------------------------------------------

class MainLogFilter(logging.Filter):
    """
    Allow a record through to the main log if:
      - level >= WARNING, OR
      - level == INFO and the message contains a whitelisted keyword.
    DEBUG records are always blocked.
    """
    def filter(self, record: logging.LogRecord) -> bool:
        if record.levelno >= logging.WARNING:
            return True
        if record.levelno == logging.INFO:
            msg = record.getMessage().lower()
            return any(kw in msg for kw in _MAIN_LOG_INFO_KEYWORDS)
        return False  # DEBUG


class NoDebugFilter(logging.Filter):
    """Block DEBUG records (used on console handler)."""
    def filter(self, record: logging.LogRecord) -> bool:
        return record.levelno > logging.DEBUG


# IBKR connectivity blips (Error 1100/1102) are logged at ERROR by ib_insync
# but are non-critical reconnect events — demote them to INFO.
_IBKR_DEMOTE_PATTERNS = (
    "error 1100",  # Connectivity lost — auto-recovers
    "error 1102",  # Connectivity restored
    "error 2104",  # Market data farm connection OK
    "error 2106",  # Historical data farm OK
    "error 2107",  # Historical data farm inactive but ready
    "error 2108",  # Market data farm connection inactive
    "error 2158",  # Sec-def data farm connection OK
)

class IBKRConnectivityFilter(logging.Filter):
    """Demote known IBKR benign connectivity errors from ERROR to INFO."""
    def filter(self, record: logging.LogRecord) -> bool:
        if record.levelno == logging.ERROR and record.name.startswith("ib_insync"):
            msg_lower = record.getMessage().lower()
            if any(pat in msg_lower for pat in _IBKR_DEMOTE_PATTERNS):
                record.levelno = logging.INFO
                record.levelname = "INFO"
        return True


# ---------------------------------------------------------------------------
# Formatters
# ---------------------------------------------------------------------------

class MainFormatter(logging.Formatter):
    """
    Clean, human-readable single-line format for the main (tailed) log.
    Adds ANSI colour when writing to a real TTY.
    """
    _COLORS = {
        logging.DEBUG:    "\033[36m",   # cyan
        logging.INFO:     "\033[32m",   # green
        logging.WARNING:  "\033[33m",   # yellow
        logging.ERROR:    "\033[31m",   # red
        logging.CRITICAL: "\033[35m",   # magenta
    }
    _RESET = "\033[0m"

    def __init__(self, use_color: bool = False):
        super().__init__()
        self._use_color = use_color

    def format(self, record: logging.LogRecord) -> str:
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        level = record.levelname[:4]  # WARN, INFO, ERRO, CRIT, DEBU
        if self._use_color:
            c = self._COLORS.get(record.levelno, "")
            level_str = f"{c}{level}{self._RESET}"
        else:
            level_str = level

        msg = record.getMessage()
        line = f"{ts} {level_str} [{record.name}] {msg}"
        if record.exc_info:
            line += "\n" + self.formatException(record.exc_info)
        return line


class DebugFormatter(logging.Formatter):
    """
    Verbose format for the debug log — includes module, line, and request context.
    """
    def format(self, record: logging.LogRecord) -> str:
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
        req_ctx = _safe_request_context()
        ctx_str = ""
        if req_ctx:
            parts = [f"{k}={v}" for k, v in req_ctx.items() if v]
            ctx_str = " [" + " ".join(parts) + "]"
        msg = record.getMessage()
        line = (
            f"{ts} {record.levelname:8} "
            f"{record.name}:{record.lineno}{ctx_str} — {msg}"
        )
        if record.exc_info:
            line += "\n" + self.formatException(record.exc_info)
        return line


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def setup_logging(
    level: str = "DEBUG",
    log_file: Optional[str] = None,
    json_format: bool = False,   # kept for signature compat, unused
    console_output: bool = True,
    context=None,                # kept for signature compat, unused
    max_bytes: int = 10 * 1024 * 1024,
    backup_count: int = 5,
    main_log_file: Optional[str] = "/private/tmp/apex_main.log",
    debug_log_file: Optional[str] = "/private/tmp/apex_debug.log",
) -> logging.Logger:
    """
    Configure the two-channel production logging pipeline.

    Parameters
    ----------
    level          : Root logger level (should stay DEBUG so debug handler captures all).
    log_file       : Legacy alias for main_log_file (kept for caller compat).
    main_log_file  : Path for the clean main log (tail -f this one).
    debug_log_file : Path for the verbose debug log.
    """
    # json_format / context kept only for call-site backward compatibility
    del json_format, context

    # Legacy alias
    if log_file and not main_log_file:
        main_log_file = log_file

    root = logging.getLogger()
    # Root level is the minimum of what any handler needs (DEBUG for the debug file).
    # `level` sets the floor for callers that want to suppress even the debug log.
    root.setLevel(getattr(logging, level.upper(), logging.DEBUG))
    root.handlers.clear()

    # ------------------------------------------------------------------
    # 1. MAIN file handler — WARNING+ plus whitelisted INFO
    # ------------------------------------------------------------------
    if main_log_file:
        Path(main_log_file).parent.mkdir(parents=True, exist_ok=True)
        main_handler = logging.handlers.RotatingFileHandler(
            main_log_file,
            maxBytes=max_bytes,
            backupCount=backup_count,
            encoding="utf-8",
        )
        main_handler.setLevel(logging.INFO)   # filter will tighten further
        main_handler.addFilter(MainLogFilter())
        main_handler.setFormatter(MainFormatter(use_color=False))
        root.addHandler(main_handler)

    # ------------------------------------------------------------------
    # 2. DEBUG file handler — everything
    # ------------------------------------------------------------------
    if debug_log_file:
        Path(debug_log_file).parent.mkdir(parents=True, exist_ok=True)
        debug_handler = logging.handlers.RotatingFileHandler(
            debug_log_file,
            maxBytes=max_bytes,
            backupCount=backup_count,
            encoding="utf-8",
        )
        debug_handler.setLevel(logging.DEBUG)
        debug_handler.setFormatter(DebugFormatter())
        root.addHandler(debug_handler)

    # ------------------------------------------------------------------
    # 3. Console — mirrors main log (WARNING+ only), coloured on TTY
    # ------------------------------------------------------------------
    if console_output:
        console = logging.StreamHandler(sys.stdout)
        console.setLevel(logging.WARNING)
        is_tty = hasattr(sys.stdout, "isatty") and sys.stdout.isatty()
        console.setFormatter(MainFormatter(use_color=is_tty))
        root.addHandler(console)

    # ------------------------------------------------------------------
    # Silence chatty third-party loggers in the main log
    # ------------------------------------------------------------------
    for noisy in (
        "uvicorn.access",
        "uvicorn.error",
        "httpx",
        "httpcore",
        "httpcore.http11",
        "httpcore.connection",
        "yfinance",
        "urllib3",
        "asyncio",
        "apscheduler",
        "websockets",
        # DB drivers — every SQL query/operation is DEBUG, far too noisy
        "aiosqlite",
        "peewee",
        "sqlalchemy.engine",
        # IBKR wire protocol is DEBUG flood; WARNING keeps connect/disconnect events
        "ib_insync.client",
        "ib_insync.wrapper",
    ):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    # ------------------------------------------------------------------
    # Demote benign IBKR connectivity blips (1100/1102) from ERROR→INFO
    # ib_insync logs these at ERROR but they are non-critical auto-recovery events
    # ------------------------------------------------------------------
    _ibkr_filter = IBKRConnectivityFilter()
    for ibkr_logger_name in ("ib_insync", "ib_insync.wrapper", "ib_insync.client"):
        logging.getLogger(ibkr_logger_name).addFilter(_ibkr_filter)

    return root


# ---------------------------------------------------------------------------
# Helpers kept for callers that import them directly
# ---------------------------------------------------------------------------

class TradingLogger:
    """Thin wrapper — exists for backward compat with existing call sites."""

    def __init__(self, name: str = "apex.trading"):
        self.logger = logging.getLogger(name)

    def order_placed(self, symbol, side, quantity, order_type, price=None, order_id=None):
        self.logger.info(
            f"Order placed: {side} {quantity} {symbol} @ {order_type}"
            + (f" ${price:.2f}" if price else ""),
            extra={"event": "order_placed", "symbol": symbol},
        )

    def order_filled(self, symbol, side, quantity, fill_price, expected_price,
                     slippage_bps, commission, order_id=None):
        self.logger.info(
            f"Order filled: {side} {quantity} {symbol} @ ${fill_price:.4f} "
            f"(slippage {slippage_bps:.1f}bps, comm ${commission:.2f})",
            extra={"event": "order_filled", "symbol": symbol},
        )

    def signal_generated(self, symbol, signal_strength, confidence, direction, regime):
        self.logger.info(
            f"{direction} signal: {symbol} strength={signal_strength:.3f} "
            f"conf={confidence:.2f} regime={regime}",
            extra={"event": "signal_generated", "symbol": symbol},
        )

    def risk_alert(self, alert_type, current_value, threshold, action):
        self.logger.warning(
            f"Risk alert [{alert_type}]: value={current_value:.4f} "
            f"threshold={threshold:.4f} → {action}",
            extra={"event": "risk_alert"},
        )

    def circuit_breaker_triggered(self, reason, value, threshold):
        self.logger.error(
            f"Circuit breaker triggered: {reason} (value={value:.4f}, limit={threshold:.4f})",
            extra={"event": "circuit_breaker_triggered"},
        )

    def position_update(self, symbol, old_quantity, new_quantity, reason):
        self.logger.info(
            f"Position updated: {symbol} {old_quantity}→{new_quantity} ({reason})",
            extra={"event": "position_update", "symbol": symbol},
        )

    def performance_metric(self, metric_name, value, unit=""):
        self.logger.info(
            f"Performance: {metric_name}={value}{unit}",
            extra={"event": "performance_metric"},
        )


def get_trading_logger(name: str = "apex.trading") -> TradingLogger:
    return TradingLogger(name)


# Compat stubs
class LogContext:
    def __init__(self, service="apex-trading", environment="production",
                 version="2.0.0", hostname=""):
        import socket
        self.service = service
        self.environment = environment
        self.version = version
        self.hostname = hostname or socket.gethostname()


def setup_structlog():
    pass  # structlog integration not needed with two-channel file logging


_initialized = False


def init_logging():
    global _initialized
    if not _initialized:
        setup_logging()
        _initialized = True
