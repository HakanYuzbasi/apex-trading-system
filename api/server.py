"""
api/server.py - APEX Trading API Server

Core backend for the SOTA UI.
Exposes REST endpoints and WebSockets for real-time data streaming.
Reads state from trading_state.json written by the ApexTrader.
"""

import asyncio
import collections
import logging
import os
import random
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from uuid import uuid4
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Depends, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import PlainTextResponse
import json
import math
import re
from datetime import datetime
from pydantic import BaseModel, Field

from config import ApexConfig
from core.logging_config import setup_logging
from core.request_context import request_context as request_log_context
from core.trading_control import (
    read_control_state,
    request_kill_switch_reset,
    request_governor_policy_reload,
    request_equity_reconciliation_latch,
    request_broker_mode_change,
    get_active_broker_mode,
)
from risk.governor_policy import (
    GovernorPolicyRepository,
    PolicyPromotionService,
    PromotionStatus,
)
from risk.social_decision_audit import SocialDecisionAuditRepository

logger = logging.getLogger("api")
from api.auth import (
    authenticate_websocket,
    require_role,
    require_user,
    verify_auth_runtime_prerequisites,
)
from api.ws_manager import manager
from api.dependencies import (
    CONTROL_COMMAND_FILE,
    GOVERNOR_POLICY_DIR,
    SOCIAL_DECISION_AUDIT_FILE,
    SOCIAL_DECISION_AUDIT_LEGACY_FILE,
    STATE_FILE,
    _mtime_ns,
    _state_is_fresh,
    async_read_trading_state,
    read_trading_state,
    sanitize_execution_metrics,
)

try:
    from prometheus_client import Counter, Gauge, Histogram, generate_latest, CONTENT_TYPE_LATEST
    PROMETHEUS_AVAILABLE = True
except ImportError:
    PROMETHEUS_AVAILABLE = False
    CONTENT_TYPE_LATEST = "text/plain; version=0.0.4"

setup_logging(
    level=ApexConfig.LOG_LEVEL,
    log_file=ApexConfig.LOG_FILE,
    json_format=False,
    console_output=True,
    max_bytes=ApexConfig.LOG_MAX_BYTES,
    backup_count=ApexConfig.LOG_BACKUP_COUNT,
    main_log_file="/private/tmp/apex_main.log",
    debug_log_file="/private/tmp/apex_debug.log",
)


# Alert aggregator for reducing log noise
from core.alert_aggregator import get_alert_aggregator
alert_agg = get_alert_aggregator(logger)

PREFLIGHT_STATUS_FILE = ApexConfig.DATA_DIR / "preflight_status.json"

# Rolling equity curve buffer fed by stream_trading_state().
# Stores (iso_timestamp, equity_float) tuples, capped at 2000 samples (~33 min at 1 Hz).
# Used by /ops/advanced-metrics so the endpoint works without an in-process engine.
_equity_curve_buffer: collections.deque = collections.deque(maxlen=2000)
_equity_curve_lock = threading.Lock()
_equity_curve_last_ts: Optional[str] = None  # deduplicate same-timestamp ticks

_preflight_metrics_lock = threading.Lock()
_preflight_metrics_mtime_ns: Optional[int] = None

_SHADOW_TERMINAL_LOG_FILE = Path("/private/tmp/apex_main.log")
_SHADOW_TERMINAL_MARKER = "[SHADOW MODE] PPO suggests:"
_shadow_terminal_lock = threading.Lock()
_shadow_terminal_lines: collections.deque = collections.deque(maxlen=80)
_shadow_terminal_file_size = 0
_shadow_terminal_mtime_ns: Optional[int] = None


def _as_finite_float(value: Any) -> Optional[float]:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(parsed):
        return None
    return parsed


def _parse_curve_timestamp(value: Any) -> Optional[datetime]:
    token = str(value or "").strip()
    if not token:
        return None
    try:
        if token.endswith("Z"):
            return datetime.fromisoformat(token.replace("Z", "+00:00"))
        return datetime.fromisoformat(token)
    except ValueError:
        return None


def _get_equity_curve_samples() -> List[Tuple[str, float]]:
    from api.dependencies import get_engine

    curve_raw: List[Tuple[str, float]] = []
    engine = get_engine()
    if engine is not None:
        pt = getattr(engine, "performance_tracker", None)
        if pt is not None:
            try:
                curve_raw = list(getattr(pt, "equity_curve", []))
            except Exception as exc:
                logger.debug("Could not read in-process equity curve: %s", exc)

    if not curve_raw:
        with _equity_curve_lock:
            curve_raw = list(_equity_curve_buffer)

    if not curve_raw:
        state = read_trading_state()
        ts = str(state.get("timestamp") or "").strip()
        equity = _as_finite_float(state.get("capital") or state.get("equity"))
        if ts and equity is not None and equity > 0:
            curve_raw = [(ts, equity)]

    sanitized: List[Tuple[str, float]] = []
    for ts, equity in curve_raw:
        parsed_equity = _as_finite_float(equity)
        if parsed_equity is None or parsed_equity <= 0:
            continue
        timestamp = str(ts or "").strip()
        if not timestamp:
            continue
        sanitized.append((timestamp, parsed_equity))

    return sanitized


def _median(values: List[float]) -> Optional[float]:
    if not values:
        return None
    ordered = sorted(values)
    mid = len(ordered) // 2
    if len(ordered) % 2 == 1:
        return ordered[mid]
    return (ordered[mid - 1] + ordered[mid]) / 2.0


def _build_pitch_metrics(
    current_state: Dict[str, Any],
    *,
    equity_override: Optional[float] = None,
    source: str,
) -> Dict[str, Any]:
    curve_samples = _get_equity_curve_samples()
    values = [equity for _, equity in curve_samples]
    timestamps = [_parse_curve_timestamp(ts) for ts, _ in curve_samples]

    interval_seconds: List[float] = []
    for idx in range(1, len(values)):
        prev_ts = timestamps[idx - 1]
        current_ts = timestamps[idx]
        if prev_ts is not None and current_ts is not None:
            delta = (current_ts - prev_ts).total_seconds()
            if delta > 0:
                interval_seconds.append(delta)

    max_drawdown = 0.0
    if values:
        peak = values[0]
        for value in values:
            peak = max(peak, value)
            if peak > 0:
                # Proper drawdown formula (peak - value) / peak
                drawdown = (peak - value) / peak
                max_drawdown = max(max_drawdown, drawdown)

    sharpe_ratio: Optional[float] = None
    engine_sharpe = current_state.get("sharpe_ratio")
    if engine_sharpe is not None and isinstance(engine_sharpe, (int, float)) and engine_sharpe > 0:
        sharpe_ratio = round(float(engine_sharpe), 2)
    elif len(values) >= 2:
        # Prevent high-frequency statistical illusion! 
        # Annualizing micro-seconds blows up the Sharpe multiplier.
        # Use stable Calmar-like proxy based on total curve curve yield vs drawdown.
        curve_total_pnl = values[-1] - values[0]
        baseline_capital = values[0]
        if baseline_capital > 0:
            sharpe_ratio = round(((curve_total_pnl / baseline_capital) / max(max_drawdown, 0.02)) * 0.8, 2)
        else:
            sharpe_ratio = 0.0
    else:
        sharpe_ratio = 0.0

    equity = _as_finite_float(equity_override)
    if equity is None or equity <= 0:
        equity = _as_finite_float(current_state.get("capital") or current_state.get("equity"))

    realized_pnl_today = _as_finite_float(
        current_state.get("daily_pnl_realized", current_state.get("realized_pnl", current_state.get("daily_pnl")))
    )
    active_margin = _as_finite_float(current_state.get("active_margin"))
    margin_utilization = (
        round(active_margin / equity, 6)
        if active_margin is not None and equity is not None and equity > 0
        else None
    )
    sample_interval_seconds = _median(interval_seconds)

    return {
        "available": any(
            value is not None
            for value in (equity, realized_pnl_today, active_margin, sharpe_ratio)
        ) or len(curve_samples) > 0,
        "error": None,
        "timestamp": current_state.get("timestamp", datetime.utcnow().isoformat() + "Z"),
        "source": source,
        "equity": round(equity, 2) if equity is not None else None,
        "realized_pnl_today": round(realized_pnl_today, 2) if realized_pnl_today is not None else None,
        "active_margin": round(active_margin, 2) if active_margin is not None else None,
        "active_margin_utilization": margin_utilization,
        "sharpe_ratio": sharpe_ratio,
        "max_drawdown": round(max_drawdown, 6) if values else None,
        "curve_points": len(curve_samples),
        "sample_interval_seconds": round(sample_interval_seconds, 2) if sample_interval_seconds is not None else None,
    }


def _refresh_shadow_terminal_lines() -> None:
    global _shadow_terminal_file_size, _shadow_terminal_mtime_ns

    try:
        stat = _SHADOW_TERMINAL_LOG_FILE.stat()
    except OSError:
        return

    with _shadow_terminal_lock:
        if _shadow_terminal_mtime_ns == stat.st_mtime_ns and _shadow_terminal_file_size == stat.st_size:
            return

        if stat.st_size < _shadow_terminal_file_size:
            _shadow_terminal_lines.clear()
            _shadow_terminal_file_size = 0

        try:
            with _SHADOW_TERMINAL_LOG_FILE.open("r", encoding="utf-8", errors="ignore") as handle:
                if _shadow_terminal_file_size == 0 and stat.st_size > 262_144:
                    handle.seek(max(stat.st_size - 262_144, 0))
                    handle.readline()
                else:
                    handle.seek(_shadow_terminal_file_size)

                for line in handle:
                    # Capture actual neural execution logs
                    if _SHADOW_TERMINAL_MARKER in line:
                        cleaned = line.rstrip()
                        if cleaned:
                            _shadow_terminal_lines.append(cleaned)

                _shadow_terminal_file_size = handle.tell()
                _shadow_terminal_mtime_ns = stat.st_mtime_ns
        except Exception as exc:
            logger.debug("Shadow terminal tail failed: %s", exc)


def _get_shadow_terminal_payload(state: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    # Phase 1: Try to extract from real-time state (High-Fidelity)
    if state and "shadow_terminal" in state:
        st_state = state["shadow_terminal"]
        if isinstance(st_state, dict) and "lines" in st_state:
            return st_state

    # Phase 2: Fallback to log-file tailing (Institutional Robustness)
    _refresh_shadow_terminal_lines()
    with _shadow_terminal_lock:
        last_updated = None
        if _shadow_terminal_mtime_ns is not None:
            last_updated = datetime.utcfromtimestamp(_shadow_terminal_mtime_ns / 1_000_000_000).isoformat() + "Z"
        return {
            "available": _SHADOW_TERMINAL_LOG_FILE.exists(),
            "marker": _SHADOW_TERMINAL_MARKER,
            "lines": list(_shadow_terminal_lines),
            "last_updated": last_updated,
        }


def _update_preflight_metrics_from_file() -> None:
    """Refresh preflight Prometheus gauges from latest persisted preflight status."""
    global _preflight_metrics_mtime_ns
    if not PROMETHEUS_AVAILABLE:
        return

    mtime_ns = _mtime_ns(PREFLIGHT_STATUS_FILE)
    with _preflight_metrics_lock:
        if _preflight_metrics_mtime_ns == mtime_ns:
            return

    if mtime_ns is None:
        return

    try:
        payload = json.loads(PREFLIGHT_STATUS_FILE.read_text(encoding="utf-8"))
    except Exception as exc:
        logger.debug("Could not read preflight status file: %s", exc)
        return

    if not isinstance(payload, dict):
        return

    total_checks = float(payload.get("total_checks", 0) or 0)
    passed_checks = float(payload.get("passed_checks", 0) or 0)
    pass_rate = float(payload.get("pass_rate", 0.0) or 0.0)
    exit_code = float(payload.get("exit_code", 0) or 0)
    run_at_raw = str(payload.get("run_at", "") or "")
    run_at_ts = 0.0
    if run_at_raw:
        try:
            run_at_ts = datetime.fromisoformat(run_at_raw.replace("Z", "+00:00")).timestamp()
        except ValueError:
            run_at_ts = 0.0

    PREFLIGHT_CHECKS_TOTAL.set(total_checks)
    PREFLIGHT_CHECKS_PASSED.set(passed_checks)
    PREFLIGHT_PASS_RATE.set(pass_rate)
    PREFLIGHT_LAST_EXIT_CODE.set(exit_code)
    PREFLIGHT_LAST_RUN_TIMESTAMP.set(run_at_ts)

    results = payload.get("results", [])
    if isinstance(results, list):
        for row in results:
            if not isinstance(row, dict):
                continue
            check_name = str(row.get("name", "") or "").strip()
            if not check_name:
                continue
            check_ok = 1.0 if bool(row.get("ok", False)) else 0.0
            PREFLIGHT_CHECK_STATUS.labels(check=check_name).set(check_ok)

    with _preflight_metrics_lock:
        _preflight_metrics_mtime_ns = mtime_ns  # noqa: F841 — global cache, read in next call

app = FastAPI(title="APEX Trading API", version="2.0.0")

from api.routers.public import router as public_router
app.include_router(public_router)

try:
    from api.routers.backtest import router as backtest_router
    app.include_router(backtest_router)
except Exception as e:
    logging.getLogger("api").warning("Backtest router not loaded: %s", e)

if PROMETHEUS_AVAILABLE:
    HTTP_REQUESTS_TOTAL = Counter(
        "apex_api_http_requests_total",
        "Total number of HTTP requests handled by API",
        ["method", "path", "status"],
    )
    HTTP_REQUEST_DURATION = Histogram(
        "apex_api_http_request_duration_seconds",
        "HTTP request duration in seconds",
        ["method", "path"],
        buckets=(0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0),
    )
    HTTP_REQUESTS_IN_PROGRESS = Gauge(
        "apex_api_http_requests_in_progress",
        "Number of in-flight HTTP requests",
        ["method", "path"],
    )
    WEBSOCKET_CONNECTIONS = Gauge(
        "apex_api_websocket_connections",
        "Number of active websocket connections",
    )
    WEBSOCKET_MESSAGES_TOTAL = Counter(
        "apex_api_websocket_messages_total",
        "Total websocket messages handled",
        ["direction"],
    )
    PREFLIGHT_PASS_RATE = Gauge(
        "apex_preflight_pass_rate",
        "Latest full-stack preflight pass rate (0-1).",
    )
    PREFLIGHT_CHECKS_TOTAL = Gauge(
        "apex_preflight_checks_total",
        "Number of checks executed in latest preflight run.",
    )
    PREFLIGHT_CHECKS_PASSED = Gauge(
        "apex_preflight_checks_passed_total",
        "Number of passing checks in latest preflight run.",
    )
    PREFLIGHT_LAST_RUN_TIMESTAMP = Gauge(
        "apex_preflight_last_run_timestamp",
        "Unix timestamp of the latest preflight run.",
    )
    PREFLIGHT_LAST_EXIT_CODE = Gauge(
        "apex_preflight_last_exit_code",
        "Latest preflight process exit code (0=pass).",
    )
    PREFLIGHT_CHECK_STATUS = Gauge(
        "apex_preflight_check_status",
        "Latest preflight status for each check (1=pass, 0=fail).",
        ["check"],
    )
else:
    HTTP_REQUESTS_TOTAL = None
    HTTP_REQUEST_DURATION = None
    HTTP_REQUESTS_IN_PROGRESS = None
    WEBSOCKET_CONNECTIONS = None
    WEBSOCKET_MESSAGES_TOTAL = None
    PREFLIGHT_PASS_RATE = None
    PREFLIGHT_CHECKS_TOTAL = None
    PREFLIGHT_CHECKS_PASSED = None
    PREFLIGHT_LAST_RUN_TIMESTAMP = None
    PREFLIGHT_LAST_EXIT_CODE = None
    PREFLIGHT_CHECK_STATUS = None

# Enable CORS for Next.js frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://localhost:3001",
        "http://127.0.0.1:3000",
        "http://127.0.0.1:3001",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def request_context_and_metrics_middleware(request: Request, call_next):
    method = request.method
    path = request.url.path
    request_id = request.headers.get("X-Request-ID") or f"req-{uuid4().hex[:12]}"
    correlation_id = request.headers.get("X-Correlation-ID") or request_id

    metric_labels = None
    if PROMETHEUS_AVAILABLE:
        metric_labels = {"method": method, "path": path}
        HTTP_REQUESTS_IN_PROGRESS.labels(**metric_labels).inc()

    start = time.perf_counter()
    status_code = 500

    with request_log_context(
        correlation_id=correlation_id,
        request_id=request_id,
        auto_generate=False,
    ):
        try:
            response = await call_next(request)
            status_code = response.status_code
        except Exception:
            logger.exception(
                "HTTP request failed",
                extra={"method": method, "path": path, "request_id": request_id},
            )
            raise
        finally:
            elapsed = time.perf_counter() - start
            
            # Improved logging: only log errors, slow requests, or sample successful requests
            should_log = False
            log_level = logging.INFO

            # Paths that are expected to be slow (long-poll, SSE, polling endpoints)
            _slow_exempt = ("/state", "/metrics", "/health", "/ws",
                            "/portfolio", "/status", "/cockpit", "/positions")
            is_slow_exempt = any(path.startswith(p) for p in _slow_exempt)

            if status_code >= 500:
                should_log = True
                log_level = logging.ERROR
            elif status_code >= 400:
                should_log = True
                log_level = logging.WARNING
            elif elapsed > 1.0 and not is_slow_exempt:  # Slow request (non-exempt)
                should_log = True
                log_level = logging.WARNING
            elif status_code < 400:
                # Sample 1% of successful requests to reduce noise
                should_log = random.random() < 0.01
                log_level = logging.DEBUG
            
            if should_log:
                msg = (
                    f"HTTP {method} {path} → {status_code} ({elapsed * 1000:.0f}ms)"
                    if log_level >= logging.WARNING
                    else "HTTP request completed"
                )
                logger.log(
                    log_level,
                    msg,
                    extra={
                        "method": method,
                        "path": path,
                        "status_code": status_code,
                        "duration_ms": round(elapsed * 1000, 3),
                        "request_id": request_id,
                    },
                )
            
            if PROMETHEUS_AVAILABLE and metric_labels is not None:
                HTTP_REQUEST_DURATION.labels(**metric_labels).observe(elapsed)
                HTTP_REQUESTS_TOTAL.labels(
                    method=method,
                    path=path,
                    status=str(status_code),
                ).inc()
                HTTP_REQUESTS_IN_PROGRESS.labels(**metric_labels).dec()

    response.headers["X-Request-ID"] = request_id
    return response

# SaaS auth: resolve user from PostgreSQL (JWT/API key) or legacy in-memory; set request.state.user
try:
    from services.auth.middleware import SaaSAuthMiddleware
    app.add_middleware(SaaSAuthMiddleware)
except Exception as e:
    logging.getLogger("api").warning("SaaS auth middleware not loaded: %s", e)

# Mount SaaS auth and feature routers
try:
    from services.auth.router import router as auth_router
    app.include_router(auth_router)
except Exception as e:
    logging.getLogger("api").warning("SaaS auth router not loaded: %s", e)

try:
    from services.backtest_validator.router import router as backtest_validator_router
    app.include_router(backtest_validator_router)
except Exception as e:
    alert_agg.add("router_load_failed", "Backtest validator router not loaded", data={"service": "backtest_validator", "error": str(e)})

try:
    from services.execution_simulator.router import router as execution_sim_router
    app.include_router(execution_sim_router)
except Exception as e:
    logging.getLogger("api").warning("Execution simulator router not loaded: %s", e)

try:
    from services.drift_monitor.router import router as drift_monitor_router
    app.include_router(drift_monitor_router)
except Exception as e:
    alert_agg.add("router_load_failed", "Drift monitor router not loaded", data={"service": "drift_monitor", "error": str(e)})

try:
    from services.compliance_copilot.router import router as compliance_copilot_router
    app.include_router(compliance_copilot_router)
except Exception as e:
    alert_agg.add("router_load_failed", "Compliance copilot router not loaded", data={"service": "compliance_copilot", "error": str(e)})

try:
    from services.portfolio_allocator.router import router as portfolio_allocator_router
    app.include_router(portfolio_allocator_router)
except Exception as e:
    alert_agg.add("router_load_failed", "Portfolio allocator router not loaded", data={"service": "portfolio_allocator", "error": str(e)})

try:
    from services.mandate_copilot.router import router as mandate_copilot_router
    app.include_router(mandate_copilot_router)
except Exception as e:
    alert_agg.add("router_load_failed", "Mandate copilot router not loaded", data={"service": "mandate_copilot", "error": str(e)})

# Health check endpoints
try:
    from api.health import router as health_router
    app.include_router(health_router)
except Exception as e:
    alert_agg.add("router_load_failed", "Health router not loaded", data={"service": "health", "error": str(e)})

try:
    from services.tca.router import router as tca_router
    app.include_router(tca_router)
except Exception as e:
    alert_agg.add("router_load_failed", "TCA router not loaded", data={"service": "tca", "error": str(e)})

try:
    from services.replay_inspector.router import router as replay_inspector_router
    app.include_router(replay_inspector_router)
except Exception as e:
    alert_agg.add("router_load_failed", "Replay inspector router not loaded", data={"service": "replay_inspector", "error": str(e)})

# Broker Service Routers
try:
    from services.broker.router import router as broker_router, portfolio_router
    app.include_router(broker_router)
    app.include_router(portfolio_router)
except Exception as e:
    alert_agg.add("router_load_failed", "Broker router not loaded", data={"service": "broker", "error": str(e)})



class KillSwitchResetRequest(BaseModel):
    reason: str = Field(min_length=8, max_length=500)


class EquityReconciliationLatchRequest(BaseModel):
    block_entries: bool
    reason: str = Field(min_length=8, max_length=500)


class GovernorPolicyApproveRequest(BaseModel):
    policy_id: str = Field(min_length=8, max_length=200)
    reason: str = Field(min_length=8, max_length=500)


class GovernorPolicyRollbackRequest(BaseModel):
    asset_class: str = Field(min_length=2, max_length=32)
    regime: str = Field(min_length=1, max_length=64)
    reason: str = Field(min_length=8, max_length=500)
    target_version: Optional[str] = Field(default=None, min_length=1, max_length=128)


def _governor_repo() -> GovernorPolicyRepository:
    return GovernorPolicyRepository(GOVERNOR_POLICY_DIR)


def _governor_service(repo: GovernorPolicyRepository) -> PolicyPromotionService:
    return PolicyPromotionService(
        repository=repo,
        environment=ApexConfig.ENVIRONMENT,
        live_trading=ApexConfig.LIVE_TRADING,
        auto_promote_non_prod=ApexConfig.GOVERNOR_AUTO_PROMOTE_NON_PROD,
    )


def _social_audit_repo() -> SocialDecisionAuditRepository:
    fallback_filepaths: List[Path] = []
    if SOCIAL_DECISION_AUDIT_LEGACY_FILE != SOCIAL_DECISION_AUDIT_FILE:
        fallback_filepaths.append(SOCIAL_DECISION_AUDIT_LEGACY_FILE)
    return SocialDecisionAuditRepository(
        SOCIAL_DECISION_AUDIT_FILE,
        fallback_filepaths=fallback_filepaths,
    )


def _social_audit_repo_for_user(user_id: str) -> SocialDecisionAuditRepository:
    """Resolve tenant-scoped social audit repository with global legacy fallbacks."""
    tenant = str(user_id or "").strip()
    if not tenant:
        return _social_audit_repo()

    tenant_root = ApexConfig.DATA_DIR if tenant == "default" else (ApexConfig.DATA_DIR / "users" / tenant)
    tenant_runtime_file = Path(
        os.getenv(
            "APEX_SOCIAL_DECISION_AUDIT_FILE",
            str(tenant_root / "runtime" / "social_governor_decisions.jsonl"),
        )
    )
    tenant_legacy_file = Path(
        os.getenv(
            "APEX_SOCIAL_DECISION_AUDIT_LEGACY_FILE",
            str(tenant_root / "audit" / "social_governor_decisions.jsonl"),
        )
    )

    fallback_filepaths: List[Path] = []
    if tenant_legacy_file != tenant_runtime_file:
        fallback_filepaths.append(tenant_legacy_file)
    # Include global files as final fallback so historical events remain visible.
    if SOCIAL_DECISION_AUDIT_FILE not in {tenant_runtime_file, tenant_legacy_file}:
        fallback_filepaths.append(SOCIAL_DECISION_AUDIT_FILE)
    if SOCIAL_DECISION_AUDIT_LEGACY_FILE not in {
        tenant_runtime_file,
        tenant_legacy_file,
        SOCIAL_DECISION_AUDIT_FILE,
    }:
        fallback_filepaths.append(SOCIAL_DECISION_AUDIT_LEGACY_FILE)

    return SocialDecisionAuditRepository(
        tenant_runtime_file,
        fallback_filepaths=fallback_filepaths,
    )

# --------------------------------------------------------------------------------
# Real-time State Streaming
# --------------------------------------------------------------------------------

async def stream_trading_state():
    """Stream tenant-scoped trading state to connected websocket clients."""
    last_state = None
    last_equity_update_time = 0
    cached_equity_data_by_tenant: Dict[str, Dict] = {}

    while True:
        try:
            # Redis-first read: sub-ms when Redis is warm, falls back to mtime cache
            current_state = await async_read_trading_state()

            # Append equity snapshot to the rolling curve buffer for /ops/advanced-metrics.
            # Uses the state timestamp as the deduplication key so we only add one point
            # per engine write cycle (~10s) rather than one per API poll cycle (~1s).
            global _equity_curve_last_ts
            _ts = current_state.get("timestamp")
            _eq = current_state.get("capital") or current_state.get("equity")
            if _ts and _ts != _equity_curve_last_ts and isinstance(_eq, (int, float)) and _eq > 0:
                with _equity_curve_lock:
                    _equity_curve_buffer.append((_ts, float(_eq)))
                _equity_curve_last_ts = _ts

            # Periodically fetch aggregated equity (every ~120 seconds to avoid API limits)
            now_ts = time.time()
            if now_ts - last_equity_update_time > 120:
                from services.broker.service import broker_service
                try:
                    tenant_ids = await broker_service.list_tenant_ids()
                    results = await asyncio.gather(
                        *[broker_service.get_tenant_equity_snapshot(tenant_id) for tenant_id in tenant_ids],
                        return_exceptions=True,
                    )
                    for tenant_id, result in zip(tenant_ids, results):
                        if isinstance(result, Exception):
                            logger.debug("Tenant equity snapshot failed for %s: %s", tenant_id, result)
                            continue
                        cached_equity_data_by_tenant[tenant_id] = result
                    last_equity_update_time = now_ts
                except Exception as e:
                    logger.debug(f"Aggregated equity background task failed: {e}")

            if current_state != last_state or cached_equity_data_by_tenant:
                for tenant_id in manager.connected_tenants():
                    if tenant_id == "public":
                        continue  # Handled by a dedicated sanitized loop in public router

                    is_admin = manager.is_tenant_admin(tenant_id)
                    update = {
                        "tenant_id": tenant_id
                    }
                    pitch_metrics = None
                    shadow_terminal = None

                    if is_admin:
                        update.update({
                            "timestamp": current_state.get("timestamp", datetime.utcnow().isoformat() + "Z"),
                            "capital": current_state.get("capital", 0),
                            "initial_capital": current_state.get("initial_capital", 0),
                            "starting_capital": current_state.get("starting_capital", 0),
                            "positions": current_state.get("positions", {}),
                            "daily_pnl": current_state.get("daily_pnl", 0),
                            "daily_pnl_realized": current_state.get("daily_pnl_realized", current_state.get("daily_pnl", 0)),
                            "daily_pnl_source": current_state.get("daily_pnl_source", "equity_delta"),
                            "total_pnl": current_state.get("total_pnl", 0),
                            "max_drawdown": current_state.get("max_drawdown", 0),
                            "sharpe_ratio": current_state.get("sharpe_ratio", 0),
                            "win_rate": current_state.get("win_rate", 0),
                            "sortino_ratio": current_state.get("sortino_ratio", 0),
                            "calmar_ratio": current_state.get("calmar_ratio", 0),
                            "profit_factor": current_state.get("profit_factor", 0),
                            "alpha_retention": current_state.get("alpha_retention", 0),
                            "sector_exposure": current_state.get("sector_exposure", {}),
                            "open_positions": current_state.get("open_positions", 0),
                            "max_positions": current_state.get("max_positions", ApexConfig.MAX_POSITIONS),
                            "total_trades": current_state.get("total_trades", 0),
                            "meta_confidence_score": current_state.get("meta_confidence_score", 1.0),
                            "bayesian_vol_prob": current_state.get("bayesian_vol_prob", 0.0),
                            "correlation_matrix": current_state.get("correlation_matrix", {}),
                            # Regime / market context
                            "regime": current_state.get("regime") or current_state.get("market_regime", "neutral"),
                            "vix": current_state.get("vix") or current_state.get("vix_level", 0),
                            "survival_probability": current_state.get("survival_probability", 1.0),
                            # Live KPI fields written by the trading engine
                            "realized_pnl": current_state.get("realized_pnl", 0),
                            "unrealized_pnl": current_state.get("unrealized_pnl", 0),
                            "active_margin": current_state.get("active_margin", 0),
                            "leverage_limit": current_state.get("leverage_limit", 1.0),
                            "latency_heatmap": current_state.get("latency_heatmap", []),
                            "broker_mode": current_state.get("broker_mode", "both"),
                            "broker_positions": current_state.get("broker_positions", []),
                            "sentiment_health": current_state.get("sentiment_health", {}),
                            "social_pulse": current_state.get("social_pulse", {}),
                            "strategy_allocation": current_state.get("strategy_allocation", {}),
                            "hedge_status": current_state.get("hedge_status", "Inactive"),
                            "broker_heartbeats": current_state.get("broker_heartbeats", {}),
                        })
                        shadow_terminal = _get_shadow_terminal_payload(current_state)
                    else:
                        update.update({
                            "timestamp": datetime.utcnow().isoformat() + "Z",
                            "capital": 0,
                            "initial_capital": 0,
                            "starting_capital": 0,
                            "positions": {},
                            "daily_pnl": 0,
                            "daily_pnl_realized": 0,
                            "daily_pnl_source": "equity_delta",
                            "total_pnl": 0,
                            "max_drawdown": 0,
                            "sharpe_ratio": 0,
                            "win_rate": 0,
                            "sector_exposure": {},
                            "open_positions": 0,
                            "max_positions": ApexConfig.MAX_POSITIONS,
                            "total_trades": 0,
                        })

                    tenant_equity = cached_equity_data_by_tenant.get(tenant_id)
                    if tenant_equity:
                        aggregated_equity = float(tenant_equity.get("total_equity", 0.0) or 0.0)
                        update["aggregated_equity"] = aggregated_equity
                        update["total_equity"] = aggregated_equity
                        if is_admin and aggregated_equity > 0:
                            # Keep WS KPI stream aligned with broker-aggregated equity.
                            update["capital"] = aggregated_equity
                        update["equity_breakdown"] = tenant_equity.get("breakdown", [])

                    if is_admin and tenant_id == "unified":
                        pitch_metrics = _build_pitch_metrics(
                            current_state,
                            equity_override=update.get("total_equity") if isinstance(update.get("total_equity"), (int, float)) else None,
                            source="ws_stream",
                        )
                        update["pitch_metrics"] = pitch_metrics
                        update["shadow_terminal"] = shadow_terminal
                    elif is_admin:
                        # For session-specific updates, we still send the terminal but skip the top-ribbon pitch metrics
                        # to prevent the front-end from flickering between session-level and portfolio-level Sharpe ratios.
                        update["shadow_terminal"] = shadow_terminal

                    # Delta-encode: sends full state_update every 30 ticks, tiny
                    # state_delta otherwise — drops wire payload by ~90%.
                    payload = manager.compute_delta_payload(tenant_id, update)
                    if not payload:
                        # Nothing changed this tick — skip broadcast entirely
                        continue

                    def increment_metrics():
                        if PROMETHEUS_AVAILABLE and WEBSOCKET_MESSAGES_TOTAL is not None:
                            WEBSOCKET_MESSAGES_TOTAL.labels(direction="outbound").inc()

                    await manager.broadcast_to_tenant(tenant_id, payload, increment_metrics)
                last_state = current_state

        except Exception as e:
            logger.error(f"Error in broadcast loop: {e}")

        await asyncio.sleep(max(float(getattr(ApexConfig, "POLL_INTERVAL_SECONDS", 3.0)), 1.0))


async def _collect_user_broker_overlay(user_id: str) -> Dict[str, object]:
    """
    Collect broker-derived status overrides for a user.
    Keeps `/status` aligned with `/portfolio/balance` + `/portfolio/positions`
    so downstream consumers do not diverge when daemon state is stale/flat.
    """
    try:
        from services.broker.service import broker_service
    except Exception as exc:  # SWALLOW: optional import path during partial startup
        logger.debug("Broker overlay unavailable (import): %s", exc)
        return {}

    snapshot_task = broker_service.get_tenant_equity_snapshot(user_id)
    positions_task = broker_service.get_positions(user_id)
    snapshot_result, positions_result = await asyncio.gather(
        snapshot_task,
        positions_task,
        return_exceptions=True,
    )

    overlay: Dict[str, object] = {}
    if isinstance(snapshot_result, Exception):
        logger.debug("Broker equity overlay failed for %s: %s", user_id, snapshot_result)
    elif isinstance(snapshot_result, dict):
        total_equity = snapshot_result.get("total_equity")
        breakdown = snapshot_result.get("breakdown")
        # Only override capital when the broker snapshot has real data (non-empty breakdown
        # with positive equity). An empty breakdown means no broker connections are
        # configured — we should fall back to the trading engine's state file value.
        if isinstance(total_equity, (int, float)) and float(total_equity) > 0 and breakdown:
            overlay["capital"] = float(total_equity)
            overlay["aggregated_equity"] = float(total_equity)
        if isinstance(breakdown, list):
            overlay["equity_breakdown"] = breakdown

    if isinstance(positions_result, Exception):
        logger.debug("Broker positions overlay failed for %s: %s", user_id, positions_result)
    elif isinstance(positions_result, list):
        if len(positions_result) == 0:
            return overlay
        equity_positions = 0
        option_positions = 0
        aggregated_unrealized = 0.0
        for row in positions_result:
            rec = row if isinstance(row, dict) else {}
            security_type = str(rec.get("security_type", "")).upper()
            symbol = str(rec.get("symbol", "")).strip().upper()
            symbol_compact = re.sub(r"\s+", "", symbol)
            has_option_fields = bool(rec.get("right")) or bool(rec.get("strike")) or bool(rec.get("expiry"))
            is_occ_option = bool(re.match(r"^[A-Z]{1,6}\d{6}[CP]\d{8}$", symbol_compact))
            is_option = "OPT" in security_type or has_option_fields or is_occ_option
            if is_option:
                option_positions += 1
            else:
                equity_positions += 1
            try:
                aggregated_unrealized += float(rec.get("unrealized_pl", 0.0) or 0.0)
            except Exception:
                continue
        overlay["open_positions"] = equity_positions
        overlay["option_positions"] = option_positions
        overlay["open_positions_total"] = equity_positions + option_positions
        overlay["aggregated_positions"] = equity_positions
        overlay["aggregated_positions_total"] = equity_positions + option_positions
        if not math.isclose(aggregated_unrealized, 0.0, abs_tol=1e-9):
            overlay["total_pnl"] = aggregated_unrealized

    return overlay

@app.on_event("startup")
async def startup_event():
    logger.info("APEX API Server Starting...")
    logger.info(f"Reading state from: {STATE_FILE}")
    await verify_auth_runtime_prerequisites()
    asyncio.create_task(stream_trading_state())


@app.get("/metrics", include_in_schema=False)
async def get_metrics(request: Request):
    """Prometheus scrape endpoint."""
    if not PROMETHEUS_AVAILABLE:
        return PlainTextResponse("prometheus_client unavailable\n", status_code=503)

    token = os.getenv("APEX_METRICS_TOKEN", "").strip()
    if token:
        supplied = request.headers.get("X-Metrics-Token")
        if supplied != token:
            return PlainTextResponse("forbidden\n", status_code=403)

    _update_preflight_metrics_from_file()
    return PlainTextResponse(generate_latest(), media_type=CONTENT_TYPE_LATEST)

# --------------------------------------------------------------------------------
# REST Endpoints
# --------------------------------------------------------------------------------

@app.get("/status")
async def get_status(user=Depends(require_user)):
    """Get daemon execution status. Only populated for admins until Phase 9 execution split."""
    broker_mode = str(getattr(ApexConfig, "BROKER_MODE", "both")).lower()
    primary_execution_broker = str(
        os.getenv(
            "APEX_PRIMARY_EXECUTION_BROKER",
            getattr(ApexConfig, "PRIMARY_EXECUTION_BROKER", "alpaca"),
        )
    ).lower()
    
    if datetime.utcnow().weekday() >= 5 and primary_execution_broker in ["ibkr", "both"]:
        primary_execution_broker = "alpaca"
    if not user.has_role("admin"):
        safe_metrics = sanitize_execution_metrics({})
        return {
            "status": "online",
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "broker_mode": broker_mode,
            "primary_execution_broker": primary_execution_broker,
            **safe_metrics,
        }
    state = read_trading_state()
    status_payload = dict(state)
    broker_overlay = await _collect_user_broker_overlay(str(getattr(user, "user_id", "") or ""))
    if broker_overlay:
        status_payload.update(broker_overlay)
        if "open_positions" in broker_overlay:
            option_positions = int(status_payload.get("option_positions", 0) or 0)
            status_payload["open_positions_total"] = max(
                int(status_payload.get("open_positions_total", 0) or 0),
                int(broker_overlay["open_positions"]) + option_positions,
            )
    safe_metrics = sanitize_execution_metrics(status_payload)
    if "aggregated_equity" in broker_overlay:
        safe_metrics["aggregated_equity"] = float(broker_overlay["aggregated_equity"])  # type: ignore[arg-type]
    if "equity_breakdown" in broker_overlay:
        safe_metrics["equity_breakdown"] = broker_overlay["equity_breakdown"]
    if "aggregated_positions" in broker_overlay:
        safe_metrics["aggregated_positions"] = int(broker_overlay["aggregated_positions"])  # type: ignore[arg-type]
    return {
        "status": "online" if _state_is_fresh(status_payload, ApexConfig.HEALTH_STALENESS_SECONDS) else "offline",
        "timestamp": status_payload.get("timestamp"),
        "broker_mode": broker_mode,
        "primary_execution_broker": primary_execution_broker,
        "broker_heartbeats": status_payload.get("broker_heartbeats", {}),
        **safe_metrics,
    }

@app.get("/positions")
async def get_positions(user=Depends(require_user)):
    """Get daemon active positions. Only populated for admins until Phase 9 execution split."""
    if not user.has_role("admin"):
        return []
    state = read_trading_state()
    positions = state.get("positions", {})

    result = []
    for symbol, data in positions.items():
        result.append({
            "symbol": symbol,
            "qty": data.get("qty", 0),
            "side": data.get("side", "LONG"),
            "entry": data.get("avg_price", 0),
            "current": data.get("current_price", 0),
            "pnl": data.get("pnl", 0),
            "pnl_pct": data.get("pnl_pct", 0),
            "signal": data.get("current_signal", 0),
            "signal_direction": data.get("signal_direction", "UNKNOWN")
        })

    return result

@app.get("/state")
async def get_full_state(user=Depends(require_user)):
    """Get complete daemon trading state."""
    if not user.has_role("admin"):
        return {}
    return read_trading_state()

@app.get("/health")
async def get_health(_user=Depends(require_user)):
    """Lightweight health check (auth required)."""
    state = read_trading_state()
    return {
        "status": "online" if _state_is_fresh(state, ApexConfig.HEALTH_STALENESS_SECONDS) else "offline",
        "timestamp": state.get("timestamp"),
        "stale_after_seconds": ApexConfig.HEALTH_STALENESS_SECONDS,
        "api": "ok",
    }

@app.get("/sectors")
async def get_sector_exposure(user=Depends(require_user)):
    """Get daemon sector exposure breakdown."""
    if not user.has_role("admin"):
        return {}
    state = read_trading_state()
    return state.get("sector_exposure", {})


@app.get("/ops/daily-report")
async def get_daily_report(date: str = None, _user=Depends(require_user)):
    """Daily P&L morning report — realized, unrealized, trend, system health."""
    try:
        from scripts.daily_report import build_daily_report
        import asyncio
        report = await asyncio.to_thread(build_daily_report, target_date=date, fetch_prices=False)
        return report
    except Exception as e:
        logger.error(f"Daily report failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/ops/tca")
async def get_tca_report(_user=Depends(require_user)):
    """Transaction Cost Analysis — execution quality, slippage & P&L attribution."""
    try:
        from monitoring.tca_report import build_tca_report
        import asyncio
        report = await asyncio.to_thread(build_tca_report)
        return report
    except Exception as e:
        logger.error(f"TCA report failed: {e}")


@app.get("/ops/walk-forward")
async def get_walkforward_report(_user=Depends(require_user)):
    """Walk-forward validation — rolling Sharpe, regime win-rate trend, component alpha."""
    try:
        from monitoring.walkforward_validator import build_walkforward_report
        import asyncio
        report = await asyncio.to_thread(build_walkforward_report)
        return report
    except Exception as e:
        logger.error(f"Walk-forward report failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/ops/stress-scenarios")
async def get_stress_scenarios(_user=Depends(require_user)):
    """Portfolio stress test — all predefined historical scenarios."""
    try:
        from risk.portfolio_stress_test import PortfolioStressTest
        import asyncio

        # Snapshot current positions and prices from running engine
        _positions: dict = {}
        _prices: dict = {}
        _capital: float = 1_000_000.0
        try:
            _trading_engine = app.state.trading_engine  # type: ignore[attr-defined]
            _positions = dict(getattr(_trading_engine, "positions", {}))
            _prices = dict(getattr(_trading_engine, "price_cache", {}))
            _capital = float(getattr(_trading_engine, "capital", 1_000_000.0))
        except Exception:
            pass

        def _run():
            engine = PortfolioStressTest(
                positions={str(k): int(v) for k, v in _positions.items() if v != 0},
                prices={str(k): float(v) for k, v in _prices.items() if v and v > 0},
                capital=_capital,
            )
            raw = engine.run_all_scenarios()
            out = []
            for scenario_id, r in raw.items():
                out.append({
                    "scenario_id": scenario_id,
                    "scenario_name": r.scenario_name,
                    "scenario_type": r.scenario_type.value,
                    "portfolio_pnl": round(r.portfolio_pnl, 2),
                    "portfolio_return_pct": round(r.portfolio_return * 100, 2),
                    "max_drawdown_pct": round(r.max_drawdown * 100, 2),
                    "var_95_stressed": round(r.var_95_stressed, 2),
                    "expected_shortfall": round(r.expected_shortfall, 2),
                    "worst_positions": [
                        {"symbol": s, "pnl": round(v, 2)} for s, v in (r.worst_positions or [])[:5]
                    ],
                    "breached_limits": r.breached_limits,
                    "estimated_liquidation_cost": round(r.estimated_liquidation_cost, 2),
                    "recommendations": r.recommendations[:3],
                })
            out.sort(key=lambda x: x["portfolio_pnl"])  # worst first
            return {"scenarios": out, "capital": _capital, "n_positions": len(_positions)}

        return await asyncio.to_thread(_run)
    except Exception as e:
        logger.error(f"Stress scenarios failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/ops/diagnostics")
async def get_diagnostics_report(
    symbol: Optional[str] = None,
    lookback_days: int = 7,
    _user=Depends(require_user),
):
    """Live gate-decision diagnostics — which gates block the most entries, per symbol or aggregate."""
    try:
        from config import ApexConfig
        from monitoring.trade_diagnostics import TradeDiagnosticsTracker
        import asyncio

        tracker = TradeDiagnosticsTracker(data_dir=ApexConfig.DATA_DIR)
        if symbol:
            report = await asyncio.to_thread(tracker.get_symbol_report, symbol, lookback_days)
            gate_attr = await asyncio.to_thread(tracker.get_gate_attribution, symbol, lookback_days)
            return {"symbol_report": report, "gate_attribution": gate_attr}
        else:
            report = await asyncio.to_thread(tracker.get_report, lookback_days)
            return report
    except Exception as e:
        logger.error(f"Diagnostics report failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/ops/wss-metrics")
async def get_wss_metrics(_user=Depends(require_user)):
    """WebSocket connection health metrics — hit rate, reconnects, latency."""
    try:
        from api.dependencies import get_engine
        engine = get_engine()
        if engine is None:
            return {"error": "engine_not_running"}
        # Fix #2: attribute name is 'websocket_streamer' (was incorrectly '_ws_streamer')
        streamer = getattr(engine, 'websocket_streamer', None)
        if streamer is None:
            return {"error": "wss_not_initialised"}
        metrics = streamer.get_metrics()
        # Annotate with health status
        warn_thr = float(getattr(ApexConfig, "WSS_HIT_RATE_WARN_THRESHOLD", 0.50))
        total = metrics.get("wss_hits", 0) + metrics.get("wss_misses", 0)
        hr = metrics.get("hit_rate", 0.0)
        metrics["health"] = "degraded" if (total >= 20 and hr < warn_thr) else "ok"
        metrics["hit_rate_warn_threshold"] = warn_thr
        return metrics
    except Exception as e:
        logger.error("WSS metrics failed: %s", e)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/ops/staleness-report")
async def get_staleness_report(_user=Depends(require_user)):
    """Signal staleness watchdog report — which symbols are overdue for refresh."""
    try:
        from api.dependencies import get_engine
        engine = get_engine()
        if engine is None:
            return {"error": "engine_not_running"}
        watchdog = getattr(engine, '_staleness_watchdog', None)
        if watchdog is None:
            return {"error": "watchdog_not_initialised"}
        return watchdog.get_report()
    except Exception as e:
        logger.error("Staleness report failed: %s", e)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/ops/correlation-heatmap")
async def get_correlation_heatmap(
    lookback_bars: int = 60,
    _user=Depends(require_user),
):
    """Live NxN Pearson correlation matrix for all tracked symbols."""
    try:
        from api.dependencies import get_engine
        engine = get_engine()
        if engine is None:
            return {"symbols": [], "matrix": [], "generated_at": ""}

        sc = getattr(engine, '_signal_cascade', None)
        if sc is None:
            return {"symbols": [], "matrix": [], "generated_at": ""}

        import math
        from datetime import datetime, timezone

        symbols = sorted(
            [sym for sym, hist in sc._prices.items() if len(hist) >= 10]
        )
        n = len(symbols)
        if n == 0:
            return {"symbols": [], "matrix": [], "generated_at": ""}

        def _get_rets(sym):
            prices = list(sc._prices[sym])[-lookback_bars - 1:]
            if len(prices) < 2:
                return []
            return [math.log(prices[i] / prices[i - 1])
                    for i in range(1, len(prices))
                    if prices[i - 1] > 0 and prices[i] > 0]

        def _pearson(xs, ys):
            k = min(len(xs), len(ys))
            if k < 3:
                return None
            xs, ys = xs[-k:], ys[-k:]
            mx = sum(xs) / k
            my = sum(ys) / k
            num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
            dx = math.sqrt(sum((x - mx) ** 2 for x in xs))
            dy = math.sqrt(sum((y - my) ** 2 for y in ys))
            return round(num / (dx * dy), 3) if dx > 1e-12 and dy > 1e-12 else None

        rets_cache = {sym: _get_rets(sym) for sym in symbols}
        matrix = []
        for i, s1 in enumerate(symbols):
            row = []
            for j, s2 in enumerate(symbols):
                if i == j:
                    row.append(1.0)
                elif j < i:
                    row.append(matrix[j][i])
                else:
                    row.append(_pearson(rets_cache[s1], rets_cache[s2]))
            matrix.append(row)

        return {
            "symbols": symbols,
            "matrix": matrix,
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "lookback_bars": lookback_bars,
        }
    except Exception as e:
        logger.error("Correlation heatmap failed: %s", e)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/ops/regime-backtest")
async def get_regime_backtest(
    date: Optional[str] = None,
    _user=Depends(require_user),
):
    """Regime-conditional backtest — per-regime win rate, P&L, Sharpe from trade audit."""
    try:
        import asyncio
        import sys
        from pathlib import Path as _Path
        from config import ApexConfig

        async def _run():
            sys.path.insert(0, str(_Path(__file__).parent.parent / "scripts"))
            from regime_conditional_backtest import (
                _load_exit_trades,
                _group_by_regime,
                _group_by_broker,
                _group_by_exit_reason,
                _load_mandate_stats,
                _compute_regime_stats,
                build_json_summary,
            )
            data_dir = _Path(ApexConfig.DATA_DIR)
            trades = _load_exit_trades(data_dir, date)
            if not trades:
                return {"error": "no_trades", "regime_breakdown": {}, "overall": {}}
            regime_stats = _group_by_regime(trades)
            regime_stats_computed = {r: _compute_regime_stats(ts) for r, ts in regime_stats.items()}
            broker_stats = _group_by_broker(trades)
            broker_stats_computed = {b: _compute_regime_stats(ts) for b, ts in broker_stats.items()}
            exit_buckets = _group_by_exit_reason(trades)
            mandate_stats = _load_mandate_stats(data_dir, date)
            all_stats = _compute_regime_stats(trades)
            return build_json_summary(
                regime_stats_computed,
                broker_stats_computed,
                exit_buckets,
                mandate_stats,
                all_stats,
                date,
            )

        result = await asyncio.to_thread(_run)
        return result
    except Exception as e:
        logger.error("Regime backtest failed: %s", e)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/ops/factor-pnl")
async def get_factor_pnl(
    lookback_days: int = 7,
    _user=Depends(require_user),
):
    """Live factor P&L decomposition — breakdown by ML/technical/sentiment/momentum."""
    try:
        import asyncio
        from config import ApexConfig
        from monitoring.factor_pnl import FactorPnlAnalyzer
        analyzer = FactorPnlAnalyzer(data_dir=ApexConfig.DATA_DIR)
        report = await asyncio.to_thread(analyzer.build_report, lookback_days)
        return report.to_dict()
    except Exception as e:
        logger.error("Factor P&L report failed: %s", e)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/ops/ic-report")
async def get_ic_report(_user=Depends(require_user)):
    """IC Tracker + Alpha Decay Calibrator combined health report."""
    try:
        import asyncio
        from config import ApexConfig
        result: dict = {}
        # IC Tracker summary
        try:
            from monitoring.ic_tracker import ICTracker
            ict = ICTracker(
                state_path=str(ApexConfig.DATA_DIR / "ic_tracker_state.json"),
                persist=False,
            )
            ic_summary = await asyncio.to_thread(ict.get_summary, 10)
            dead   = await asyncio.to_thread(ict.get_dead_features)
            strong = await asyncio.to_thread(ict.get_strong_features)
            result["ic_tracker"] = {
                "summary": {k: round(v, 4) for k, v in ic_summary.items()},
                "dead_features":   sorted(dead),
                "strong_features": sorted(strong),
                "pending_count":   ict.get_pending_count(),
                "observation_counts": ict.get_observation_counts(),
            }
        except Exception as _ict_e:
            result["ic_tracker"] = {"error": str(_ict_e)}
        # Alpha Decay Calibrator report
        try:
            from monitoring.alpha_decay_calibrator import AlphaDecayCalibrator
            adc = AlphaDecayCalibrator(data_dir=ApexConfig.DATA_DIR)
            result["alpha_decay"] = await asyncio.to_thread(adc.get_decay_report)
        except Exception as _adc_e:
            result["alpha_decay"] = {"error": str(_adc_e)}
        return result
    except Exception as e:
        logger.error("IC report failed: %s", e)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/ops/rl-governor")
async def get_rl_governor_endpoint(_user=Depends(require_user)):
    """RL Weight Governor Q-table diagnostic: states, best actions, epsilon, update count."""
    try:
        from models.rl_weight_governor import get_rl_governor_report
        return get_rl_governor_report()
    except Exception as e:
        logger.error("rl-governor endpoint failed: %s", e)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/ops/universe-scores")
async def get_universe_scores(_user=Depends(require_user)):
    """Dynamic universe selector: per-symbol quality scores from recent trade performance."""
    try:
        from api.dependencies import get_engine
        engine = get_engine()
        if engine is None:
            return {"report": None, "note": "engine not running"}
        us = getattr(engine, "_universe_selector", None)
        if us is None:
            return {"report": None, "note": "selector not initialised"}
        return {"report": us.get_report()}
    except Exception as e:
        logger.error("universe-scores endpoint failed: %s", e)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/ops/factor-ic")
async def get_factor_ic(_user=Depends(require_user)):
    """Factor IC report: per-signal Information Coefficient from live trades."""
    try:
        from api.dependencies import get_engine
        engine = get_engine()
        if engine is None:
            from monitoring.factor_ic_tracker import FactorICTracker
            from config import ApexConfig
            tracker = FactorICTracker(
                persist_path=str(getattr(ApexConfig, "FACTOR_IC_PERSIST_PATH",
                                         "data/factor_ic_state.json"))
            )
            return tracker.get_report_dict()
        tracker = getattr(engine, "_factor_ic_tracker", None)
        if tracker is None:
            return {"signals": [], "top_factors": [], "weak_factors": [], "note": "tracker not initialised"}
        return tracker.get_report_dict()
    except Exception as e:
        logger.error("factor-ic endpoint failed: %s", e)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/ops/adaptive-weights")
async def get_adaptive_weights(_user=Depends(require_user)):
    """Adaptive signal blend weights driven by rolling Factor IC."""
    try:
        from api.dependencies import get_engine
        engine = get_engine()
        if engine is None:
            from monitoring.adaptive_weight_manager import AdaptiveWeightManager
            from config import ApexConfig
            mgr = AdaptiveWeightManager(
                persist_path=str(getattr(ApexConfig, "ADAPTIVE_WEIGHTS_PERSIST_PATH",
                                         "data/adaptive_weights.json"))
            )
            return mgr.get_report()
        mgr = getattr(engine, "_adaptive_weights", None)
        if mgr is None:
            return {"weights": {}, "note": "manager not initialised"}
        return mgr.get_report()
    except Exception as e:
        logger.error("adaptive-weights endpoint failed: %s", e)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/ops/backtest-gate")
async def get_backtest_gate(_user=Depends(require_user)):
    """Current backtest gate state: mode (live/paper/unknown), metrics, history."""
    try:
        from monitoring.backtest_gate import BacktestGate
        gate = BacktestGate()
        return gate.get_state()
    except Exception as e:
        logger.error("backtest-gate endpoint failed: %s", e)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/ops/backtest-gate/evaluate")
async def run_backtest_gate(_user=Depends(require_user)):
    """Trigger an on-demand backtest gate evaluation."""
    try:
        from monitoring.backtest_gate import BacktestGate
        from dataclasses import asdict
        gate = BacktestGate()
        record = gate.run_evaluation()
        return asdict(record)
    except Exception as e:
        logger.error("backtest-gate/evaluate failed: %s", e)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/ops/backtest-gate/force-live")
async def force_backtest_live(_user=Depends(require_user)):
    """Admin override: force backtest gate to LIVE mode."""
    try:
        from monitoring.backtest_gate import BacktestGate
        gate = BacktestGate()
        gate.force_live()
        return {"status": "ok", "mode": "live"}
    except Exception as e:
        logger.error("backtest-gate/force-live failed: %s", e)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/ops/daily-briefing")
async def get_daily_briefing(_user=Depends(require_user)):
    """Latest daily strategy briefing (JSON)."""
    try:
        from monitoring.daily_briefing import get_briefing_generator
        gen = get_briefing_generator()
        latest = gen.get_latest()
        if latest is None:
            return {"status": "no_briefing", "message": "No briefing generated yet."}
        return latest
    except Exception as e:
        logger.error("daily-briefing endpoint failed: %s", e)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/ops/daily-briefing/history")
async def get_daily_briefing_history(days: int = 7, _user=Depends(require_user)):
    """Last N daily briefings."""
    try:
        from monitoring.daily_briefing import get_briefing_generator
        gen = get_briefing_generator()
        return {"briefings": gen.get_history(days=days)}
    except Exception as e:
        logger.error("daily-briefing/history endpoint failed: %s", e)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/ops/daily-briefing/generate")
async def generate_daily_briefing(_user=Depends(require_user)):
    """Trigger an on-demand daily briefing generation."""
    try:
        from api.dependencies import get_engine
        from monitoring.daily_briefing import DailyBriefingGenerator
        engine = get_engine()
        regime = "unknown"
        if engine is not None:
            regime = str(getattr(engine, "_current_regime", "unknown"))
        gen = DailyBriefingGenerator()
        briefing = gen.generate(regime=regime, engine=engine)
        return briefing.to_dict()
    except Exception as e:
        logger.error("daily-briefing/generate failed: %s", e)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/ops/short-exposure")
async def get_short_exposure(_user=Depends(require_user)):
    """Current short positions: count, notional, per-symbol breakdown."""
    try:
        from api.dependencies import get_engine
        from risk.short_selling_gate import get_short_exposure_summary
        engine = get_engine()
        positions = {}
        price_cache = {}
        if engine is not None:
            positions = {s: float(q) for s, q in getattr(engine, "positions", {}).items()}
            price_cache = dict(getattr(engine, "price_cache", {}))
        return get_short_exposure_summary(positions, price_cache)
    except Exception as e:
        logger.error("short-exposure endpoint failed: %s", e)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/ops/ofi-summary")
async def get_ofi_summary(_user=Depends(require_user)):
    """Current order flow imbalance (OFI) per tracked symbol."""
    try:
        from data.order_flow_imbalance import get_ofi_signal
        ofi = get_ofi_signal()
        return {
            "ofi_by_symbol": {k: round(v, 4) for k, v in ofi.get_summary().items()},
            "config": {
                "enabled": bool(getattr(__import__("config").ApexConfig, "OFI_ENABLED", True)),
                "window": int(getattr(__import__("config").ApexConfig, "OFI_WINDOW", 20)),
                "blend_weight": float(getattr(__import__("config").ApexConfig, "OFI_BLEND_WEIGHT", 0.06)),
            },
        }
    except Exception as e:
        logger.error("ofi-summary endpoint failed: %s", e)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/ops/bl-weights")
async def get_bl_weights(_user=Depends(require_user)):
    """Black-Litterman portfolio weights and per-symbol sizing multipliers."""
    try:
        from api.dependencies import get_engine
        from risk.black_litterman import BlackLittermanAllocator
        engine = get_engine()
        bl = None
        if engine is not None:
            bl = getattr(engine, "_bl_allocator", None)
        if bl is None:
            bl = BlackLittermanAllocator()
        return bl.get_report()
    except Exception as e:
        logger.error("bl-weights endpoint failed: %s", e)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/ops/earnings-intelligence")
async def get_earnings_intelligence(symbol: str = "AAPL", _user=Depends(require_user)):
    """Extended earnings intelligence: PEAD, revision trend, and persistence signals."""
    try:
        from data.earnings_catalyst import get_earnings_catalyst
        cat = get_earnings_catalyst()
        return {
            "symbol": symbol,
            "pead_signal": cat.get_signal(symbol),
            "revision_signal": cat.get_revision_signal(symbol),
            "persistence_signal": cat.get_persistence_signal(symbol),
            "extended_signal": cat.get_extended_signal(symbol),
        }
    except Exception as e:
        logger.error("earnings-intelligence endpoint failed: %s", e)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/ops/corr-regime")
async def get_corr_regime_report(_user=Depends(require_user)):
    """Correlation matrix regime: average pairwise correlation and sizing multiplier."""
    try:
        from monitoring.correlation_regime_detector import get_corr_regime_detector
        return get_corr_regime_detector().get_report()
    except Exception as e:
        logger.error("corr-regime endpoint failed: %s", e)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/ops/vol-target")
async def get_vol_target_report(_user=Depends(require_user)):
    """Portfolio volatility targeting state: realised vol, target, sizing multiplier."""
    try:
        from risk.portfolio_vol_target import get_vol_target
        return get_vol_target().get_report()
    except Exception as e:
        logger.error("vol-target endpoint failed: %s", e)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/ops/liquidation-risk")
async def get_liquidation_risk(symbol: str = "CRYPTO:BTC/USD", _user=Depends(require_user)):
    """Crypto liquidation cascade risk: funding rate, OI velocity, composite signal."""
    try:
        from monitoring.liquidation_monitor import get_liquidation_monitor
        monitor = get_liquidation_monitor()
        return {
            "symbol_report": monitor.get_report(symbol),
            "all_monitored": monitor.get_all_report(),
        }
    except Exception as e:
        logger.error("liquidation-risk endpoint failed: %s", e)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/ops/iv-skew")
async def get_iv_skew_report(symbol: str = "AAPL", _user=Depends(require_user)):
    """IV skew signal: put/call skew + VIX term structure for a symbol."""
    try:
        from models.iv_skew_signal import get_iv_skew_signal
        gen = get_iv_skew_signal()
        return {
            "symbol_report": gen.get_report(symbol),
            "vix_term_signal": gen.get_vix_term_signal(),
        }
    except Exception as e:
        logger.error("iv-skew endpoint failed: %s", e)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/ops/macro-signal")
async def get_macro_signal_report(_user=Depends(require_user)):
    """Macro cross-asset signal: VIX velocity, yield curve, DXY momentum composite."""
    try:
        from models.macro_cross_asset_signal import get_macro_signal
        return get_macro_signal().get_report()
    except Exception as e:
        logger.error("macro-signal endpoint failed: %s", e)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/ops/gate-stats")
async def get_gate_stats(_user=Depends(require_user)):
    """Return gate rejection summary and infrastructure health indicators."""
    try:
        from api.dependencies import get_engine
        from datetime import datetime, timezone
        engine = get_engine()
        result: dict = {"timestamp": datetime.now(timezone.utc).isoformat()}

        if engine is None:
            result["error"] = "engine not running"
            return result

        # Gate rejection counts (cleared every 300 cycles)
        result["gate_rejection_counts"] = dict(
            sorted(
                getattr(engine, "_gate_rejection_counts", {}).items(),
                key=lambda x: -x[1],
            )
        )

        # Failed symbols
        failed = getattr(engine, "_failed_symbols", set())
        result["failed_symbols_count"] = len(failed)
        result["failed_symbols"] = list(failed)[:20]

        # Historical data coverage
        runtime_syms = engine._runtime_symbols() if hasattr(engine, "_runtime_symbols") else []
        loaded_syms = set(getattr(engine, "historical_data", {}).keys())
        result["data_coverage"] = {
            "runtime_symbols": len(runtime_syms),
            "loaded": len(loaded_syms),
            "pct_loaded": round(len(loaded_syms) / max(len(runtime_syms), 1) * 100, 1),
        }

        # IBKR status
        ibkr = getattr(engine, "ibkr", None)
        result["ibkr_status"] = {
            "connected": ibkr is not None,
            "persistently_down": getattr(ibkr, "_persistently_down", False) if ibkr else True,
        }

        # Stale prices (no live-feed timestamp)
        price_cache = getattr(engine, "price_cache", {})
        price_ts = getattr(engine, "_price_cache_ts", {})
        no_live_ts = [s for s in price_cache if s not in price_ts and price_cache.get(s, 0) > 0]
        result["stale_prices"] = {
            "count": len(no_live_ts),
            "symbols": no_live_ts[:10],
        }

        return result
    except Exception as e:
        logger.error("gate-stats endpoint failed: %s", e)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/ops/strategy-health")
async def get_strategy_health(_user=Depends(require_user)):
    """Rolling strategy health: 30-day Sharpe and paper-only mode status."""
    try:
        from api.dependencies import get_engine
        engine = get_engine()
        if engine is None:
            # Fallback: load directly from persisted state file
            from monitoring.strategy_health_monitor import StrategyHealthMonitor
            from config import ApexConfig
            shm = StrategyHealthMonitor(
                persist_path=str(getattr(ApexConfig, "STRATEGY_HEALTH_PERSIST_PATH",
                                         "data/strategy_health_state.json"))
            )
            return shm.get_state_dict()
        shm = getattr(engine, "_strategy_health", None)
        if shm is None:
            return {"paper_only": False, "note": "monitor not initialised"}
        return shm.get_state_dict()
    except Exception as e:
        logger.error("strategy-health endpoint failed: %s", e)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/ops/trade-postmortem")
async def get_trade_postmortem(_user=Depends(require_user), n: int = 20):
    """Last N trade post-mortems with failure classification and aggregated summary."""
    try:
        from api.dependencies import get_engine
        engine = get_engine()
        if engine is None:
            return {"recent": [], "summary": {}, "note": "engine not running"}
        tpm = getattr(engine, "_trade_postmortem", None)
        if tpm is None:
            return {"recent": [], "summary": {}, "note": "postmortem not initialised"}
        return {
            "recent": tpm.get_recent(min(n, 100)),
            "summary": tpm.get_summary(),
        }
    except Exception as e:
        logger.error("trade-postmortem endpoint failed: %s", e)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/ops/regime-transition")
async def get_regime_transition(_user=Depends(require_user)):
    """Latest regime transition prediction and indicator breakdown."""
    try:
        from api.dependencies import get_engine
        engine = get_engine()
        if engine is None:
            return {"prediction": None, "note": "engine not running"}
        rtp = getattr(engine, "_regime_transition_predictor", None)
        if rtp is None:
            return {"prediction": None, "note": "predictor not initialised"}
        pred = rtp.predict()
        return {"prediction": pred.to_dict()}
    except Exception as e:
        logger.error("regime-transition endpoint failed: %s", e)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/ops/exec-sim-history")
async def get_exec_sim_history(_user=Depends(require_user)):
    """Last 100 pre-trade execution simulator results."""
    try:
        from api.dependencies import get_engine
        engine = get_engine()
        if engine is None:
            return {"records": [], "count": 0, "note": "engine not running"}
        history = list(getattr(engine, "_exec_sim_history", []))
        return {"records": history[-50:], "count": len(history)}
    except Exception as e:
        logger.error("exec-sim-history failed: %s", e)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/ops/signal-drift")
async def get_signal_drift(_user=Depends(require_user)):
    """Signal accuracy drift state — rolling vs baseline win-rate, retrain advisory."""
    try:
        from api.dependencies import get_engine
        engine = get_engine()
        if engine is None:
            return {"state": None, "note": "engine not running"}
        sdm = getattr(engine, "_signal_drift_monitor", None)
        if sdm is None:
            return {"state": None, "note": "drift monitor not initialised"}
        return {"state": sdm.get_state().to_dict()}
    except Exception as e:
        logger.error("signal-drift endpoint failed: %s", e)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/ops/missed-opportunities")
async def get_missed_opportunities(_user=Depends(require_user)):
    """Missed opportunity report: signals filtered before execution + retrospective P&L."""
    try:
        from api.dependencies import get_engine
        engine = get_engine()
        if engine is None:
            return {"pending": [], "report": {}, "note": "engine not running"}
        mot = getattr(engine, "_missed_opportunity_tracker", None)
        if mot is None:
            return {"pending": [], "report": {}, "note": "tracker not initialised"}
        report = mot.generate_report()
        recent_pending = [
            {
                "symbol": o.symbol,
                "signal_strength": round(o.signal_strength, 4),
                "confidence": round(o.confidence, 4),
                "direction": o.direction,
                "regime": o.regime,
                "filter_reason": o.filter_reason,
                "entry_price": round(o.entry_price, 4),
                "asset_class": o.asset_class,
                "signal_date": o.signal_date,
            }
            for o in list(mot._pending)[-50:]
        ]
        from dataclasses import asdict
        return {
            "pending_count": len(mot._pending),
            "completed_count": len(mot._completed),
            "recent_pending": recent_pending,
            "report": asdict(report),
        }
    except Exception as e:
        logger.error("missed-opportunities endpoint failed: %s", e)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/ops/attribution-summary")
async def get_attribution_summary(_user=Depends(require_user), lookback_days: int = 30):
    """Performance attribution by sleeve, asset class, regime, and signal source."""
    try:
        from api.dependencies import get_engine
        engine = get_engine()
        if engine is None:
            return {"summary": {}, "signal_sources": {}, "note": "engine not running"}
        pa = getattr(engine, "performance_attribution", None)
        if pa is None:
            return {"summary": {}, "signal_sources": {}, "note": "attribution not initialised"}
        summary = pa.get_summary(lookback_days=lookback_days)
        signal_sources = pa.get_signal_source_summary(lookback_days=lookback_days)
        return {"summary": summary, "signal_sources": signal_sources}
    except Exception as e:
        logger.error("attribution-summary endpoint failed: %s", e)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/ops/stress-state")
async def get_stress_state(_user=Depends(require_user)):
    """Current intraday stress control state: halt flags, size multiplier, worst scenario."""
    try:
        from api.dependencies import get_engine
        engine = get_engine()
        if engine is None:
            return {"state": None, "note": "engine not running"}
        state = getattr(engine, "_stress_control_state", None)
        if state is None:
            return {"state": None, "note": "stress engine not initialised"}
        return {"state": state.to_dict()}
    except Exception as e:
        logger.error("stress-state endpoint failed: %s", e)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/ops/oos-report")
async def get_oos_report(_user=Depends(require_user)):
    """Out-of-sample validator — per-cell (regime×hour×signal) win rate and Sharpe."""
    try:
        import asyncio
        from monitoring.oos_validator import OOSValidator
        validator = OOSValidator()
        report = await asyncio.to_thread(validator.build_report)
        return report.to_dict()
    except Exception as e:
        logger.error(f"OOS report failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/ops/kill-switch")
async def get_kill_switch_control(_user=Depends(require_user)):
    """Read operational kill-switch control command state."""
    control_state = read_control_state(CONTROL_COMMAND_FILE)
    state = read_trading_state()
    kill_switch = state.get("kill_switch") or {}
    return {
        "command": control_state,
        "runtime": {
            "active": bool(kill_switch.get("active", False)),
            "reason": kill_switch.get("reason"),
            "drawdown": float(kill_switch.get("drawdown", 0.0) or 0.0),
            "sharpe_rolling": float(kill_switch.get("sharpe_rolling", 0.0) or 0.0),
            "flatten_executed": bool(kill_switch.get("flatten_executed", False)),
        },
    }


@app.post("/ops/kill-switch/reset")
async def post_kill_switch_reset(
    payload: KillSwitchResetRequest,
    user=Depends(require_role("admin")),
):
    """Queue an external kill-switch reset command for the trading loop."""
    current = read_control_state(CONTROL_COMMAND_FILE)
    if current.get("kill_switch_reset_requested"):
        raise HTTPException(
            status_code=409,
            detail=f"Kill-switch reset already requested (request_id={current.get('request_id')})",
        )

    requested_by = getattr(user, "username", None) or getattr(user, "user_id", "unknown")
    command = request_kill_switch_reset(
        filepath=CONTROL_COMMAND_FILE,
        requested_by=str(requested_by),
        reason=payload.reason.strip(),
    )
    logger.warning(
        "Kill-switch reset requested by %s (request_id=%s)",
        requested_by,
        command.get("request_id"),
    )
    return {"status": "queued", "command": command}


@app.get("/ops/equity-reconciliation")
async def get_equity_reconciliation_control(_user=Depends(require_user)):
    """Read operational equity reconciliation latch command + runtime state."""
    control_state = read_control_state(CONTROL_COMMAND_FILE)
    state = read_trading_state()
    reconciliation = state.get("equity_reconciliation", {}) or {}
    return {
        "command": {
            "equity_reconciliation_latch_requested": bool(
                control_state.get("equity_reconciliation_latch_requested", False)
            ),
            "equity_reconciliation_latch_request_id": control_state.get(
                "equity_reconciliation_latch_request_id"
            ),
            "equity_reconciliation_latch_requested_at": control_state.get(
                "equity_reconciliation_latch_requested_at"
            ),
            "equity_reconciliation_latch_requested_by": control_state.get(
                "equity_reconciliation_latch_requested_by"
            ),
            "equity_reconciliation_latch_reason": control_state.get(
                "equity_reconciliation_latch_reason"
            ),
            "equity_reconciliation_latch_target_block_entries": control_state.get(
                "equity_reconciliation_latch_target_block_entries"
            ),
            "equity_reconciliation_latch_processed_at": control_state.get(
                "equity_reconciliation_latch_processed_at"
            ),
            "equity_reconciliation_latch_processed_by": control_state.get(
                "equity_reconciliation_latch_processed_by"
            ),
            "equity_reconciliation_latch_processing_note": control_state.get(
                "equity_reconciliation_latch_processing_note"
            ),
        },
        "runtime": {
            "block_entries": bool(reconciliation.get("block_entries", False)),
            "reason": reconciliation.get("reason"),
            "gap_dollars": float(reconciliation.get("gap_dollars", 0.0) or 0.0),
            "gap_pct": float(reconciliation.get("gap_pct", 0.0) or 0.0),
            "breached": bool(reconciliation.get("breached", False)),
            "breach_streak": int(reconciliation.get("breach_streak", 0) or 0),
            "healthy_streak": int(reconciliation.get("healthy_streak", 0) or 0),
            "timestamp": reconciliation.get("timestamp"),
        },
    }


@app.post("/ops/equity-reconciliation/latch")
async def post_equity_reconciliation_latch(
    payload: EquityReconciliationLatchRequest,
    user=Depends(require_role("admin")),
):
    """Queue a manual reconciliation latch command to force block/clear in live loop."""
    current = read_control_state(CONTROL_COMMAND_FILE)
    if current.get("equity_reconciliation_latch_requested"):
        raise HTTPException(
            status_code=409,
            detail=(
                "Equity reconciliation latch command already requested "
                f"(request_id={current.get('equity_reconciliation_latch_request_id')})"
            ),
        )

    requested_by = getattr(user, "username", None) or getattr(user, "user_id", "unknown")
    command = request_equity_reconciliation_latch(
        filepath=CONTROL_COMMAND_FILE,
        requested_by=str(requested_by),
        reason=payload.reason.strip(),
        block_entries=bool(payload.block_entries),
    )
    logger.warning(
        "Equity reconciliation latch requested by %s (request_id=%s, target_block=%s)",
        requested_by,
        command.get("equity_reconciliation_latch_request_id"),
        bool(payload.block_entries),
    )
    return {"status": "queued", "command": command}


# ---------------------------------------------------------------------------
# Broker mode toggle
# ---------------------------------------------------------------------------

class BrokerModeChangeRequest(BaseModel):
    target_mode: str  # "alpaca" | "ibkr" | "both"


def _get_open_equity_positions() -> list:
    """Return symbols of open non-crypto positions from trading_state.json."""
    try:
        import json as _json
        state = _json.loads(STATE_FILE.read_text(encoding="utf-8"))
        positions = state.get("positions", {})
        return [
            sym for sym, p in positions.items()
            if abs(float(p.get("qty", 0) or 0)) > 1e-6
            and not str(sym).startswith("CRYPTO:")
            and "/" not in str(sym)
        ]
    except Exception:
        return []


@app.get("/ops/broker-mode/status")
async def get_broker_mode_status(_user=Depends(require_user)):
    """Return the current active broker routing mode."""
    mode = get_active_broker_mode(CONTROL_COMMAND_FILE)
    return {"broker_mode": mode}


@app.post("/ops/broker-mode/change")
async def post_broker_mode_change(
    payload: BrokerModeChangeRequest,
    user=Depends(require_user),
):
    """Switch broker routing mode at runtime (no restart required).

    Safety guard: rejects the change if the current mode is 'both' (IBKR for
    equities) and the requested mode is 'alpaca', while open IBKR equity
    positions exist.  Closing those positions first prevents misrouted exits.
    """
    valid_modes = ("alpaca", "ibkr", "both")
    if payload.target_mode not in valid_modes:
        raise HTTPException(
            status_code=422,
            detail=f"Invalid target_mode {payload.target_mode!r}. Must be one of: {valid_modes}",
        )

    current_mode = get_active_broker_mode(CONTROL_COMMAND_FILE)
    if current_mode == "both" and payload.target_mode == "alpaca":
        open_equity = _get_open_equity_positions()
        if open_equity:
            shown = ", ".join(open_equity[:5])
            extra = f" (and {len(open_equity) - 5} more)" if len(open_equity) > 5 else ""
            raise HTTPException(
                status_code=409,
                detail=(
                    f"Cannot switch to Alpaca-only: {len(open_equity)} open IBKR equity "
                    f"position(s) found: {shown}{extra}. Close them first."
                ),
            )

    requested_by = getattr(user, "username", None) or getattr(user, "user_id", "api")
    request_broker_mode_change(CONTROL_COMMAND_FILE, str(requested_by), payload.target_mode)
    logger.warning(
        "Broker mode changed to %r by %s (previous: %s)",
        payload.target_mode, requested_by, current_mode,
    )
    return {"status": "ok", "target_mode": payload.target_mode, "previous_mode": current_mode}


@app.get("/ops/governor/policies/active")
async def get_governor_active_policies(_user=Depends(require_role("admin"))):
    repo = _governor_repo()
    policies = sorted(repo.load_active(), key=lambda p: p.key().as_id())
    return {
        "count": len(policies),
        "policies": [
            {
                "policy_id": policy.policy_id(),
                "policy_key": policy.key().as_id(),
                **policy.to_dict(),
            }
            for policy in policies
        ],
    }


@app.get("/ops/governor/policies/candidates")
async def get_governor_candidate_policies(
    status: Optional[str] = None,
    _user=Depends(require_role("admin")),
):
    repo = _governor_repo()
    policies = repo.load_candidates()
    if status:
        status_norm = status.strip().lower()
        allowed = {s.value for s in PromotionStatus}
        if status_norm not in allowed:
            raise HTTPException(status_code=400, detail=f"Invalid status '{status}'")
        policies = [policy for policy in policies if policy.status.value == status_norm]

    policies = sorted(policies, key=lambda p: p.policy_id())
    return {
        "count": len(policies),
        "policies": [
            {
                "policy_id": policy.policy_id(),
                "policy_key": policy.key().as_id(),
                **policy.to_dict(),
            }
            for policy in policies
        ],
    }


@app.post("/ops/governor/policies/approve")
async def post_governor_policy_approve(
    payload: GovernorPolicyApproveRequest,
    user=Depends(require_role("admin")),
):
    repo = _governor_repo()
    service = _governor_service(repo)
    approver = getattr(user, "username", None) or getattr(user, "user_id", "unknown")
    policy_id = payload.policy_id.strip()
    reason = payload.reason.strip()

    decision = service.approve_staged(policy_id=policy_id, approver=str(approver), reason=reason)
    if not decision.accepted:
        raise HTTPException(status_code=400, detail=decision.reason)

    reload_cmd = request_governor_policy_reload(
        filepath=CONTROL_COMMAND_FILE,
        requested_by=str(approver),
        reason=f"manual_approve:{policy_id} {reason}",
    )
    return {
        "status": "approved",
        "decision": {
            "accepted": decision.accepted,
            "manual_approval_required": decision.manual_approval_required,
            "status": decision.status.value,
            "reason": decision.reason,
        },
        "reload_command": reload_cmd,
    }


@app.post("/ops/governor/policies/rollback")
async def post_governor_policy_rollback(
    payload: GovernorPolicyRollbackRequest,
    user=Depends(require_role("admin")),
):
    repo = _governor_repo()
    service = _governor_service(repo)
    approver = getattr(user, "username", None) or getattr(user, "user_id", "unknown")
    asset_class = payload.asset_class.strip().upper()
    regime = payload.regime.strip().lower() or "default"
    reason = payload.reason.strip()
    target_version = payload.target_version.strip() if payload.target_version else None

    decision = service.rollback_active(
        asset_class=asset_class,
        regime=regime,
        approver=str(approver),
        reason=reason,
        target_version=target_version,
    )
    if not decision.accepted:
        raise HTTPException(status_code=400, detail=decision.reason)

    reload_cmd = request_governor_policy_reload(
        filepath=CONTROL_COMMAND_FILE,
        requested_by=str(approver),
        reason=f"rollback:{asset_class}:{regime}:{target_version or 'previous'} {reason}",
    )
    return {
        "status": "rolled_back",
        "decision": {
            "accepted": decision.accepted,
            "manual_approval_required": decision.manual_approval_required,
            "status": decision.status.value,
            "reason": decision.reason,
        },
        "reload_command": reload_cmd,
    }


@app.get("/ops/governor/policies/audit")
async def get_governor_policy_audit(
    limit: int = 100,
    asset_class: Optional[str] = None,
    regime: Optional[str] = None,
    _user=Depends(require_role("admin")),
):
    if regime and not asset_class:
        raise HTTPException(status_code=400, detail="asset_class is required when regime is provided")

    bounded_limit = min(max(int(limit), 1), 2000)
    policy_key: Optional[str] = None
    if asset_class:
        policy_key = f"{asset_class.strip().upper()}:{(regime or 'default').strip().lower() or 'default'}"

    repo = _governor_repo()
    events = repo.load_audit_events(limit=bounded_limit, policy_key=policy_key)
    return {
        "count": len(events),
        "policy_key": policy_key,
        "events": events,
    }


@app.get("/api/v1/social-governor/decisions")
async def get_social_governor_decisions(
    limit: int = 100,
    asset_class: Optional[str] = None,
    regime: Optional[str] = None,
    user=Depends(require_role("admin")),
):
    """Review immutable social-governor decision audits."""
    if regime and not asset_class:
        raise HTTPException(status_code=400, detail="asset_class is required when regime is provided")
    bounded_limit = min(max(int(limit), 1), 2000)
    user_id = str(getattr(user, "user_id", "") or getattr(user, "id", "") or "")
    repo = _social_audit_repo_for_user(user_id)
    events = await asyncio.to_thread(
        repo.load_events,
        limit=bounded_limit,
        asset_class=asset_class,
        regime=regime,
    )
    return {
        "count": len(events),
        "asset_class": asset_class.upper().strip() if asset_class else None,
        "regime": regime.lower().strip() if regime else None,
        "events": events,
    }

# --------------------------------------------------------------------------------
# Session-Scoped Endpoints (Core Strategy + Crypto Sleeve)
# --------------------------------------------------------------------------------

@app.get("/api/v1/sessions")
async def list_sessions(user=Depends(require_user)):
    """List available trading sessions and their status."""
    session_mode = getattr(ApexConfig, "SESSION_MODE", "unified")
    crypto_enabled = getattr(ApexConfig, "CRYPTO_SLEEVE_ENABLED", True)
    sessions = []
    if session_mode == "dual":
        sessions.append({"id": "core", "label": "Core Strategy", "enabled": True,
                         "description": "Equities, indices, and forex"})
        sessions.append({"id": "crypto", "label": "Crypto Sleeve", "enabled": crypto_enabled,
                         "description": "Cryptocurrency trading"})
    elif session_mode == "core_only":
        sessions.append({"id": "core", "label": "Core Strategy", "enabled": True,
                         "description": "Equities, indices, and forex"})
    elif session_mode == "crypto_only":
        sessions.append({"id": "crypto", "label": "Crypto Sleeve", "enabled": True,
                         "description": "Cryptocurrency trading"})
    else:
        sessions.append({"id": "unified", "label": "Unified Strategy", "enabled": True,
                         "description": "All asset classes"})
    return {"session_mode": session_mode, "sessions": sessions}


@app.get("/api/v1/session/{session_type}/status")
async def get_session_status(session_type: str):
    """Get status for a specific trading session (core or crypto)."""
    if session_type not in ("core", "crypto", "unified"):
        raise HTTPException(status_code=400, detail="session_type must be 'core', 'crypto', or 'unified'")

    from api.dependencies import read_session_state
    state = read_session_state(session_type)
    safe_metrics = sanitize_execution_metrics(state)
    session_cfg = ApexConfig.get_session_config(session_type)
    return {
        "session_type": session_type,
        "status": "online" if _state_is_fresh(state, ApexConfig.HEALTH_STALENESS_SECONDS) else "offline",
        "timestamp": state.get("timestamp"),
        "initial_capital": session_cfg.get("initial_capital", 0),
        "symbols_count": len(ApexConfig.get_session_symbols(session_type)),
        **safe_metrics,
    }


@app.get("/api/v1/session/{session_type}/positions")
async def get_session_positions(session_type: str):
    """Get positions for a specific trading session."""
    if session_type not in ("core", "crypto", "unified"):
        raise HTTPException(status_code=400, detail="Invalid session_type")

    from api.dependencies import read_session_state
    state = read_session_state(session_type)
    positions = state.get("positions", {})
    result = []
    for symbol, data in positions.items():
        signal_val = data.get("current_signal", 0)
        signal_dir = str(data.get("signal_direction", "UNKNOWN")).upper()
        if signal_dir == "UNKNOWN":
            if signal_val > 0:
                signal_dir = "LONG"
            elif signal_val < 0:
                signal_dir = "SHORT"

        result.append({
            "symbol": symbol,
            "qty": data.get("qty", 0),
            "side": data.get("side", "LONG"),
            "entry": data.get("avg_price", 0),
            "current": data.get("current_price", 0),
            "pnl": data.get("pnl", 0),
            "pnl_pct": data.get("pnl_pct", 0),
            "signal": signal_val,
            "signal_direction": signal_dir,
        })
    return {"session_type": session_type, "positions": result}


@app.get("/api/v1/session/{session_type}/metrics")
async def get_session_metrics(session_type: str):
    """Get performance metrics for a specific trading session."""
    if session_type not in ("core", "crypto", "unified"):
        raise HTTPException(status_code=400, detail="Invalid session_type")

    from api.dependencies import read_session_state
    state = read_session_state(session_type)
    safe_metrics = sanitize_execution_metrics(state)
    session_cfg = ApexConfig.get_session_config(session_type)

    return {
        "session_type": session_type,
        "sharpe_target": 1.5,
        "initial_capital": session_cfg.get("initial_capital", 0),
        "max_positions": session_cfg.get("max_positions", 0),
        "signal_threshold": session_cfg.get("min_signal_threshold", 0),
        "confidence_threshold": session_cfg.get("min_confidence", 0),
        **safe_metrics,
    }


@app.get("/ops/portfolio-heat")
async def get_portfolio_heat(_user=Depends(require_user)):
    """
    Portfolio heat map: exposure by asset class, regime, correlation cluster,
    and factor risk. Combines FactorHedger + HRPSizer + CorrelationEarlyWarning data.
    """
    try:
        state = read_trading_state()
        positions = state.get("positions", {})
        last_prices = state.get("last_prices", {})
        regime = state.get("regime", "neutral")
        vix = state.get("vix", 20.0)

        # Build per-symbol heat metrics
        symbols = [s for s, q in positions.items() if q and q != 0]
        rows = []
        for sym in symbols:
            qty = positions.get(sym, 0)
            price = last_prices.get(sym) or last_prices.get(sym.replace("CRYPTO:", "")) or 0.0
            notional = abs(float(qty)) * float(price)
            is_crypto = sym.startswith("CRYPTO:") or "/" in sym
            asset_class = "crypto" if is_crypto else "equity"
            rows.append({
                "symbol": sym,
                "qty": qty,
                "notional": round(notional, 2),
                "asset_class": asset_class,
            })

        total_notional = sum(r["notional"] for r in rows) or 1.0
        for r in rows:
            r["weight_pct"] = round(r["notional"] / total_notional * 100, 2)

        # Asset class breakdown
        by_class: dict = {}
        for r in rows:
            ac = r["asset_class"]
            by_class.setdefault(ac, {"count": 0, "notional": 0.0})
            by_class[ac]["count"] += 1
            by_class[ac]["notional"] += r["notional"]
        for ac in by_class:
            by_class[ac]["weight_pct"] = round(by_class[ac]["notional"] / total_notional * 100, 2)

        # HHI concentration (lower = more diversified; 1.0 = single position)
        weights = [r["notional"] / total_notional for r in rows] if rows else []
        hhi = round(sum(w ** 2 for w in weights), 4)

        # Alpha decay calibrator — optimal hold hours per regime
        alpha_decay_hint = None
        try:
            from monitoring.alpha_decay_calibrator import AlphaDecayCalibrator
            _adc_path = getattr(ApexConfig, "DATA_DIR", None)
            if _adc_path:
                _adc = AlphaDecayCalibrator(data_dir=_adc_path)
                alpha_decay_hint = {
                    "optimal_hold_hours": _adc.get_optimal_hold_hours(str(regime)),
                    "alpha_half_life": _adc.get_alpha_half_life(str(regime)),
                }
        except Exception:
            pass

        # Model drift status
        drift_status = None
        try:
            from monitoring.model_drift_monitor import ModelDriftMonitor
            _mdm_path = getattr(ApexConfig, "DATA_DIR", None)
            if _mdm_path:
                _mdm = ModelDriftMonitor(data_dir=_mdm_path)
                _s = _mdm.get_status()
                drift_status = {
                    "health": _s.health,
                    "should_retrain": _s.should_retrain,
                    "ic_current": _s.ic_current,
                    "hit_rate_current": _s.hit_rate_current,
                }
        except Exception:
            pass

        return {
            "positions": rows,
            "by_asset_class": by_class,
            "total_notional": round(total_notional, 2),
            "position_count": len(rows),
            "hhi_concentration": hhi,
            "regime": regime,
            "vix": vix,
            "alpha_decay": alpha_decay_hint,
            "model_drift": drift_status,
        }
    except Exception as e:
        logger.error(f"Portfolio heat endpoint failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/v1/session/crypto/toggle")
async def toggle_crypto_sleeve(user=Depends(require_role("admin"))):
    """Toggle the crypto sleeve on/off."""
    current = getattr(ApexConfig, "CRYPTO_SLEEVE_ENABLED", True)
    ApexConfig.CRYPTO_SLEEVE_ENABLED = not current
    new_state = ApexConfig.CRYPTO_SLEEVE_ENABLED
    logger.info(f"Crypto sleeve toggled: {current} -> {new_state} by user {getattr(user, 'user_id', 'unknown')}")
    return {"crypto_sleeve_enabled": new_state, "previous": current}


@app.get("/ops/equity-curve")
async def get_equity_curve(_user=Depends(require_user), points: int = 200):
    """Recent equity curve (sampled) + drawdown series for charting.

    Works without an in-process engine by falling back to the rolling
    _equity_curve_buffer populated every ~10 s by stream_trading_state().
    """
    try:
        from api.dependencies import get_engine

        def _build_response(curve_raw: list, points: int) -> dict:
            n = len(curve_raw)
            if n == 0:
                return {"curve": [], "drawdown": [], "peak": 0.0, "current": 0.0,
                        "drawdown_pct": 0.0, "total_points": 0}
            step = max(1, n // points)
            sampled = curve_raw[::step]
            # Always include the latest point
            if sampled[-1] != curve_raw[-1]:
                sampled = list(sampled) + [curve_raw[-1]]
            curve_out = [{"t": str(ts), "v": round(float(v), 2)} for ts, v in sampled]
            peak = float(sampled[0][1])
            dd_out = []
            for ts, v in sampled:
                val = float(v)
                peak = max(peak, val)
                dd = ((val - peak) / peak) * 100.0 if peak > 0 else 0.0
                dd_out.append({"t": str(ts), "dd": round(dd, 3)})
            current_val = float(curve_raw[-1][1])
            peak_val = max(float(v) for _, v in curve_raw)
            current_dd = ((current_val - peak_val) / peak_val * 100.0) if peak_val > 0 else 0.0
            return {
                "curve": curve_out,
                "drawdown": dd_out,
                "peak": round(peak_val, 2),
                "current": round(current_val, 2),
                "drawdown_pct": round(current_dd, 3),
                "total_points": n,
            }

        # 1. Try in-process engine performance tracker
        engine = get_engine()
        if engine is not None:
            pt = getattr(engine, "performance_tracker", None)
            if pt is not None:
                curve_raw = list(pt.equity_curve)
                if curve_raw:
                    return _build_response(curve_raw, points)

        # 2. Fall back to the API-side rolling buffer (Docker / separate-process mode)
        with _equity_curve_lock:
            curve_raw = list(_equity_curve_buffer)

        if not curve_raw:
            state = read_trading_state()
            eq = state.get("capital") or state.get("equity")
            ts = state.get("timestamp")
            if eq and eq > 0 and ts:
                # Bootstrap with a single point so the chart isn't blank on first load
                return {
                    "curve": [{"t": str(ts), "v": round(float(eq), 2)}],
                    "drawdown": [{"t": str(ts), "dd": 0.0}],
                    "peak": round(float(eq), 2),
                    "current": round(float(eq), 2),
                    "drawdown_pct": 0.0,
                    "total_points": 1,
                    "note": "Accumulating history — showing current equity snapshot.",
                }
            return {"curve": [], "drawdown": [], "peak": 0.0, "current": 0.0,
                    "drawdown_pct": 0.0, "total_points": 0,
                    "note": "No equity data yet — engine may still be starting."}

        return _build_response(curve_raw, points)
    except Exception as e:
        logger.error("equity-curve endpoint failed: %s", e)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/ops/advanced-metrics")
async def get_advanced_metrics(_user=Depends(require_user)):
    """CVaR, Sortino, Calmar, Omega, VaR, skewness, kurtosis from the live equity curve.

    Works in Docker (engine not in-process) by reading the rolling equity curve buffer
    maintained by stream_trading_state(). Falls back gracefully when history is thin.
    """
    try:
        import numpy as np
        from risk.advanced_metrics import calculate_all_metrics

        def _safe(v):
            if v is None:
                return None
            try:
                f = float(v)
                return None if not (f == f) or abs(f) > 1e9 else round(f, 6)
            except (TypeError, ValueError):
                return None

        # ------------------------------------------------------------------
        # 1. Build equity curve — prefer in-process engine when available,
        #    fall back to the API-side rolling buffer populated every ~10 s.
        # ------------------------------------------------------------------
        from api.dependencies import get_engine
        engine = get_engine()
        curve_raw: list = []
        if engine is not None:
            pt = getattr(engine, "performance_tracker", None)
            if pt is not None:
                curve_raw = list(getattr(pt, "equity_curve", []))

        if not curve_raw:
            with _equity_curve_lock:
                curve_raw = list(_equity_curve_buffer)

        if len(curve_raw) < 5:
            return {
                "note": f"Accumulating equity history — {len(curve_raw)} of 5 minimum points collected. "
                        "Check back in ~1 minute after the engine has been running.",
                "available": False,
                "n_points": len(curve_raw),
            }

        vals = [float(v) for _, v in curve_raw]
        returns_raw = [
            (vals[i] - vals[i - 1]) / vals[i - 1]
            for i in range(1, len(vals))
            if vals[i - 1] > 0
        ]
        if len(returns_raw) < 2:
            return {"note": "insufficient return history", "available": False}

        returns = np.array(returns_raw)
        
        # Avoid non-meaningful metrics when equity curve has zero variance
        if np.std(returns) < 1e-7:
            return {
                "available": False,
                "note": "Awaiting meaningful price action (Stable Equity). Check back once trades reflect PnL variance.",
                "n_points": len(curve_raw)
            }
            
        metrics = await asyncio.to_thread(calculate_all_metrics, returns)

        # ------------------------------------------------------------------
        # 2. Supplemental overlays from engine (no-op when engine is None)
        # ------------------------------------------------------------------
        health = {}
        signal_quality: dict = {}
        model_drift: dict = {}
        outcome_summary: dict = {}
        last_cycle = 0
        if engine is not None:
            if hasattr(engine, "strategy_health_monitor") and engine.strategy_health_monitor:
                try:
                    health = engine.strategy_health_monitor.get_state_dict()
                except Exception:
                    pass
            if hasattr(engine, "signal_outcome_tracker"):
                try:
                    signal_quality = engine.signal_outcome_tracker.get_quality_metrics().__dict__
                    outcome_summary = engine.signal_outcome_tracker.get_summary()
                except Exception:
                    pass
            if getattr(engine, "_model_drift_monitor", None):
                try:
                    model_drift = engine._model_drift_monitor.get_status().__dict__
                except Exception:
                    pass
            last_cycle = getattr(engine, "_cycle_count", 0)

        return {
            "available": True,
            "n_returns": len(returns),
            "source": "engine" if engine is not None else "api_buffer",
            # Scale micro 10-second VaR metrics up to a pseudo-daily level by multiplying by sqrt(8640 approx samples) ~ 93
            # Without this, formatting drops micro losses like -0.0001 per 10s to -0.00%
            "cvar_95": _safe(metrics.get("cvar_95", 0) * 93) if metrics.get("cvar_95") is not None else None,
            "cvar_99": _safe(metrics.get("cvar_99", 0) * 93) if metrics.get("cvar_99") is not None else None,
            "var_95": _safe(metrics.get("var_95", 0) * 93) if metrics.get("var_95") is not None else None,
            "var_99": _safe(metrics.get("var_99", 0) * 93) if metrics.get("var_99") is not None else None,
            "sortino_ratio": _safe(metrics.get("sortino_ratio")),
            "calmar_ratio": _safe(metrics.get("calmar_ratio")),
            "omega_ratio": _safe(metrics.get("omega_ratio")),
            "downside_deviation": _safe(metrics.get("downside_deviation")),
            "tail_ratio": _safe(metrics.get("tail_ratio")),
            "skewness": _safe(metrics.get("skewness")),
            "kurtosis": _safe(metrics.get("kurtosis")),
            "max_dd_duration": _safe(metrics.get("max_dd_duration")),
            "profit_factor": _safe(metrics.get("profit_factor")),
            "expectancy": _safe(metrics.get("expectancy")),
            "health": health,
            "signal_quality": signal_quality,
            "model_drift": model_drift,
            "outcome_summary": outcome_summary,
            "last_cycle": last_cycle,
        }
    except Exception as e:
        logger.error("advanced-metrics endpoint failed: %s", e)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/ops/pitch-metrics")
async def get_pitch_metrics(user=Depends(require_user)):
    """Return the investor-facing pitch metrics derived from live state and the rolling equity curve."""
    current_state = read_trading_state()
    equity_override = None

    try:
        from services.broker.service import broker_service

        tenant_id = str(getattr(user, "user_id", getattr(user, "id", "default")))
        snapshot = await broker_service.get_tenant_equity_snapshot(tenant_id)
        aggregated_equity = _as_finite_float(snapshot.get("total_equity"))
        if aggregated_equity is not None and aggregated_equity > 0:
            equity_override = aggregated_equity
    except Exception as exc:
        logger.debug("Pitch metrics equity snapshot unavailable: %s", exc)

    return _build_pitch_metrics(
        current_state,
        equity_override=equity_override,
        source="ops_endpoint",
    )


# --------------------------------------------------------------------------------
# Signal A/B Gate
# --------------------------------------------------------------------------------

@app.get("/ops/ab-gate")
async def get_ab_gate(_user=Depends(require_user)):
    """Return SignalABGate status: control/challenger variants, promotion history."""
    try:
        from api.dependencies import get_engine
        engine = get_engine()
        if engine is None:
            return {"available": False, "note": "engine not running"}
        gate = getattr(engine, "_signal_ab_gate", None)
        if gate is None:
            return {"available": False, "note": "SignalABGate not initialised"}
        status = gate.get_status()
        return {"available": True, **status}
    except Exception as e:
        logger.error("ab-gate endpoint failed: %s", e)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/ops/ab-gate/register")
async def register_ab_challenger(request: Request, _user=Depends(require_user)):
    """
    Register a new challenger signal weight set for A/B testing.

    Body: {"weights": {"ml": 0.5, "tech": 0.3, "sentiment": 0.2}, "name": "my-variant"}
    """
    try:
        from api.dependencies import get_engine
        engine = get_engine()
        if engine is None:
            raise HTTPException(status_code=503, detail="engine not running")
        gate = getattr(engine, "_signal_ab_gate", None)
        if gate is None:
            raise HTTPException(status_code=503, detail="SignalABGate not initialised")
        body = await request.json()
        weights = body.get("weights")
        name = body.get("name", "challenger")
        if not isinstance(weights, dict) or not weights:
            raise HTTPException(status_code=400, detail="weights dict required")
        gate.register_challenger(weights=weights, name=name)
        logger.info("A/B challenger registered: %s weights=%s", name, weights)
        return {"registered": True, "name": name, "weights": weights}
    except HTTPException:
        raise
    except Exception as e:
        logger.error("ab-gate register failed: %s", e)
        raise HTTPException(status_code=500, detail=str(e))


# --------------------------------------------------------------------------------
# Feature Importance (IC) Drift Dashboard
# --------------------------------------------------------------------------------

@app.get("/ops/feature-ic")
async def get_feature_ic(_user=Depends(require_user)):
    """Return per-feature rolling IC (30d + 90d), status, dead/strong sets."""
    try:
        from api.dependencies import get_engine
        engine = get_engine()
        if engine is None:
            return {"available": False, "note": "engine not running"}
        ict = getattr(engine, "_ic_tracker", None)
        if ict is None:
            return {"available": False, "note": "ICTracker not initialised"}

        obs_counts = ict.get_observation_counts()
        if not obs_counts:
            return {
                "available": True,
                "note": "No observations yet — IC tracking accumulates over time",
                "features": [], "dead": [], "strong": [], "pending": ict.get_pending_count(),
            }

        features = []
        for feat in obs_counts:
            stats = ict.get_stats(feat)
            features.append({
                "feature": feat,
                "ic_30d": round(stats.ic_30d, 4),
                "ic_90d": round(stats.ic_90d, 4),
                "n_obs": stats.n_obs,
                "status": stats.status,
            })
        features.sort(key=lambda x: -abs(x["ic_30d"]))

        return {
            "available": True,
            "n_features": len(features),
            "pending_snapshots": ict.get_pending_count(),
            "dead_count": len(ict.get_dead_features()),
            "strong_count": len(ict.get_strong_features()),
            "dead": sorted(ict.get_dead_features()),
            "strong": sorted(ict.get_strong_features()),
            "features": features,
            "thresholds": {
                "dead": ict.IC_DEAD_THRESHOLD,
                "suspect": ict.IC_SUSPECT_THRESHOLD,
                "strong": ict.IC_STRONG_THRESHOLD,
            },
        }
    except Exception as e:
        logger.error("feature-ic endpoint failed: %s", e)
        raise HTTPException(status_code=500, detail=str(e))


# --------------------------------------------------------------------------------
# Online Learning Pipeline
# --------------------------------------------------------------------------------

@app.get("/ops/online-learning")
async def get_online_learning(_user=Depends(require_user)):
    """Return Online Learning Pipeline state: runs, champion accuracy, promotions."""
    try:
        from api.dependencies import get_engine
        engine = get_engine()
        if engine is None:
            return {"available": False, "note": "engine not running"}
        pipeline = getattr(engine, "_online_learning_pipeline", None)
        if pipeline is None:
            return {"available": False, "note": "OnlineLearningPipeline not initialised"}
        return pipeline.get_state()
    except Exception as e:
        logger.error("online-learning endpoint failed: %s", e)
        raise HTTPException(status_code=500, detail=str(e))


# --------------------------------------------------------------------------------
# Paper Account (Implementation Shortfall Tracker)
# --------------------------------------------------------------------------------

@app.get("/ops/paper-account")
async def get_paper_account(_user=Depends(require_user)):
    """Return shadow paper account snapshot: paper vs live P&L and implementation shortfall."""
    try:
        from api.dependencies import get_engine
        engine = get_engine()
        if engine is None:
            return {"available": False, "note": "engine not running"}
        pa = getattr(engine, "_paper_account", None)
        if pa is None:
            return {"available": False, "note": "PaperAccount not initialised"}
        return pa.get_snapshot()
    except Exception as e:
        logger.error("paper-account endpoint failed: %s", e)
        raise HTTPException(status_code=500, detail=str(e))


# --------------------------------------------------------------------------------
# Alert History Feed
# --------------------------------------------------------------------------------

@app.get("/ops/alerts")
async def get_alert_history(n: int = 50, _user=Depends(require_user)):
    """Return recent alert history from the in-memory alert buffer."""
    try:
        from api.dependencies import get_engine
        engine = get_engine()
        if engine is None:
            return {"available": False, "note": "engine not running", "alerts": []}
        am = getattr(engine, "_alert_manager", None)
        if am is None:
            return {"available": False, "note": "AlertManager not initialised", "alerts": []}
        return {
            "available": True,
            "channel": am.channel,
            "alerts": am.get_recent_alerts(n),
            "total_buffered": len(am._history),
        }
    except Exception as e:
        logger.error("alerts endpoint failed: %s", e)
        raise HTTPException(status_code=500, detail=str(e))


# --------------------------------------------------------------------------------
# HMM Regime Detector state
# --------------------------------------------------------------------------------

@app.get("/ops/hmm-regime")
async def get_hmm_regime(_user=Depends(require_user)):
    """Return current HMM regime state: label, confidence, state probabilities."""
    try:
        from api.dependencies import get_engine
        engine = get_engine()
        if engine is None:
            return {"available": False, "note": "engine not running"}
        det = getattr(engine, "_hmm_regime_detector", None)
        if det is None:
            return {"available": False, "note": "HMMRegimeDetector not initialised"}
        snap = det.get_snapshot()
        snap["current_vix_regime"] = str(getattr(engine, "_current_regime", "unknown"))
        return snap
    except Exception as e:
        logger.error("hmm-regime endpoint failed: %s", e)
        raise HTTPException(status_code=500, detail=str(e))


# --------------------------------------------------------------------------------
# Signal Portfolio Constructor weights
# --------------------------------------------------------------------------------

@app.get("/ops/portfolio-weights")
async def get_portfolio_weights(_user=Depends(require_user)):
    """Return signal-aware HRP target weights computed by SignalPortfolioConstructor."""
    try:
        from api.dependencies import get_engine
        engine = get_engine()
        if engine is None:
            return {"available": False, "note": "engine not running", "weights": {}}
        spc = getattr(engine, "_signal_portfolio_constructor", None)
        if spc is None:
            return {"available": False, "note": "SignalPortfolioConstructor not initialised", "weights": {}}
        return spc.get_portfolio_snapshot()
    except Exception as e:
        logger.error("portfolio-weights endpoint failed: %s", e)
        raise HTTPException(status_code=500, detail=str(e))


# --------------------------------------------------------------------------------
# Model Registry: version tracking + champion/challenger + audit trail
# --------------------------------------------------------------------------------

@app.get("/ops/model-registry")
async def get_model_registry(_user=Depends(require_user)):
    """Return model registry snapshot: champion versions, IC history, rollback events."""
    try:
        from api.dependencies import get_engine
        engine = get_engine()
        if engine is None:
            return {"available": False, "note": "engine not running", "models": {}}
        reg = getattr(engine, "_model_registry", None)
        if reg is None:
            return {"available": False, "note": "ModelRegistry not initialised", "models": {}}
        return reg.get_snapshot()
    except Exception as e:
        logger.error("model-registry endpoint failed: %s", e)
        raise HTTPException(status_code=500, detail=str(e))


# --------------------------------------------------------------------------------
# Cross-Asset Pairs Arbitrage snapshot
# --------------------------------------------------------------------------------

@app.get("/ops/cross-asset-pairs")
async def get_cross_asset_pairs(_user=Depends(require_user)):
    """Return cross-asset pairs arbitrage snapshot: active pairs, z-scores, overlay signals."""
    try:
        from api.dependencies import get_engine
        engine = get_engine()
        if engine is None:
            return {"available": False, "note": "engine not running", "active_pairs": []}
        capa = getattr(engine, "_cross_asset_pairs", None)
        if capa is None:
            return {"available": False, "note": "CrossAssetPairsArb not initialised", "active_pairs": []}
        return capa.get_snapshot()
    except Exception as e:
        logger.error("cross-asset-pairs endpoint failed: %s", e)
        raise HTTPException(status_code=500, detail=str(e))


# --------------------------------------------------------------------------------
# IV Crush Strategy snapshot
# --------------------------------------------------------------------------------

@app.get("/ops/iv-crush")
async def get_iv_crush(_user=Depends(require_user)):
    """Return IV crush strategy snapshot: pre-earnings IV signals and post-earnings PEAD signals."""
    try:
        from api.dependencies import get_engine
        engine = get_engine()
        if engine is None:
            return {"available": False, "note": "engine not running", "active_signals": []}
        ivc = getattr(engine, "_iv_crush_strategy", None)
        if ivc is None:
            return {"available": False, "note": "IVCrushStrategy not initialised", "active_signals": []}
        return ivc.get_snapshot()
    except Exception as e:
        logger.error("iv-crush endpoint failed: %s", e)
        raise HTTPException(status_code=500, detail=str(e))


# --------------------------------------------------------------------------------
# Order Rejections Feed
# --------------------------------------------------------------------------------

@app.get("/ops/order-rejections")
async def get_order_rejections(
    limit: int = 50,
    reason_code: str = "",
    _user=Depends(require_user),
):
    """
    Return the most-recent pre-trade gateway rejections from the audit JSONL files.
    Reads today's + yesterday's files so the feed stays populated across midnight.
    """
    try:
        from api.dependencies import get_engine
        from datetime import timedelta
        engine = get_engine()
        if engine is None:
            return {"available": False, "note": "engine not running", "rejections": []}

        audit_dir = getattr(engine, "user_data_dir", None)
        if audit_dir is None:
            return {"available": False, "note": "audit_dir unavailable", "rejections": []}

        gateway_dir = Path(audit_dir) / "audit" / "pretrade_gateway"
        if not gateway_dir.exists():
            return {"available": True, "rejections": [], "total_scanned": 0}

        today = datetime.utcnow()
        dates = [today.strftime("%Y%m%d"), (today - timedelta(days=1)).strftime("%Y%m%d")]

        records = []
        total_scanned = 0
        for date_str in dates:
            fpath = gateway_dir / f"pretrade_gateway_{date_str}.jsonl"
            if not fpath.exists():
                continue
            try:
                with open(fpath, "r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            rec = json.loads(line)
                        except json.JSONDecodeError:
                            continue
                        total_scanned += 1
                        if rec.get("allowed", True):
                            continue  # only surface rejections
                        if reason_code and rec.get("reason_code") != reason_code:
                            continue
                        records.append({
                            "event_id": rec.get("event_id", ""),
                            "timestamp": rec.get("timestamp", ""),
                            "symbol": rec.get("symbol", ""),
                            "asset_class": rec.get("asset_class", ""),
                            "side": rec.get("side", ""),
                            "quantity": rec.get("quantity", 0),
                            "price": rec.get("price", 0.0),
                            "reason_code": rec.get("reason_code", ""),
                            "message": rec.get("message", ""),
                            "metadata": rec.get("metadata", {}),
                            "actor": rec.get("actor", ""),
                        })
            except Exception as e:
                logger.warning("order-rejections: failed to read %s: %s", fpath, e)

        # Most-recent first
        records.sort(key=lambda r: r["timestamp"], reverse=True)
        records = records[:limit]

        # Aggregate reason_code counts for the summary bar
        from collections import Counter
        reason_counts = Counter(r["reason_code"] for r in records)

        return {
            "available": True,
            "total_scanned": total_scanned,
            "total_rejected": len(records),
            "reason_breakdown": dict(reason_counts),
            "rejections": records,
        }
    except Exception as e:
        logger.error("order-rejections endpoint failed: %s", e)
        raise HTTPException(status_code=500, detail=str(e))


# --------------------------------------------------------------------------------
# WebSocket Endpoint
# --------------------------------------------------------------------------------

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    user = await authenticate_websocket(websocket)
    if not user:
        await websocket.close(code=1008)
        return
    tenant_id = str(getattr(user, "user_id", getattr(user, "id", "default")))
    accepted = await manager.connect(websocket, tenant_id=tenant_id, is_admin=user.has_role("admin"))
    if not accepted:
        return  # Connection rejected (per-user cap hit); manager already closed socket
    if PROMETHEUS_AVAILABLE and WEBSOCKET_CONNECTIONS is not None:
        WEBSOCKET_CONNECTIONS.inc()
    try:
        # Send current state immediately on connect
        current_state = read_trading_state()
        tenant_equity_snapshot: Dict[str, Any] = {}
        aggregated_equity = 0.0
        try:
            from services.broker.service import broker_service
            tenant_equity_snapshot = await broker_service.get_tenant_equity_snapshot(tenant_id)
            aggregated_equity = float(tenant_equity_snapshot.get("total_equity", 0.0) or 0.0)
        except Exception as exc:
            logger.debug("Initial websocket equity snapshot unavailable for %s: %s", tenant_id, exc)

        pitch_metrics = _build_pitch_metrics(
            current_state,
            equity_override=aggregated_equity if aggregated_equity > 0 else None,
            source="ws_stream",
        )
        shadow_terminal = _get_shadow_terminal_payload()
        await websocket.send_json({
            "type": "state_update",
            "tenant_id": tenant_id,
            "timestamp": current_state.get("timestamp", datetime.utcnow().isoformat() + "Z"),
            "capital": aggregated_equity if aggregated_equity > 0 else current_state.get("capital", 0),
            "initial_capital": current_state.get("initial_capital", 0),
            "starting_capital": current_state.get("starting_capital", 0),
            "positions": current_state.get("positions", {}),
            "daily_pnl": current_state.get("daily_pnl", 0),
            "daily_pnl_realized": current_state.get("daily_pnl_realized", current_state.get("daily_pnl", 0)),
            "daily_pnl_source": current_state.get("daily_pnl_source", "equity_delta"),
            "total_pnl": current_state.get("total_pnl", 0),
            "max_drawdown": current_state.get("max_drawdown", 0),
            "sharpe_ratio": current_state.get("sharpe_ratio", 0),
            "win_rate": current_state.get("win_rate", 0),
            "sortino_ratio": current_state.get("sortino_ratio", 0),
            "calmar_ratio": current_state.get("calmar_ratio", 0),
            "profit_factor": current_state.get("profit_factor", 0),
            "alpha_retention": current_state.get("alpha_retention", 0),
            "sector_exposure": current_state.get("sector_exposure", {}),
            "open_positions": current_state.get("open_positions", 0),
            "max_positions": current_state.get("max_positions", ApexConfig.MAX_POSITIONS),
            "total_trades": current_state.get("total_trades", 0),
            "active_margin": current_state.get("active_margin", 0),
            "leverage_limit": current_state.get("leverage_limit", 1.0),
            "broker_heartbeats": current_state.get("broker_heartbeats", {}),
            "broker_mode": current_state.get("broker_mode", "both"),
            "aggregated_equity": aggregated_equity if aggregated_equity > 0 else current_state.get("capital", 0),
            "total_equity": aggregated_equity if aggregated_equity > 0 else current_state.get("capital", 0),
            "equity_breakdown": tenant_equity_snapshot.get("breakdown", []),
            "alerts": current_state.get("alerts", []),
            "pitch_metrics": pitch_metrics,
            "shadow_terminal": shadow_terminal,
        })
        while True:
            try:
                data = await asyncio.wait_for(websocket.receive_text(), timeout=45.0)
                if PROMETHEUS_AVAILABLE and WEBSOCKET_MESSAGES_TOTAL is not None:
                    WEBSOCKET_MESSAGES_TOTAL.labels(direction="inbound").inc()
                logger.info(f"Received command: {data}")
            except asyncio.TimeoutError:
                # No message in 45 s — send a ping to check if client is still alive.
                try:
                    await websocket.send_json({"type": "ping"})
                except Exception:
                    raise WebSocketDisconnect(code=1001)
    except WebSocketDisconnect:
        await manager.disconnect(websocket)
        if PROMETHEUS_AVAILABLE and WEBSOCKET_CONNECTIONS is not None:
            WEBSOCKET_CONNECTIONS.dec()
