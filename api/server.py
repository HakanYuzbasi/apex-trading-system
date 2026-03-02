"""
api/server.py - APEX Trading API Server

Core backend for the SOTA UI.
Exposes REST endpoints and WebSockets for real-time data streaming.
Reads state from trading_state.json written by the ApexTrader.
"""

import asyncio
import logging
import os
import random
import threading
import time
from pathlib import Path
from typing import Dict, List, Optional
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

_preflight_metrics_lock = threading.Lock()
_preflight_metrics_mtime_ns: Optional[int] = None


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

            # Periodically fetch aggregated equity (every ~15 seconds to avoid API limits)
            now_ts = time.time()
            if now_ts - last_equity_update_time > 15:
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
                        })
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

        await asyncio.sleep(getattr(ApexConfig, "POLL_INTERVAL_SECONDS", 1.0))


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
        if isinstance(total_equity, (int, float)):
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


@app.get("/ops/kill-switch")
async def get_kill_switch_control(_user=Depends(require_user)):
    """Read operational kill-switch control command state."""
    control_state = read_control_state(CONTROL_COMMAND_FILE)
    state = read_trading_state()
    kill_switch = state.get("kill_switch", {})
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
    events = repo.load_events(
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
        from services.broker.service import broker_service
        tenant_equity_snapshot = await broker_service.get_tenant_equity_snapshot(tenant_id)
        aggregated_equity = float(tenant_equity_snapshot.get("total_equity", 0.0) or 0.0)
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
            "aggregated_equity": aggregated_equity,
            "total_equity": aggregated_equity,
            "equity_breakdown": tenant_equity_snapshot.get("breakdown", []),
        })
        while True:
            data = await websocket.receive_text()
            if PROMETHEUS_AVAILABLE and WEBSOCKET_MESSAGES_TOTAL is not None:
                WEBSOCKET_MESSAGES_TOTAL.labels(direction="inbound").inc()
            logger.info(f"Received command: {data}")
    except WebSocketDisconnect:
        await manager.disconnect(websocket)
        if PROMETHEUS_AVAILABLE and WEBSOCKET_CONNECTIONS is not None:
            WEBSOCKET_CONNECTIONS.dec()
