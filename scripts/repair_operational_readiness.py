#!/usr/bin/env python3
"""
Repair local operational readiness state for paper trading:
- Persist security secrets in .env (admin API key, admin password, JWT secret, metrics token)
- Align admin credentials in data/users.json
- Rebase stale risk/performance/trading state with timestamped backups
- Enqueue unified latch reset + reconciliation latch clear commands
- Verify local IBKR API connectivity (port 7497 by default)
"""

from __future__ import annotations

import argparse
import json
import secrets
import shutil
import socket
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, Tuple

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))
DATA_DIR = ROOT_DIR / "data"
ENV_FILE = ROOT_DIR / ".env"
USERS_FILE = DATA_DIR / "users.json"
RISK_STATE_FILE = DATA_DIR / "risk_state.json"
PERFORMANCE_FILE = DATA_DIR / "performance_history.json"
TRADING_STATE_FILE = DATA_DIR / "trading_state.json"
CONTROL_FILE = DATA_DIR / "trading_control_commands.json"


def _load_env(path: Path) -> Dict[str, str]:
    env: Dict[str, str] = {}
    if not path.exists():
        return env
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        key, value = stripped.split("=", 1)
        parsed_value = value.strip()
        if " #" in parsed_value:
            parsed_value = parsed_value.split(" #", 1)[0].strip()
        env[key.strip()] = parsed_value
    return env


def _upsert_env(path: Path, updates: Dict[str, str]) -> None:
    lines = path.read_text(encoding="utf-8").splitlines() if path.exists() else []
    index_by_key: Dict[str, int] = {}
    for i, line in enumerate(lines):
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        key = stripped.split("=", 1)[0].strip()
        if key:
            index_by_key[key] = i

    for key, value in updates.items():
        rendered = f"{key}={value}"
        if key in index_by_key:
            lines[index_by_key[key]] = rendered
        else:
            lines.append(rendered)

    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def _backup_file(path: Path) -> Path | None:
    if not path.exists():
        return None
    timestamp = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    backup = path.with_suffix(path.suffix + f".bak.{timestamp}")
    shutil.copy2(path, backup)
    return backup


def _read_json(path: Path, default):
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return default


def _write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _determine_baseline_capital(trading_state: Dict, env: Dict[str, str]) -> float:
    candidates = []
    candidates.append(trading_state.get("capital"))
    candidates.append(trading_state.get("starting_capital"))
    candidates.append(env.get("APEX_INITIAL_CAPITAL"))
    candidates.append(env.get("INITIAL_CAPITAL"))
    for value in candidates:
        try:
            parsed = float(value)
            if parsed > 0:
                return round(parsed, 2)
        except Exception:
            continue
    return 100_000.00


def _ensure_admin_user(users_payload: Dict, admin_api_key: str, admin_password_hash: str) -> None:
    users = users_payload.setdefault("users", [])
    admin_entry = None
    for user in users:
        if not isinstance(user, dict):
            continue
        if user.get("user_id") == "admin" or user.get("username") == "admin":
            admin_entry = user
            break

    if admin_entry is None:
        admin_entry = {
            "user_id": "admin",
            "username": "admin",
            "email": "admin@apex.local",
            "roles": ["admin", "user"],
            "permissions": ["read", "write", "trade", "admin"],
            "created_at": datetime.utcnow().isoformat(),
            "tier": "enterprise",
        }
        users.insert(0, admin_entry)

    admin_entry["user_id"] = "admin"
    admin_entry["username"] = "admin"
    admin_entry["email"] = admin_entry.get("email") or "admin@apex.local"
    admin_entry["roles"] = sorted(set((admin_entry.get("roles") or []) + ["admin", "user"]))
    admin_entry["permissions"] = sorted(
        set((admin_entry.get("permissions") or []) + ["admin", "read", "write", "trade"])
    )
    admin_entry["api_key"] = admin_api_key
    admin_entry["password_hash"] = admin_password_hash


def _connectivity_check(host: str, port: int, timeout: float = 2.0) -> Tuple[bool, str]:
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True, "ok"
    except Exception as exc:
        return False, str(exc)


def main() -> int:
    parser = argparse.ArgumentParser(description="Repair APEX paper operational readiness state.")
    parser.add_argument("--ibkr-host", default="127.0.0.1")
    parser.add_argument("--ibkr-port", type=int, default=7497)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    env = _load_env(ENV_FILE)

    # Canonical env keys (keep backward-compatible legacy keys as-is)
    admin_api_key = env.get("APEX_ADMIN_API_KEY") or f"apex-{secrets.token_hex(16)}"
    admin_password = env.get("APEX_ADMIN_PASSWORD") or secrets.token_urlsafe(18)
    jwt_secret = env.get("APEX_SECRET_KEY") or secrets.token_hex(32)
    metrics_token = env.get("APEX_METRICS_TOKEN") or secrets.token_urlsafe(24)

    env_updates = {
        "APEX_ADMIN_API_KEY": admin_api_key,
        "APEX_ADMIN_PASSWORD": admin_password,
        "APEX_SECRET_KEY": jwt_secret,
        "APEX_METRICS_TOKEN": metrics_token,
        "APEX_AUTH_ENABLED": env.get("APEX_AUTH_ENABLED", "true"),
        "APEX_BROKER_MODE": env.get("APEX_BROKER_MODE", env.get("BROKER_MODE", "both")),
        "APEX_LIVE_TRADING": env.get("APEX_LIVE_TRADING", env.get("LIVE_TRADING", "true")).lower(),
        "APEX_IBKR_HOST": env.get("APEX_IBKR_HOST", env.get("IBKR_HOST", args.ibkr_host)),
        "APEX_IBKR_PORT": env.get("APEX_IBKR_PORT", env.get("IBKR_PORT", str(args.ibkr_port))),
        "APEX_INITIAL_CAPITAL": env.get(
            "APEX_INITIAL_CAPITAL", env.get("INITIAL_CAPITAL", "100000")
        ),
        "APEX_PAPER_STARTUP_RISK_SELF_HEAL_ENABLED": env.get(
            "APEX_PAPER_STARTUP_RISK_SELF_HEAL_ENABLED", "true"
        ),
        "APEX_PAPER_STARTUP_RESET_CIRCUIT_BREAKER": env.get(
            "APEX_PAPER_STARTUP_RESET_CIRCUIT_BREAKER", "true"
        ),
        "APEX_PAPER_STARTUP_PERFORMANCE_REBASE_ENABLED": env.get(
            "APEX_PAPER_STARTUP_PERFORMANCE_REBASE_ENABLED", "true"
        ),
        "APEX_UNIFIED_LATCH_RESET_REBASE_RISK_BASELINES": env.get(
            "APEX_UNIFIED_LATCH_RESET_REBASE_RISK_BASELINES", "true"
        ),
        "APEX_UNIFIED_LATCH_RESET_REBASE_PERFORMANCE": env.get(
            "APEX_UNIFIED_LATCH_RESET_REBASE_PERFORMANCE", "true"
        ),
    }

    trading_state = _read_json(TRADING_STATE_FILE, {})
    baseline_capital = _determine_baseline_capital(trading_state, env_updates)
    now_iso = datetime.utcnow().isoformat()
    today = datetime.utcnow().strftime("%Y-%m-%d")

    # Prepare payload updates
    users_payload = _read_json(USERS_FILE, {"users": []})
    from services.auth.service import hash_password  # Uses installed bcrypt fallback chain

    _ensure_admin_user(
        users_payload=users_payload,
        admin_api_key=admin_api_key,
        admin_password_hash=hash_password(admin_password),
    )

    risk_state = _read_json(RISK_STATE_FILE, {})
    risk_state.update(
        {
            "starting_capital": baseline_capital,
            "peak_capital": baseline_capital,
            "day_start_capital": baseline_capital,
            "current_day": today,
            "circuit_breaker": {
                "is_tripped": False,
                "reason": None,
                "trip_time": None,
                "consecutive_losses": 0,
                "recent_trades": 0,
            },
            "updated_at": now_iso,
        }
    )

    perf_payload = _read_json(PERFORMANCE_FILE, {})
    perf_payload.update(
        {
            "starting_capital": baseline_capital,
            "trades": [],
            "equity_curve": [[now_iso, baseline_capital]],
            "updated_at": now_iso,
        }
    )

    if not isinstance(trading_state, dict):
        trading_state = {}
    trading_state.update(
        {
            "timestamp": now_iso,
            "capital": baseline_capital,
            "initial_capital": baseline_capital,
            "starting_capital": baseline_capital,
            "daily_pnl": 0.0,
            "total_pnl": 0.0,
            "max_drawdown": 0.0,
            "sharpe_ratio": 0.0,
            "sortino_ratio": 0.0,
            "total_trades": 0,
            "equity_reconciliation": {
                "block_entries": False,
                "reason": "manual_operational_repair",
            },
        }
    )

    if args.dry_run:
        print("DRY RUN: no files modified")
    else:
        _upsert_env(ENV_FILE, env_updates)
        for path in (USERS_FILE, RISK_STATE_FILE, PERFORMANCE_FILE, TRADING_STATE_FILE):
            _backup_file(path)
        _write_json(USERS_FILE, users_payload)
        _write_json(RISK_STATE_FILE, risk_state)
        _write_json(PERFORMANCE_FILE, perf_payload)
        _write_json(TRADING_STATE_FILE, trading_state)

        from core.trading_control import (
            request_equity_reconciliation_latch,
            request_kill_switch_reset,
        )

        request_kill_switch_reset(
            filepath=CONTROL_FILE,
            requested_by="ops-repair-script",
            reason="operational_readiness_repair",
        )
        request_equity_reconciliation_latch(
            filepath=CONTROL_FILE,
            requested_by="ops-repair-script",
            reason="clear_reconciliation_latch_after_state_repair",
            block_entries=False,
        )

    host = env_updates["APEX_IBKR_HOST"]
    port = int(env_updates["APEX_IBKR_PORT"])
    ok, detail = _connectivity_check(host, port)

    print("=== APEX Operational Repair ===")
    print(f"baseline_capital={baseline_capital:.2f}")
    print(f"ibkr_connectivity host={host} port={port} status={'OK' if ok else 'FAIL'} detail={detail}")
    print("admin_username=admin")
    print(f"admin_api_key={admin_api_key}")
    print(f"admin_password={admin_password}")
    print(f"jwt_secret_set={'yes' if bool(jwt_secret) else 'no'}")
    print(f"metrics_token_set={'yes' if bool(metrics_token) else 'no'}")
    print(f"control_file={CONTROL_FILE}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
