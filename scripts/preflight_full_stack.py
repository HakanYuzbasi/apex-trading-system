#!/usr/bin/env python3
"""
Full-stack operational preflight for APEX core + USP endpoints.

Checks:
- Backend auth, state, status
- Mandate copilot (evaluate/calibration/audit/workflows/report)
- Frontend API proxies (optional)
"""

from __future__ import annotations

import argparse
import json
import sys
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
ENV_PATH = ROOT / ".env"
STATUS_FILE = ROOT / "data" / "preflight_status.json"
HISTORY_FILE = ROOT / "data" / "preflight_history.jsonl"


@dataclass
class CheckResult:
    name: str
    ok: bool
    status: int | None
    detail: str


def _load_env(path: Path) -> dict[str, str]:
    env: dict[str, str] = {}
    if not path.exists():
        return env
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        k, v = stripped.split("=", 1)
        env[k.strip()] = v.strip()
    return env


def _json_request(
    method: str,
    url: str,
    headers: dict[str, str] | None = None,
    body: dict[str, Any] | None = None,
) -> tuple[int, Any]:
    payload = None
    req_headers = dict(headers or {})
    if body is not None:
        payload = json.dumps(body).encode("utf-8")
        req_headers.setdefault("Content-Type", "application/json")
    req = urllib.request.Request(url=url, data=payload, method=method, headers=req_headers)
    try:
        with urllib.request.urlopen(req, timeout=8) as resp:
            status = int(resp.getcode())
            raw = resp.read().decode("utf-8", errors="replace")
            try:
                parsed = json.loads(raw) if raw else {}
            except json.JSONDecodeError:
                parsed = raw
            return status, parsed
    except urllib.error.HTTPError as exc:
        raw = exc.read().decode("utf-8", errors="replace")
        try:
            parsed = json.loads(raw) if raw else {}
        except json.JSONDecodeError:
            parsed = raw
        return int(exc.code), parsed


def _add_result(results: list[CheckResult], name: str, status: int | None, ok: bool, detail: str) -> None:
    results.append(CheckResult(name=name, ok=ok, status=status, detail=detail))


def _persist_results(results: list[CheckResult], exit_code: int) -> None:
    STATUS_FILE.parent.mkdir(parents=True, exist_ok=True)
    run_at = datetime.now(timezone.utc).isoformat()
    total_checks = len(results)
    passed_checks = sum(1 for row in results if row.ok)
    pass_rate = (passed_checks / total_checks) if total_checks else 0.0
    payload = {
        "run_at": run_at,
        "exit_code": int(exit_code),
        "total_checks": total_checks,
        "passed_checks": passed_checks,
        "pass_rate": round(pass_rate, 6),
        "results": [
            {
                "name": row.name,
                "ok": row.ok,
                "status": row.status,
                "detail": row.detail,
            }
            for row in results
        ],
    }
    STATUS_FILE.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    with HISTORY_FILE.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(payload, separators=(",", ":"), ensure_ascii=True) + "\n")


def run_preflight(
    backend_base: str,
    frontend_base: str,
    username: str,
    password: str,
    include_frontend: bool,
    exercise_workflow: bool,
    ) -> int:
    results: list[CheckResult] = []
    backend = backend_base.rstrip("/")
    frontend = frontend_base.rstrip("/")

    # 1) Backend login
    try:
        status, body = _json_request(
            "POST",
            f"{backend}/auth/login",
            body={"username": username, "password": password},
        )
    except Exception as exc:
        _add_result(results, "backend.auth.login", None, False, f"transport error: {exc}")
        _print_summary(results)
        _persist_results(results, exit_code=2)
        return 2

    token = body.get("access_token") if isinstance(body, dict) else None
    if status == 200 and isinstance(token, str) and token:
        _add_result(results, "backend.auth.login", status, True, "ok")
    else:
        _add_result(results, "backend.auth.login", status, False, str(body)[:220])
        _print_summary(results)
        _persist_results(results, exit_code=2)
        return 2

    auth_headers = {"Authorization": f"Bearer {token}"}

    # 2) Core backend checks
    for name, path in [
        ("backend.auth.me", "/auth/me"),
        ("backend.status", "/status"),
        ("backend.state", "/state"),
    ]:
        try:
            status, body = _json_request("GET", f"{backend}{path}", headers=auth_headers)
            _add_result(results, name, status, status == 200, "ok" if status == 200 else str(body)[:220])
        except Exception as exc:
            _add_result(results, name, None, False, f"transport error: {exc}")

    # 3) USP: Mandate copilot checks
    request_id = ""
    try:
        status, body = _json_request(
            "POST",
            f"{backend}/api/v1/mandate-copilot/evaluate",
            headers=auth_headers,
            body={
                "intent": "Target 10% in 60 days for energy and technology sleeves.",
                "recommendation_mode": "POLICY_ONLY",
                "suitability_profile": "balanced",
            },
        )
        request_id = body.get("request_id", "") if isinstance(body, dict) else ""
        _add_result(
            results,
            "usp.mandate.evaluate",
            status,
            status == 200 and bool(request_id),
            "ok" if status == 200 else str(body)[:220],
        )
    except Exception as exc:
        _add_result(results, "usp.mandate.evaluate", None, False, f"transport error: {exc}")

    for name, path in [
        ("usp.mandate.calibration", "/api/v1/mandate-copilot/calibration?limit=100"),
        ("usp.mandate.audit", "/api/v1/mandate-copilot/audit?limit=5"),
        ("usp.mandate.workflows", "/api/v1/mandate-copilot/workflows?limit=5"),
        ("usp.mandate.report", "/api/v1/mandate-copilot/reports/monthly?lookback=300"),
    ]:
        try:
            status, body = _json_request("GET", f"{backend}{path}", headers=auth_headers)
            _add_result(results, name, status, status == 200, "ok" if status == 200 else str(body)[:220])
        except Exception as exc:
            _add_result(results, name, None, False, f"transport error: {exc}")

    if exercise_workflow and request_id:
        try:
            status, body = _json_request(
                "POST",
                f"{backend}/api/v1/mandate-copilot/workflows/initiate",
                headers=auth_headers,
                body={"request_id": request_id, "note": "preflight workflow initiation check"},
            )
            _add_result(
                results,
                "usp.mandate.workflow.initiate",
                status,
                status == 200,
                "ok" if status == 200 else str(body)[:220],
            )
        except Exception as exc:
            _add_result(results, "usp.mandate.workflow.initiate", None, False, f"transport error: {exc}")

    # 4) Frontend proxy checks
    if include_frontend:
        cookie = {"Cookie": f"token={token}"}
        for name, path in [
            ("frontend.api.metrics", "/api/v1/metrics"),
            ("frontend.api.cockpit", "/api/v1/cockpit"),
            ("frontend.api.mandate.calibration", "/api/v1/mandate/calibration?limit=50"),
            ("frontend.api.mandate.report", "/api/v1/mandate/reports/monthly?lookback=200"),
        ]:
            try:
                status, body = _json_request("GET", f"{frontend}{path}", headers=cookie)
                _add_result(results, name, status, status == 200, "ok" if status == 200 else str(body)[:220])
            except Exception as exc:
                _add_result(results, name, None, False, f"transport error: {exc}")

    _print_summary(results)
    exit_code = 0 if all(r.ok for r in results) else 1
    _persist_results(results, exit_code=exit_code)
    return exit_code


def _print_summary(results: list[CheckResult]) -> None:
    print("\n=== APEX Full-Stack Preflight ===")
    ok_count = sum(1 for r in results if r.ok)
    for row in results:
        status_str = str(row.status) if row.status is not None else "NA"
        marker = "PASS" if row.ok else "FAIL"
        print(f"[{marker}] {row.name:<36} status={status_str:<3} detail={row.detail}")
    print(f"Summary: {ok_count}/{len(results)} checks passed")


def main() -> int:
    parser = argparse.ArgumentParser(description="Run APEX full-stack functional preflight.")
    parser.add_argument("--backend-base", default="http://127.0.0.1:8000")
    parser.add_argument("--frontend-base", default="http://127.0.0.1:3000")
    parser.add_argument("--username", default="admin")
    parser.add_argument("--password", default="")
    parser.add_argument("--skip-frontend", action="store_true", help="Skip frontend proxy checks.")
    parser.add_argument(
        "--exercise-workflow",
        action="store_true",
        help="Also call workflow initiation endpoint (writes a workflow record).",
    )
    args = parser.parse_args()

    env = _load_env(ENV_PATH)
    password = args.password or env.get("APEX_ADMIN_PASSWORD", "")
    if not password:
        print("ERROR: admin password not provided and APEX_ADMIN_PASSWORD missing from .env", file=sys.stderr)
        return 2

    return run_preflight(
        backend_base=args.backend_base,
        frontend_base=args.frontend_base,
        username=args.username,
        password=password,
        include_frontend=not args.skip_frontend,
        exercise_workflow=args.exercise_workflow,
    )


if __name__ == "__main__":
    raise SystemExit(main())
