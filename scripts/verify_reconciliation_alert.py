#!/usr/bin/env python3
"""
Reload Grafana provisioning, verify reconciliation rule presence,
force reconciliation latch, and confirm alert firing.
"""

from __future__ import annotations

import argparse
import base64
import json
import sys
import time
from pathlib import Path
from typing import Any
from urllib import error, request

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from config import ApexConfig
from core.trading_control import request_equity_reconciliation_latch


def _local_file_check(rule_uid: str) -> None:
    alert_file = ROOT_DIR / "monitoring" / "grafana" / "provisioning" / "alerting" / "execution_gate_alerts.yml"
    dashboard_file = ROOT_DIR / "monitoring" / "grafana" / "dashboards" / "apex_governor_risk_overview.json"
    if not alert_file.exists():
        raise RuntimeError(f"Alert provisioning file not found: {alert_file}")
    if not dashboard_file.exists():
        raise RuntimeError(f"Dashboard file not found: {dashboard_file}")

    alert_text = alert_file.read_text(encoding="utf-8")
    if f"uid: {rule_uid}" not in alert_text:
        raise RuntimeError(f"Rule UID {rule_uid} not found in {alert_file}")

    dashboard = json.loads(dashboard_file.read_text(encoding="utf-8"))
    panels = dashboard.get("panels", []) if isinstance(dashboard, dict) else []
    alert_list_filter = None
    for panel in panels:
        if not isinstance(panel, dict):
            continue
        if panel.get("type") == "alertlist":
            options = panel.get("options", {}) or {}
            alert_list_filter = options.get("alertInstanceLabelFilter")
            break
    if not alert_list_filter or rule_uid not in str(alert_list_filter):
        raise RuntimeError(
            f"Rule UID {rule_uid} missing from dashboard alert list filter in {dashboard_file}"
        )


def _auth_header(username: str, password: str) -> str:
    raw = f"{username}:{password}".encode("utf-8")
    return "Basic " + base64.b64encode(raw).decode("ascii")


def _http_json(
    method: str,
    url: str,
    *,
    username: str,
    password: str,
    payload: dict[str, Any] | None = None,
) -> tuple[int, Any]:
    headers = {
        "Accept": "application/json",
        "Authorization": _auth_header(username, password),
    }
    data = None
    if payload is not None:
        headers["Content-Type"] = "application/json"
        data = json.dumps(payload).encode("utf-8")

    req = request.Request(url=url, method=method.upper(), headers=headers, data=data)
    try:
        with request.urlopen(req, timeout=15) as resp:
            body = resp.read().decode("utf-8")
            if not body:
                return int(resp.status), {}
            try:
                return int(resp.status), json.loads(body)
            except json.JSONDecodeError:
                return int(resp.status), {"raw": body}
    except error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"{method} {url} failed ({exc.code}): {body}") from exc
    except error.URLError as exc:
        raise RuntimeError(f"{method} {url} failed: {exc}") from exc


def _rule_exists(rules_payload: Any, rule_uid: str) -> bool:
    if isinstance(rules_payload, dict):
        if isinstance(rules_payload.get("items"), list):
            rules = rules_payload["items"]
        else:
            rules = [rules_payload]
    elif isinstance(rules_payload, list):
        rules = rules_payload
    else:
        return False

    for rule in rules:
        if not isinstance(rule, dict):
            continue
        if str(rule.get("uid") or "") == rule_uid:
            return True
    return False


def _is_rule_firing(alerts_payload: Any, rule_uid: str) -> bool:
    if not isinstance(alerts_payload, list):
        return False
    for alert in alerts_payload:
        if not isinstance(alert, dict):
            continue
        labels = alert.get("labels", {}) or {}
        if str(labels.get("grafana_rule_uid") or "") == rule_uid:
            return True
    return False


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(
        description="Reload Grafana provisioning and verify reconciliation alert firing."
    )
    parser.add_argument("--grafana-url", default="http://localhost:3002")
    parser.add_argument("--grafana-user", default="admin")
    parser.add_argument("--grafana-password", default="admin")
    parser.add_argument("--rule-uid", default="apex_equity_reconciliation_gap_blocked")
    parser.add_argument(
        "--control-file",
        default=str(ApexConfig.DATA_DIR / "trading_control_commands.json"),
    )
    parser.add_argument("--requested-by", default="ops-verifier")
    parser.add_argument("--timeout-seconds", type=int, default=180)
    parser.add_argument("--poll-seconds", type=float, default=5.0)
    parser.add_argument("--no-force-latch", action="store_true")
    parser.add_argument("--no-auto-clear", action="store_true")
    parser.add_argument(
        "--local-only",
        action="store_true",
        help="Validate local Grafana files only (no HTTP calls).",
    )
    args = parser.parse_args(argv)

    grafana_base = args.grafana_url.rstrip("/")
    username = args.grafana_user
    password = args.grafana_password
    rule_uid = args.rule_uid
    control_file = Path(args.control_file)

    if args.local_only:
        print("Checking local Grafana files only...")
        _local_file_check(rule_uid=rule_uid)
        print("Local provisioning/dashboard wiring looks correct.")
        return 0

    print("Reloading Grafana provisioning (dashboards + alerting)...")
    _http_json(
        "POST",
        f"{grafana_base}/api/admin/provisioning/dashboards/reload",
        username=username,
        password=password,
    )
    _http_json(
        "POST",
        f"{grafana_base}/api/admin/provisioning/alerting/reload",
        username=username,
        password=password,
    )

    print(f"Checking alert rule UID={rule_uid} exists...")
    _, rules_payload = _http_json(
        "GET",
        f"{grafana_base}/api/v1/provisioning/alert-rules",
        username=username,
        password=password,
    )
    if not _rule_exists(rules_payload, rule_uid):
        raise RuntimeError(f"Rule {rule_uid} not found after provisioning reload")
    print(f"Rule {rule_uid} is present.")

    if not args.no_force_latch:
        print("Requesting forced reconciliation latch ON...")
        request_equity_reconciliation_latch(
            filepath=control_file,
            requested_by=args.requested_by,
            reason="Forced latch for Grafana alert verification",
            block_entries=True,
        )

    print("Polling Grafana Alertmanager for firing state...")
    deadline = time.time() + max(5, int(args.timeout_seconds))
    fired = False
    while time.time() < deadline:
        _, alerts_payload = _http_json(
            "GET",
            f"{grafana_base}/api/alertmanager/grafana/api/v2/alerts?active=true",
            username=username,
            password=password,
        )
        if _is_rule_firing(alerts_payload, rule_uid):
            fired = True
            break
        time.sleep(max(0.5, float(args.poll_seconds)))

    if not args.no_force_latch and not args.no_auto_clear:
        print("Requesting forced reconciliation latch OFF...")
        request_equity_reconciliation_latch(
            filepath=control_file,
            requested_by=args.requested_by,
            reason="Clear forced latch after Grafana alert verification",
            block_entries=False,
        )

    if not fired:
        raise RuntimeError(
            f"Rule {rule_uid} did not enter firing state within {args.timeout_seconds}s"
        )

    print(f"SUCCESS: Rule {rule_uid} is provisioned and firing on forced latch scenario.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
