#!/usr/bin/env python3
"""
Tune execution spread/slippage/edge defaults from recent live telemetry.

Inputs:
- Connector/execution logs in logs/apex.log*
- Attribution state in data/performance_attribution.json

Recommends:
- APEX_EXECUTION_MAX_SPREAD_BPS_{EQUITY,FX,CRYPTO}
- APEX_EXECUTION_SLIPPAGE_BUDGET_BPS_{EQUITY,FX,CRYPTO}
- APEX_EXECUTION_SIGNAL_TO_EDGE_BPS
- APEX_EXECUTION_MIN_EDGE_OVER_COST_BPS_{EQUITY,FX,CRYPTO}
"""

from __future__ import annotations

import argparse
import difflib
import json
import re
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, List, Optional
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.symbols import parse_symbol


PAT_TS = re.compile(r"^(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}),\d{3}\s+-")
PAT_ALPACA = re.compile(
    r"execution\.alpaca_connector - .*?\b(?:BUY|SELL)\s+[\d.]+\s+([A-Z0-9:/._-]+)\s+@\s*\$[\d.]+\s+\(Alpaca,\s*slippage:\s*([+-]?\d+(?:\.\d+)?)\s*bps\)",
    re.I,
)
PAT_IBKR_HI = re.compile(
    r"execution\.ibkr_connector - .*?High slippage on\s+([^:]+):\s*([+-]?\d+(?:\.\d+)?)\s*bps",
    re.I,
)
PAT_IBKR_INFO = re.compile(
    r"execution\.ibkr_connector - .*?\b([A-Z0-9:/._-]+)\s+slippage:\s*([+-]?\d+(?:\.\d+)?)\s*bps",
    re.I,
)


PRIORS_SPREAD = {"EQUITY": 12.0, "FOREX": 8.0, "CRYPTO": 26.0}
PRIORS_BUDGET = {"EQUITY": 221.0, "FOREX": 180.0, "CRYPTO": 156.0}
PRIORS_EDGE_BUFFER = {"EQUITY": 8.0, "FOREX": 6.0, "CRYPTO": 12.0}
PRIORS_SIGNAL_TO_EDGE = 80.0


def _parse_line_timestamp(line: str) -> Optional[datetime]:
    match = PAT_TS.match(line)
    if not match:
        return None
    try:
        return datetime.strptime(match.group(1), "%Y-%m-%d %H:%M:%S")
    except Exception:
        return None


def _parse_iso_ts(value: str) -> Optional[datetime]:
    if not value:
        return None
    try:
        dt = datetime.fromisoformat(str(value))
    except Exception:
        return None
    if dt.tzinfo is not None:
        return dt.astimezone(timezone.utc).replace(tzinfo=None)
    return dt


def _asset_from_symbol(symbol: str) -> Optional[str]:
    try:
        asset = parse_symbol(symbol).asset_class.value
    except Exception:
        return None
    normalized = str(asset).upper()
    if normalized not in {"EQUITY", "FOREX", "CRYPTO"}:
        return None
    return normalized


def parse_live_slippage(
    log_dir: Path,
    tail_lines_per_file: Optional[int],
    cutoff: datetime,
) -> Dict[str, List[float]]:
    """Parse absolute slippage bps from connector logs by asset class."""
    by_asset: Dict[str, List[float]] = defaultdict(list)
    files = sorted(log_dir.glob("apex.log*"), key=lambda p: p.stat().st_mtime, reverse=True)
    for fp in files:
        text = fp.read_text(errors="ignore")
        lines = text.splitlines()
        if tail_lines_per_file and tail_lines_per_file > 0:
            lines = lines[-tail_lines_per_file:]

        for line in lines:
            ts = _parse_line_timestamp(line)
            if ts is not None and ts < cutoff:
                continue

            symbol = None
            slippage = None
            m = PAT_ALPACA.search(line)
            if m:
                symbol, slippage = m.group(1), float(m.group(2))
            else:
                m = PAT_IBKR_HI.search(line)
                if m:
                    symbol, slippage = m.group(1), float(m.group(2))
                else:
                    m = PAT_IBKR_INFO.search(line)
                    if m:
                        symbol, slippage = m.group(1), float(m.group(2))

            if symbol is None or slippage is None:
                continue
            asset = _asset_from_symbol(symbol)
            if asset is None:
                continue
            by_asset[asset].append(abs(float(slippage)))
    return by_asset


def parse_attribution_telemetry(
    attribution_file: Path,
    cutoff: datetime,
) -> Dict[str, Dict[str, List[float]]]:
    """
    Parse attribution telemetry for edge/cost calibration.

    Returns per asset class lists:
    - implied_signal_to_edge
    - entry_slippage_bps
    - execution_cost_bps
    """
    result: Dict[str, Dict[str, List[float]]] = {
        asset: {
            "implied_signal_to_edge": [],
            "entry_slippage_bps": [],
            "execution_cost_bps": [],
        }
        for asset in ("EQUITY", "FOREX", "CRYPTO")
    }
    if not attribution_file.exists():
        return result

    try:
        payload = json.loads(attribution_file.read_text(encoding="utf-8"))
    except Exception:
        return result

    open_positions = payload.get("open_positions", {}) or {}
    if isinstance(open_positions, dict):
        rows = list(open_positions.values())
    else:
        rows = []

    closed_trades = payload.get("closed_trades", []) or []
    if isinstance(closed_trades, list):
        rows.extend(closed_trades)

    for row in rows:
        if not isinstance(row, dict):
            continue
        ts = _parse_iso_ts(str(row.get("exit_time") or row.get("entry_time") or ""))
        if ts and ts < cutoff:
            continue
        asset = str(row.get("asset_class", "EQUITY")).upper()
        if asset not in result:
            asset = _asset_from_symbol(str(row.get("symbol", ""))) or "EQUITY"

        signal = abs(float(row.get("entry_signal", 0.0) or 0.0))
        conf = max(0.0, min(1.0, float(row.get("entry_confidence", 0.0) or 0.0)))
        entry_slip = abs(float(row.get("entry_slippage_bps", 0.0) or 0.0))
        if entry_slip > 0:
            result[asset]["entry_slippage_bps"].append(entry_slip)
        if signal >= 0.01 and entry_slip >= 0.1:
            denom = signal * (0.4 + 0.6 * conf)
            if denom > 1e-6:
                implied = entry_slip / denom
                if np.isfinite(implied) and 1.0 <= implied <= 500.0:
                    result[asset]["implied_signal_to_edge"].append(float(implied))

        quantity = abs(float(row.get("quantity", 0.0) or 0.0))
        entry_price = abs(float(row.get("entry_price", 0.0) or 0.0))
        notional = quantity * entry_price
        modeled_exec_drag = abs(float(row.get("modeled_execution_drag", 0.0) or 0.0))
        if notional > 1e-9 and modeled_exec_drag > 0:
            cost_bps = modeled_exec_drag / notional * 10000.0
            if np.isfinite(cost_bps):
                result[asset]["execution_cost_bps"].append(float(cost_bps))

    return result


def _safe_percentile(samples: List[float], q: float, default: float = 0.0) -> float:
    if not samples:
        return float(default)
    arr = np.asarray(samples, dtype=float)
    if arr.size == 0:
        return float(default)
    return float(np.percentile(arr, q))


def recommend_thresholds(
    by_asset_slippage: Dict[str, List[float]],
    attribution: Dict[str, Dict[str, List[float]]],
    window: int,
) -> Dict[str, Dict[str, float]]:
    """
    Produce tuned spread/budget/edge recommendations.
    Uses shrinkage toward current defaults when sample size is low.
    """
    spread_floor = {"EQUITY": 8.0, "FOREX": 5.0, "CRYPTO": 18.0}
    spread_cap = {"EQUITY": 30.0, "FOREX": 20.0, "CRYPTO": 70.0}
    budget_floor = {"EQUITY": 120.0, "FOREX": 100.0, "CRYPTO": 120.0}
    budget_cap = {"EQUITY": 450.0, "FOREX": 350.0, "CRYPTO": 500.0}
    edge_floor = {"EQUITY": 6.0, "FOREX": 4.0, "CRYPTO": 8.0}
    edge_cap = {"EQUITY": 24.0, "FOREX": 18.0, "CRYPTO": 32.0}

    rec: Dict[str, Dict[str, float]] = {}
    implied_global: List[float] = []

    for asset in ("EQUITY", "FOREX", "CRYPTO"):
        fill_samples = list(by_asset_slippage.get(asset, []))
        att = attribution.get(asset, {})
        implied_samples = list(att.get("implied_signal_to_edge", []))
        implied_global.extend(implied_samples)
        cost_samples = (
            fill_samples
            + list(att.get("entry_slippage_bps", []))
            + list(att.get("execution_cost_bps", []))
        )

        n = len(fill_samples)
        p75 = _safe_percentile(fill_samples, 75)
        p95 = _safe_percentile(fill_samples, 95)

        if n == 0:
            spread = PRIORS_SPREAD[asset]
            budget = PRIORS_BUDGET[asset]
        else:
            derived_spread = min(
                spread_cap[asset],
                max(spread_floor[asset], round(p95 * 1.25)),
            )
            derived_budget = min(
                budget_cap[asset],
                max(
                    budget_floor[asset],
                    round(max(p75 * window * 1.15, derived_spread * 6)),
                ),
            )
            weight = min(1.0, n / 15.0)
            spread = round((weight * derived_spread) + ((1.0 - weight) * PRIORS_SPREAD[asset]))
            budget = round((weight * derived_budget) + ((1.0 - weight) * PRIORS_BUDGET[asset]))

        c75 = _safe_percentile(cost_samples, 75)
        c90 = _safe_percentile(cost_samples, 90)
        cost_n = len(cost_samples)
        if cost_n == 0:
            edge_buffer = PRIORS_EDGE_BUFFER[asset]
        else:
            derived_edge_buffer = min(
                edge_cap[asset],
                max(edge_floor[asset], round((c75 * 0.35) + (c90 * 0.15))),
            )
            weight = min(1.0, cost_n / 60.0)
            edge_buffer = round(
                (weight * derived_edge_buffer)
                + ((1.0 - weight) * PRIORS_EDGE_BUFFER[asset])
            )

        rec[asset] = {
            "samples": float(n),
            "cost_samples": float(cost_n),
            "implied_samples": float(len(implied_samples)),
            "p75_abs_bps": round(p75, 2),
            "p95_abs_bps": round(p95, 2),
            "spread_bps": float(spread),
            "slippage_budget_bps": float(budget),
            "edge_buffer_bps": float(edge_buffer),
        }

    implied_n = len(implied_global)
    if implied_n == 0:
        signal_to_edge = PRIORS_SIGNAL_TO_EDGE
    else:
        implied_p65 = _safe_percentile(implied_global, 65)
        derived_signal_to_edge = max(40.0, min(220.0, implied_p65 * 1.1))
        weight = min(1.0, implied_n / 120.0)
        signal_to_edge = round(
            (weight * derived_signal_to_edge)
            + ((1.0 - weight) * PRIORS_SIGNAL_TO_EDGE),
            1,
        )

    rec["_meta"] = {
        "signal_to_edge_bps": float(signal_to_edge),
        "signal_to_edge_samples": float(implied_n),
    }
    return rec


def _recommended_env_map(rec: Dict[str, Dict[str, float]]) -> Dict[str, str]:
    meta = rec.get("_meta", {})
    return {
        "APEX_EXECUTION_MAX_SPREAD_BPS_EQUITY": str(int(rec["EQUITY"]["spread_bps"])),
        "APEX_EXECUTION_MAX_SPREAD_BPS_FX": str(int(rec["FOREX"]["spread_bps"])),
        "APEX_EXECUTION_MAX_SPREAD_BPS_CRYPTO": str(int(rec["CRYPTO"]["spread_bps"])),
        "APEX_EXECUTION_SLIPPAGE_BUDGET_BPS_EQUITY": str(int(rec["EQUITY"]["slippage_budget_bps"])),
        "APEX_EXECUTION_SLIPPAGE_BUDGET_BPS_FX": str(int(rec["FOREX"]["slippage_budget_bps"])),
        "APEX_EXECUTION_SLIPPAGE_BUDGET_BPS_CRYPTO": str(int(rec["CRYPTO"]["slippage_budget_bps"])),
        "APEX_EXECUTION_SIGNAL_TO_EDGE_BPS": str(float(meta.get("signal_to_edge_bps", PRIORS_SIGNAL_TO_EDGE))),
        "APEX_EXECUTION_MIN_EDGE_OVER_COST_BPS_EQUITY": str(int(rec["EQUITY"]["edge_buffer_bps"])),
        "APEX_EXECUTION_MIN_EDGE_OVER_COST_BPS_FX": str(int(rec["FOREX"]["edge_buffer_bps"])),
        "APEX_EXECUTION_MIN_EDGE_OVER_COST_BPS_CRYPTO": str(int(rec["CRYPTO"]["edge_buffer_bps"])),
    }


def _patched_env_content(env_content: str, replacements: Dict[str, str]) -> str:
    lines = env_content.splitlines()
    seen = set()
    out_lines: List[str] = []
    for line in lines:
        if "=" not in line or line.lstrip().startswith("#"):
            out_lines.append(line)
            continue
        key, _, _ = line.partition("=")
        key = key.strip()
        if key in replacements:
            out_lines.append(f"{key}={replacements[key]}")
            seen.add(key)
        else:
            out_lines.append(line)
    for key, value in replacements.items():
        if key not in seen:
            out_lines.append(f"{key}={value}")
    return "\n".join(out_lines) + "\n"


def _render_report(
    rec: Dict[str, Dict[str, float]],
    window: int,
    log_dir: str,
    attribution_file: str,
    lookback_days: int,
    env_diff_path: str,
) -> str:
    ts = datetime.now(timezone.utc).isoformat()
    meta = rec.get("_meta", {})
    signal_to_edge = float(meta.get("signal_to_edge_bps", PRIORS_SIGNAL_TO_EDGE))
    signal_samples = int(meta.get("signal_to_edge_samples", 0))
    lines = [
        "# Execution Threshold Tuning Report",
        "",
        f"- Generated at: `{ts}`",
        f"- Log source: `{log_dir}`",
        f"- Attribution source: `{attribution_file}`",
        f"- Lookback window: `{lookback_days}` days",
        f"- Slippage budget window: `{window}` fills",
        "",
        "## Recommendations by Asset Class",
        "",
        "| Asset | Fill Samples | Cost Samples | P75 Abs Slippage (bps) | P95 Abs Slippage (bps) | Max Spread (bps) | Slippage Budget (bps) | Min Edge Buffer (bps) |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for asset in ("EQUITY", "FOREX", "CRYPTO"):
        row = rec[asset]
        lines.append(
            f"| {asset} | {int(row['samples'])} | {int(row['cost_samples'])} | "
            f"{row['p75_abs_bps']} | {row['p95_abs_bps']} | {row['spread_bps']} | "
            f"{row['slippage_budget_bps']} | {row['edge_buffer_bps']} |"
        )
    lines.extend(
        [
            "",
            "## Global Edge Calibration",
            "",
            f"- Recommended `APEX_EXECUTION_SIGNAL_TO_EDGE_BPS`: `{signal_to_edge}`",
            f"- Attribution-derived signal/edge samples: `{signal_samples}`",
            "",
            "## Approval Gate",
            "",
            f"- Review proposed env diff: `{env_diff_path}`",
            "- Apply only after human approval.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--log-dir", default="logs")
    parser.add_argument("--attribution-file", default="data/performance_attribution.json")
    parser.add_argument(
        "--lookback-days",
        type=int,
        default=28,
        help="Telemetry lookback window in days (recommended 14-28).",
    )
    parser.add_argument(
        "--tail-lines-per-file",
        type=int,
        default=0,
        help="Only parse the most recent N lines in each log file (0 = full file).",
    )
    parser.add_argument(
        "--window",
        type=int,
        default=20,
        help="Rolling fill window for slippage budget calibration.",
    )
    parser.add_argument("--env-file", default=".env.example")
    parser.add_argument("--write-overrides", default="")
    parser.add_argument("--write-diff", default="")
    parser.add_argument("--write-report", default="")
    args = parser.parse_args()

    lookback_days = max(1, int(args.lookback_days))
    cutoff = datetime.now() - timedelta(days=lookback_days)

    by_asset_slippage = parse_live_slippage(
        log_dir=Path(args.log_dir),
        tail_lines_per_file=args.tail_lines_per_file if args.tail_lines_per_file > 0 else None,
        cutoff=cutoff,
    )
    attribution = parse_attribution_telemetry(
        attribution_file=Path(args.attribution_file),
        cutoff=cutoff,
    )
    rec = recommend_thresholds(
        by_asset_slippage=by_asset_slippage,
        attribution=attribution,
        window=max(1, int(args.window)),
    )
    env_replacements = _recommended_env_map(rec)

    env_file = Path(args.env_file)
    env_before = env_file.read_text() if env_file.exists() else ""
    env_after = _patched_env_content(env_before, env_replacements)
    diff_lines = list(
        difflib.unified_diff(
            env_before.splitlines(keepends=True),
            env_after.splitlines(keepends=True),
            fromfile=f"{args.env_file}.current",
            tofile=f"{args.env_file}.recommended",
        )
    )
    env_diff = "".join(diff_lines)

    print("Execution threshold recommendations:")
    for asset in ("EQUITY", "FOREX", "CRYPTO"):
        row = rec[asset]
        print(
            f"{asset}: fill_samples={int(row['samples'])}, cost_samples={int(row['cost_samples'])}, "
            f"p75={row['p75_abs_bps']}bps, p95={row['p95_abs_bps']}bps, "
            f"spread={row['spread_bps']}bps, budget={row['slippage_budget_bps']}bps, "
            f"edge_buffer={row['edge_buffer_bps']}bps"
        )
    meta = rec.get("_meta", {})
    print(
        f"SIGNAL_TO_EDGE: {float(meta.get('signal_to_edge_bps', PRIORS_SIGNAL_TO_EDGE))} "
        f"(samples={int(meta.get('signal_to_edge_samples', 0))})"
    )

    print("\nEnv overrides:")
    for key, value in env_replacements.items():
        print(f"{key}={value}")

    if args.write_overrides:
        out = Path(args.write_overrides)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text("\n".join(f"{k}={v}" for k, v in env_replacements.items()) + "\n")

    if args.write_diff:
        out = Path(args.write_diff)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(env_diff)

    if args.write_report:
        out = Path(args.write_report)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(
            _render_report(
                rec=rec,
                window=max(1, int(args.window)),
                log_dir=args.log_dir,
                attribution_file=args.attribution_file,
                lookback_days=lookback_days,
                env_diff_path=args.write_diff or "(not generated)",
            )
        )

    if env_diff:
        print("\nRecommended env diff:")
        print(env_diff)
    else:
        print("\nNo env changes recommended.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
