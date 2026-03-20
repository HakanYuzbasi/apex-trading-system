#!/usr/bin/env python3
"""
scripts/signal_comparison_report.py — Regression vs Binary Signal Accuracy Report
====================================================================================
Reads the live signal_accuracy_state.json produced by SignalAccuracyTracker
and prints a formatted comparison table.

Usage:
    python3 scripts/signal_comparison_report.py
    python3 scripts/signal_comparison_report.py --json
    python3 scripts/signal_comparison_report.py --state path/to/signal_accuracy_state.json
"""

import argparse
import json
import os
import sys

_DEFAULT_STATE = os.path.join(
    os.path.dirname(__file__), "..", "data", "signal_accuracy_state.json"
)


def _pct(v: float | None) -> str:
    if v is None:
        return "n/a"
    return f"{v * 100:.1f}%"


def _winner(reg: float | None, bin_: float | None) -> str:
    if reg is None or bin_ is None:
        return ""
    if abs(reg - bin_) < 0.01:
        return "  ~TIE"
    return "  ✅ BINARY" if bin_ > reg else "  ✅ REGRESSION"


def print_report(state_path: str, as_json: bool = False) -> None:
    if not os.path.exists(state_path):
        print(f"State file not found: {state_path}")
        print("Run the engine with BINARY_SIGNAL_ENABLED=true to generate tracking data.")
        sys.exit(1)

    with open(state_path) as fh:
        state = json.load(fh)

    resolved = state.get("resolved", [])
    n = len(resolved)

    if n == 0:
        print("No resolved predictions yet — wait for at least one completed trade.")
        sys.exit(0)

    reg_correct = [r["regression_correct"] for r in resolved]
    bin_correct = [r["binary_correct"] for r in resolved]

    reg_acc = sum(reg_correct) / n
    bin_acc = sum(bin_correct) / n
    advantage = bin_acc - reg_acc

    # Per-regime breakdown
    by_regime: dict = {}
    for r in resolved:
        reg = r.get("regime", "unknown")
        by_regime.setdefault(reg, {"reg": [], "bin": []})
        by_regime[reg]["reg"].append(r["regression_correct"])
        by_regime[reg]["bin"].append(r["binary_correct"])

    if as_json:
        output = {
            "n": n,
            "window": state.get("window", 50),
            "regression_accuracy": reg_acc,
            "binary_accuracy": bin_acc,
            "binary_advantage": advantage,
            "recommended": "binary" if advantage > 0 else "regression",
            "by_regime": {
                reg: {
                    "n": len(d["reg"]),
                    "regression_accuracy": sum(d["reg"]) / len(d["reg"]),
                    "binary_accuracy": sum(d["bin"]) / len(d["bin"]),
                }
                for reg, d in by_regime.items()
            },
        }
        print(json.dumps(output, indent=2))
        return

    # --- Human-readable output ---
    print()
    print("=" * 60)
    print("  SIGNAL ACCURACY COMPARISON: Regression vs Binary")
    print("=" * 60)
    print(f"  Resolved trades : {n}  (rolling window: {state.get('window', 50)})")
    print()
    print(f"  {'Metric':<30} {'Regression':>12} {'Binary':>12}")
    print(f"  {'-'*30} {'-'*12} {'-'*12}")
    print(f"  {'Directional accuracy':<30} {_pct(reg_acc):>12} {_pct(bin_acc):>12}{_winner(reg_acc, bin_acc)}")
    print()
    print(f"  Advantage (Binary − Regression): {advantage:+.1%}")
    rec = "BINARY" if advantage > 0.01 else ("REGRESSION" if advantage < -0.01 else "TIED")
    print(f"  Recommended primary signal     : {rec}")
    print()
    print(f"  {'By Regime':<30} {'Regression':>12} {'Binary':>12}")
    print(f"  {'-'*30} {'-'*12} {'-'*12}")
    for reg, d in sorted(by_regime.items()):
        r_acc = sum(d["reg"]) / len(d["reg"])
        b_acc = sum(d["bin"]) / len(d["bin"])
        print(
            f"  {reg + ' (n=' + str(len(d['reg'])) + ')' :<30} {_pct(r_acc):>12} {_pct(b_acc):>12}{_winner(r_acc, b_acc)}"
        )
    print()
    print("  Signal blend config (config.py):")
    try:
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
        from config import ApexConfig
        enabled = getattr(ApexConfig, "BINARY_SIGNAL_ENABLED", True)
        weight = getattr(ApexConfig, "BINARY_SIGNAL_WEIGHT", 0.40)
        horizon = getattr(ApexConfig, "BINARY_LABEL_HORIZON_DAYS", 1)
        print(f"    BINARY_SIGNAL_ENABLED       = {enabled}")
        print(f"    BINARY_SIGNAL_WEIGHT        = {weight:.0%}")
        print(f"    BINARY_LABEL_HORIZON_DAYS   = {horizon}")
        if enabled and advantage < -0.03:
            print()
            print("  ⚠️  Binary is underperforming regression by >3pp.")
            print("     Consider: BINARY_SIGNAL_WEIGHT=0.20 or BINARY_SIGNAL_ENABLED=false")
        elif enabled and advantage > 0.05:
            print()
            print("  ✅ Binary is outperforming regression by >5pp.")
            print("     Consider: BINARY_SIGNAL_WEIGHT=0.60 to increase its influence.")
    except ImportError:
        pass
    print("=" * 60)
    print()


def main() -> None:
    parser = argparse.ArgumentParser(description="Regression vs Binary signal accuracy report")
    parser.add_argument(
        "--state",
        default=_DEFAULT_STATE,
        help="Path to signal_accuracy_state.json (default: data/signal_accuracy_state.json)",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        default=False,
        help="Output as JSON",
    )
    args = parser.parse_args()
    print_report(args.state, as_json=args.json)


if __name__ == "__main__":
    main()
