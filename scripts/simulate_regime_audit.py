#!/usr/bin/env python3
"""
scripts/simulate_regime_audit.py
=================================
Simulation: Would today's changes (RegimeRouter + crypto beta scalar) have
prevented the 566 insufficient-balance rejections observed on 2026-04-28?

Methodology
-----------
1. Reconstruct today's pair portfolio from the harness BACKBENCH_UNIVERSE.
2. Apply VolatilitySizer scale (use 1.0x as proxy — sizer data unavailable offline).
3. For each cycle (10 crypto pairs × 2 legs), compute:
   a. OLD behaviour: raw leg_notional × 1.0 (no regime, no beta)
   b. NEW behaviour: leg_notional × regime_mult × crypto_beta_scalar
4. Compare total cycle demand vs. simulated account buying power.
5. Count "would-have-been-rejected" cycles in old vs new.

Run with:
    cd /Users/hakanyuzbasioglu/apex-trading
    python scripts/simulate_regime_audit.py
"""
from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np

# ---------------------------------------------------------------------------
# Crypto pairs from the live harness BACKBENCH_UNIVERSE
# ---------------------------------------------------------------------------
CRYPTO_PAIRS = [
    {"a": "CRYPTO:BTC/USD",  "b": "CRYPTO:SOL/USD",  "leg_notional": 1500.0},
    {"a": "CRYPTO:BTC/USD",  "b": "CRYPTO:AVAX/USD", "leg_notional": 1500.0},
    {"a": "CRYPTO:SOL/USD",  "b": "CRYPTO:AVAX/USD", "leg_notional": 1500.0},
    {"a": "CRYPTO:ETH/USD",  "b": "CRYPTO:LINK/USD", "leg_notional": 1200.0},
    {"a": "CRYPTO:ETH/USD",  "b": "CRYPTO:AAVE/USD", "leg_notional": 1200.0},
    {"a": "CRYPTO:BTC/USD",  "b": "CRYPTO:LTC/USD",  "leg_notional": 1200.0},
    {"a": "CRYPTO:BTC/USD",  "b": "CRYPTO:BCH/USD",  "leg_notional": 1200.0},
    {"a": "CRYPTO:ETH/USD",  "b": "CRYPTO:UNI/USD",  "leg_notional": 1000.0},
    {"a": "CRYPTO:BTC/USD",  "b": "CRYPTO:XRP/USD",  "leg_notional": 1200.0},
    {"a": "CRYPTO:ETH/USD",  "b": "CRYPTO:DOT/USD",  "leg_notional": 1000.0},
]

# Today's observed account balance range (USD, from Docker logs)
ACCOUNT_STATES_USD = [
    23601.18,  # 21:38 reading
    24403.12,  # 21:34 reading
    27232.59,  # 21:36 reading
    21514.88,  # 21:38 AVAX reading
]

# ---------------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------------
CRYPTO_BETA_SCALAR = 0.55          # New: from RegimeRouter
CASH_GATE_BUFFER   = 0.85          # New: fraction of balance available
REGIME_MULT_TODAY  = 0.85          # Today's estimated regime mult (RANGING)
# Combined new multiplier for crypto
NEW_MULT = REGIME_MULT_TODAY * CRYPTO_BETA_SCALAR  # = 0.4675

# ---------------------------------------------------------------------------
# Simulation
# ---------------------------------------------------------------------------
def simulate() -> None:
    print("=" * 65)
    print("APEX Regime Router — Simulation Replay: 2026-04-28")
    print("=" * 65)

    # Total per-cycle order demand (all 10 pairs × 2 legs)
    old_total = sum(p["leg_notional"] * 2 for p in CRYPTO_PAIRS)
    new_total = sum(p["leg_notional"] * 2 * NEW_MULT for p in CRYPTO_PAIRS)

    print(f"\nCrypto pairs active : {len(CRYPTO_PAIRS)}")
    print(f"\n{'Metric':<40} {'OLD':>12} {'NEW':>12}")
    print("-" * 65)
    print(f"{'Total cycle demand (USD)':<40} ${old_total:>10,.0f} ${new_total:>10,.0f}")
    print(f"{'Crypto beta scalar':<40} {'1.00':>12} {CRYPTO_BETA_SCALAR:>12.2f}")
    print(f"{'Regime notional mult (RANGING)':<40} {'1.00':>12} {REGIME_MULT_TODAY:>12.2f}")
    print(f"{'Combined multiplier':<40} {'1.00':>12} {NEW_MULT:>12.4f}")
    print(f"{'Demand reduction':<40} {'—':>12} {(1-NEW_MULT)*100:>10.1f}%")

    print("\n--- Rejection simulation per account balance snapshot ---")
    print(f"{'Balance':<15} {'OldDemand':>12} {'OldReject?':>12} {'NewDemand':>12} {'NewReject?':>12}")
    print("-" * 65)

    old_rejects = 0
    new_rejects = 0
    for bal in ACCOUNT_STATES_USD:
        available = bal * CASH_GATE_BUFFER
        old_rej = "🔴 YES" if old_total > available else "✅ NO"
        new_rej = "🔴 YES" if new_total > available else "✅ NO"
        if old_total > available:
            old_rejects += 1
        if new_total > available:
            new_rejects += 1
        print(f"  ${bal:>10,.2f}   ${old_total:>9,.0f}  {old_rej:>12}   ${new_total:>9,.0f}  {new_rej:>12}")

    print(f"\nRejected cycles : OLD={old_rejects}/{len(ACCOUNT_STATES_USD)}  "
          f"NEW={new_rejects}/{len(ACCOUNT_STATES_USD)}")

    # Scale to full-day estimate
    # Docker logs showed 566 rejections over ~24h = ~9 cycles/hour × 24 hours
    CYCLES_PER_DAY = 216
    old_scaled = int((old_rejects / len(ACCOUNT_STATES_USD)) * CYCLES_PER_DAY * len(CRYPTO_PAIRS))
    new_scaled = int((new_rejects / len(ACCOUNT_STATES_USD)) * CYCLES_PER_DAY * len(CRYPTO_PAIRS))

    print(f"\n--- Extrapolated full-day rejection estimate ---")
    print(f"  OLD logic  : ~{old_scaled} rejections/day  (observed: 566)")
    print(f"  NEW logic  : ~{new_scaled} rejections/day  (target: <20)")

    reduction = max(0, (old_scaled - new_scaled) / max(old_scaled, 1) * 100)
    print(f"  Reduction  : {reduction:.0f}%")

    print("\n--- Per-pair notional impact ---")
    print(f"  {'Pair':<35} {'OldLeg':>9} {'NewLeg':>9} {'Delta':>9}")
    print("  " + "-" * 62)
    for p in CRYPTO_PAIRS:
        old_leg = p["leg_notional"]
        new_leg = old_leg * NEW_MULT
        print(f"  {p['a'].split(':')[1]:<16} ↔ {p['b'].split(':')[1]:<16}  "
              f"${old_leg:>7,.0f}  ${new_leg:>7,.0f}  -${old_leg-new_leg:>6,.0f}")

    print("\n" + "=" * 65)
    if new_scaled < 20:
        print("✅ SIMULATION PASSED: Regime router would eliminate ~95%+ of")
        print("   insufficient-balance rejections under today's conditions.")
    else:
        print("⚠️  SIMULATION WARNING: Regime router reduces but does not")
        print("   eliminate rejections. Consider lowering CRYPTO_BETA_SCALAR further.")
    print("=" * 65)


if __name__ == "__main__":
    simulate()
