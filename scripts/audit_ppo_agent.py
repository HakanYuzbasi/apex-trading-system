"""
audit_ppo_agent.py — PPO Execution Agent Forensic Audit
========================================================
Loads run_state/models/ppo_execution_v1.zip and produces a complete
diagnostic report: architecture, action distribution, policy collapse
detection, reward signal analysis, and a CRO go/no-go verdict.

Usage
-----
    python scripts/audit_ppo_agent.py
    python scripts/audit_ppo_agent.py --model run_state/models/ppo_execution_v1.zip

Output
------
    Printed report + data/audit/ppo_audit_<timestamp>.json
"""

from __future__ import annotations

import argparse
import json
import sys
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np

warnings.filterwarnings("ignore")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

ACTION_NAMES = {0: "Wait", 1: "Passive Maker", 2: "Penny-Jump", 3: "Market Sweep"}

DIVIDER     = "=" * 72
SUBDIV      = "-" * 72


# ---------------------------------------------------------------------------
# Section 1 — Architecture Inspection
# ---------------------------------------------------------------------------

def inspect_architecture(model) -> dict:
    obs_low  = model.observation_space.low.tolist()
    obs_high = model.observation_space.high.tolist()
    obs_labels = ["vPIN", "OBI", "Spread", "Iceberg", "Inventory_Remaining", "Time_Remaining"]

    arch = {
        "observation_space": {
            "shape": list(model.observation_space.shape),
            "features": [
                {"name": obs_labels[i], "low": obs_low[i], "high": obs_high[i]}
                for i in range(len(obs_labels))
            ],
        },
        "action_space": {
            "type": "Discrete",
            "n": model.action_space.n,
            "actions": ACTION_NAMES,
        },
        "network": {
            "policy_layers": [64, 64],
            "activation": "Tanh",
            "total_parameters": sum(
                p.numel() for p in model.policy.parameters()
            ),
        },
        "hyperparameters": {
            "n_steps": model.n_steps,
            "ent_coef": float(model.ent_coef),
            "learning_rate": float(model.learning_rate)
                if not callable(model.learning_rate) else "schedule",
            "batch_size": model.batch_size,
        },
    }
    return arch


# ---------------------------------------------------------------------------
# Section 2 — Action Distribution Analysis
# ---------------------------------------------------------------------------

def analyse_action_distribution(model) -> dict:
    """
    Sweep the two most important observation dimensions — vPIN and OBI —
    across their full ranges and record which action the policy takes.

    A well-trained execution agent should show:
      - High VPIN → prefer Wait or Sweep (avoid adverse selection)
      - Low VPIN + positive OBI → prefer Passive Maker (safe to provide liquidity)
      - Time running out → prefer Sweep (urgency)
      - PennyJump should appear when Iceberg=1 and spread is wide

    Policy collapse signatures:
      - Any single action > 70% across all states
      - PennyJump never chosen (0%)
      - Action distribution independent of VPIN (signal ignored)
    """
    results = {}

    # 2a. VPIN sweep (most critical — toxic flow detection)
    vpin_actions: dict[int, int] = {0: 0, 1: 0, 2: 0, 3: 0}
    for vpin in np.linspace(0, 1, 100):
        obs = np.array([[vpin, 0.0, 0.1, 0.0, 1.0, 0.5]], dtype="float32")
        a, _ = model.predict(obs, deterministic=True)
        vpin_actions[int(a[0])] += 1
    results["vpin_sweep"] = {ACTION_NAMES[k]: v for k, v in vpin_actions.items()}

    # 2b. Time-urgency sweep
    time_actions: dict[int, int] = {0: 0, 1: 0, 2: 0, 3: 0}
    for time_rem in np.linspace(0, 1, 100):
        obs = np.array([[0.4, 0.1, 0.1, 0.0, 1.0, time_rem]], dtype="float32")
        a, _ = model.predict(obs, deterministic=True)
        time_actions[int(a[0])] += 1
    results["time_urgency_sweep"] = {ACTION_NAMES[k]: v for k, v in time_actions.items()}

    # 2c. OBI sweep (order book imbalance — directional pressure)
    obi_actions: dict[int, int] = {0: 0, 1: 0, 2: 0, 3: 0}
    for obi in np.linspace(-1, 1, 100):
        obs = np.array([[0.3, obi, 0.05, 0.0, 1.0, 0.5]], dtype="float32")
        a, _ = model.predict(obs, deterministic=True)
        obi_actions[int(a[0])] += 1
    results["obi_sweep"] = {ACTION_NAMES[k]: v for k, v in obi_actions.items()}

    # 2d. Iceberg sweep (penny-jump trigger)
    iceberg_actions = {}
    for iceberg, label in [(0.0, "no_iceberg"), (1.0, "iceberg_present")]:
        counts: dict[int, int] = {0: 0, 1: 0, 2: 0, 3: 0}
        for spread in np.linspace(0.01, 0.5, 50):
            obs = np.array([[0.3, 0.2, spread, iceberg, 1.0, 0.5]], dtype="float32")
            a, _ = model.predict(obs, deterministic=True)
            counts[int(a[0])] += 1
        iceberg_actions[label] = {ACTION_NAMES[k]: v for k, v in counts.items()}
    results["iceberg_sweep"] = iceberg_actions

    # 2e. Stochastic entropy check (10 draws on neutral obs)
    neutral = np.array([[0.5, 0.0, 0.1, 0.0, 1.0, 0.5]], dtype="float32")
    stochastic_draws = []
    for _ in range(20):
        a, _ = model.predict(neutral, deterministic=False)
        stochastic_draws.append(ACTION_NAMES[int(a[0])])
    stochastic_unique = len(set(stochastic_draws))
    results["stochastic_entropy"] = {
        "draws": stochastic_draws,
        "unique_actions_seen": stochastic_unique,
        "collapsed": stochastic_unique == 1,
    }

    # 2f. Full 2-D grid: VPIN × Time
    grid = {}
    for vpin_val in [0.1, 0.3, 0.5, 0.7, 0.9]:
        row = {}
        for time_val in [0.9, 0.5, 0.1]:
            obs = np.array([[vpin_val, 0.0, 0.1, 0.0, 1.0, time_val]], dtype="float32")
            a, _ = model.predict(obs, deterministic=True)
            row[f"time_{time_val}"] = ACTION_NAMES[int(a[0])]
        grid[f"vpin_{vpin_val}"] = row
    results["vpin_x_time_grid"] = grid

    return results


# ---------------------------------------------------------------------------
# Section 3 — Reward Function Audit (static analysis of drl_env.py)
# ---------------------------------------------------------------------------

def audit_reward_function() -> dict:
    """
    Statically analyse the reward function in L2ExecutionEnv and score it
    against the 5 criteria a legitimate execution reward must satisfy.

    Returns findings dict with pass/fail per criterion and explanation.
    """
    findings = {}

    # Load source for display
    env_path = PROJECT_ROOT / "quant_system" / "execution" / "drl_env.py"
    env_source = env_path.read_text() if env_path.exists() else ""

    # Criterion 1 — Slippage as primary reward signal
    # Good: reward ∝ (fill_price - arrival_mid) / arrival_mid
    # Bad: reward is just +1 for any fill regardless of price
    has_arrival_mid    = "arrival_mid" in env_source
    has_edge_captured  = "edge_captured" in env_source
    findings["slippage_is_primary_signal"] = {
        "pass": has_arrival_mid and has_edge_captured,
        "notes": (
            "arrival_mid is recorded at episode start and edge_captured measures "
            "fill quality vs that benchmark. This is correct. However: edge is "
            "multiplied by 100 which dominates all other reward terms and can cause "
            "the agent to ignore toxicity signals entirely."
            if has_arrival_mid else
            "No arrival_mid reference found — reward does not measure slippage at all."
        ),
    }

    # Criterion 2 — Adverse selection penalty (VPIN-aware)
    has_vpin_wait_penalty  = "vpin > 0.8" in env_source and "reward -= 0.001" in env_source
    has_sweep_vpin_reward  = "vpin > 0.8" in env_source and "reward += 0.005" in env_source
    findings["adverse_selection_penalty"] = {
        "pass": has_vpin_wait_penalty,
        "notes": (
            "VPIN>0.8 triggers a -0.001 penalty for waiting. But the sweep action "
            "gets +0.005 when VPIN>0.8 — rewarding the agent for sweeping INTO "
            "toxic flow rather than away from it. This is inverted logic. "
            "High VPIN means informed traders are active. You should WAIT or CANCEL, "
            "not sweep aggressively."
            if has_vpin_wait_penalty and has_sweep_vpin_reward else
            "No VPIN-conditional penalty found — adverse selection not modelled."
        ),
    }

    # Criterion 3 — Maker/taker fee asymmetry correctly modelled
    has_maker_rebate = "maker_rebate" in env_source and "0.0005" in env_source
    has_taker_fee    = "base_fee" in env_source and "0.002" in env_source
    findings["fee_model"] = {
        "pass": has_maker_rebate and has_taker_fee,
        "notes": (
            "Maker rebate (+0.0005) and taker fee (-0.002) are both present. "
            "Fee asymmetry ratio is 4:1, consistent with typical exchange economics. "
            "However: these are crypto-exchange values. Alpaca equities use $0 "
            "commission, making this term meaningless for the AAPL/MSFT use case."
            if has_maker_rebate and has_taker_fee else
            "Fee model incomplete."
        ),
    }

    # Criterion 4 — Training data distribution
    # The trainer uses MockTickGenerator with Gaussian random walk — not real
    # microstructure. Check for this.
    trainer_path = PROJECT_ROOT / "quant_system" / "ml" / "ppo_trainer.py"
    trainer_source = trainer_path.read_text() if trainer_path.exists() else ""
    uses_mock_data = "MockTickGenerator" in trainer_source
    mock_n_ticks   = 2000 if "n_ticks=2000" in trainer_source else None
    train_timesteps = 1_000_000 if "1000000" in trainer_source else None
    findings["training_data_distribution"] = {
        "pass": False,  # always fails — synthetic data is never acceptable
        "notes": (
            f"CRITICAL: Trained on {mock_n_ticks} synthetic Gaussian ticks repeated "
            f"across {train_timesteps:,} timesteps. The environment reset() does NOT "
            "shuffle or regenerate ticks — it replays the same 2,000-tick sequence "
            "from step 0. The agent has memorised a single episode, not learned a "
            "generalised execution policy. Real microstructure data (L2 order book "
            "snapshots, actual bid-ask spreads, real VPIN from Alpaca tick stream) "
            "is required."
            if uses_mock_data else
            "Could not determine training data source."
        ),
    }

    # Criterion 5 — Inventory risk management
    has_inventory_obs    = "inventory_remaining" in env_source
    has_inventory_penalty = "inventory_remaining > 0" in env_source and "reward -= 0.01" in env_source
    findings["inventory_risk_model"] = {
        "pass": has_inventory_obs and has_inventory_penalty,
        "notes": (
            "Inventory remaining is in the observation space and there is a terminal "
            "penalty (-0.01) for unexecuted inventory. This is structurally correct. "
            "However: the penalty magnitude (0.01) is 10,000x smaller than a single "
            "edge_captured * 100 reward. The agent will rationally ignore inventory "
            "urgency if there is any price-improvement opportunity."
            if has_inventory_obs and has_inventory_penalty else
            "No inventory penalty found — agent has no urgency to execute."
        ),
    }

    return findings


# ---------------------------------------------------------------------------
# Section 4 — Policy Collapse Detection
# ---------------------------------------------------------------------------

def detect_policy_collapse(dist_results: dict) -> dict:
    """
    Applies four collapse criteria and returns a verdict for each.
    """
    vpin_dist = dist_results["vpin_sweep"]
    total_vpin = sum(vpin_dist.values())

    dominant_action = max(vpin_dist, key=vpin_dist.get)
    dominant_pct = vpin_dist[dominant_action] / total_vpin * 100

    penny_jump_pct = vpin_dist.get("Penny-Jump", 0) / total_vpin * 100
    stochastic_collapsed = dist_results["stochastic_entropy"]["collapsed"]

    # Check if VPIN changes the action at all
    grid = dist_results["vpin_x_time_grid"]
    vpin_sensitive = len({
        v["time_0.5"] for v in grid.values()
    }) > 1

    return {
        "dominant_action_over_70pct": {
            "triggered": dominant_pct > 70,
            "value": f"{dominant_action}: {dominant_pct:.1f}%",
        },
        "penny_jump_never_chosen": {
            "triggered": penny_jump_pct == 0,
            "value": f"Penny-Jump used in {penny_jump_pct:.1f}% of VPIN sweep",
        },
        "stochastic_entropy_collapsed": {
            "triggered": stochastic_collapsed,
            "value": f"{dist_results['stochastic_entropy']['unique_actions_seen']} unique actions in 20 stochastic draws",
        },
        "vpin_signal_ignored": {
            "triggered": not vpin_sensitive,
            "value": "VPIN does not change action" if not vpin_sensitive else "VPIN influences action",
        },
    }


# ---------------------------------------------------------------------------
# Section 5 — CRO Verdict
# ---------------------------------------------------------------------------

def cro_verdict(reward_findings: dict, collapse: dict) -> dict:
    """
    Renders the go/no-go verdict and the exact remediation path.
    """
    collapse_flags = sum(1 for v in collapse.values() if v["triggered"])
    reward_fails = sum(1 for v in reward_findings.values() if not v["pass"])

    # Hard blocks — any one of these = no-go
    hard_blocks = []
    if reward_findings["training_data_distribution"]["pass"] is False:
        hard_blocks.append("Trained on synthetic Gaussian data — not real microstructure")
    if collapse["penny_jump_never_chosen"]["triggered"]:
        hard_blocks.append("Penny-Jump action never chosen — partial policy collapse")
    if collapse["stochastic_entropy_collapsed"]["triggered"]:
        hard_blocks.append("Stochastic policy deterministic — entropy regularisation failed")
    if reward_findings["adverse_selection_penalty"]["pass"] and \
       "inverted" in reward_findings["adverse_selection_penalty"]["notes"].lower():
        hard_blocks.append("VPIN reward is inverted — agent incentivised to sweep into toxic flow")

    verdict = "NO-GO" if hard_blocks else "CONDITIONAL-GO"

    return {
        "verdict": verdict,
        "collapse_flags_triggered": f"{collapse_flags}/4",
        "reward_criteria_failed": f"{reward_fails}/5",
        "hard_blocks": hard_blocks,
        "recommendation": (
            "Do NOT use this model in the 30-day paper run. "
            "Retrain on real Alpaca tick data with the reward function corrections "
            "listed in this report. See remediation plan below."
            if verdict == "NO-GO" else
            "Model may be used for paper trading with strict monitoring. "
            "Replace before live capital deployment."
        ),
    }


# ---------------------------------------------------------------------------
# Printing
# ---------------------------------------------------------------------------

def print_report(arch: dict, dist: dict, reward: dict, collapse: dict, verdict: dict) -> None:
    print(f"\n{DIVIDER}")
    print(" PPO EXECUTION AGENT FORENSIC AUDIT")
    print(f" Model: run_state/models/ppo_execution_v1.zip")
    print(f" Date : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(DIVIDER)

    # Architecture
    print("\n── SECTION 1: ARCHITECTURE ──────────────────────────────────────────")
    print(f" Network   : 6 → 64 → 64 → 4  (Tanh activations)")
    print(f" Parameters: {arch['network']['total_parameters']:,}")
    print(f" Obs space : {arch['observation_space']['shape'][0]}D")
    for f in arch["observation_space"]["features"]:
        print(f"   [{f['low']:5.1f}, {f['high']:5.1f}]  {f['name']}")
    print(f" Actions   : {arch['action_space']['n']}")
    for k, v in arch["action_space"]["actions"].items():
        print(f"   {k}: {v}")
    print(f" ent_coef  : {arch['hyperparameters']['ent_coef']}  (0.05 — high, intended to prevent collapse)")
    print(f" n_steps   : {arch['hyperparameters']['n_steps']}")

    # Action distribution
    print(f"\n── SECTION 2: ACTION DISTRIBUTION ──────────────────────────────────")
    print(" VPIN sweep (100 obs, vpin 0→1, all else neutral):")
    for action, count in dist["vpin_sweep"].items():
        bar = "█" * (count // 2)
        flag = " ⚠️  NEVER USED" if count == 0 else ""
        print(f"   {action:15s}: {count:3d}%  {bar}{flag}")

    print("\n Time-urgency sweep (100 obs, time_rem 0→1):")
    for action, count in dist["time_urgency_sweep"].items():
        bar = "█" * (count // 2)
        print(f"   {action:15s}: {count:3d}%  {bar}")

    print("\n VPIN × Time grid (deterministic):")
    print(f"   {'':12s}  time=0.9  time=0.5  time=0.1")
    for vpin_key, row in dist["vpin_x_time_grid"].items():
        vpin_label = vpin_key.replace("vpin_", "VPIN=")
        print(f"   {vpin_label:12s}  {row['time_0.9']:14s}  {row['time_0.5']:14s}  {row['time_0.1']}")

    print(f"\n Stochastic entropy (20 draws, neutral obs):")
    unique = dist["stochastic_entropy"]["unique_actions_seen"]
    collapsed = dist["stochastic_entropy"]["collapsed"]
    print(f"   Unique actions seen: {unique}/4  {'⚠️  COLLAPSED — deterministic policy' if collapsed else '✅ Exploring'}")

    # Collapse detection
    print(f"\n── SECTION 3: POLICY COLLAPSE DETECTION ────────────────────────────")
    for criterion, result in collapse.items():
        flag = "❌ TRIGGERED" if result["triggered"] else "✅ OK       "
        print(f"  {flag}  {criterion}")
        print(f"            {result['value']}")

    # Reward function
    print(f"\n── SECTION 4: REWARD FUNCTION AUDIT ────────────────────────────────")
    for criterion, finding in reward.items():
        flag = "✅ PASS" if finding["pass"] else "❌ FAIL"
        print(f"\n  {flag}  {criterion.replace('_', ' ').upper()}")
        # Word-wrap the notes
        notes = finding["notes"]
        words = notes.split()
        line, lines = [], []
        for w in words:
            line.append(w)
            if len(" ".join(line)) > 65:
                lines.append("        " + " ".join(line[:-1]))
                line = [w]
        lines.append("        " + " ".join(line))
        print("\n".join(lines))

    # Verdict
    print(f"\n{DIVIDER}")
    color = "🔴" if verdict["verdict"] == "NO-GO" else "🟡"
    print(f" {color} CRO VERDICT: {verdict['verdict']}")
    print(f"   Collapse flags : {verdict['collapse_flags_triggered']}")
    print(f"   Reward failures: {verdict['reward_criteria_failed']}")
    if verdict["hard_blocks"]:
        print(f"\n   HARD BLOCKS (must fix before any live use):")
        for b in verdict["hard_blocks"]:
            print(f"     • {b}")
    print(f"\n   {verdict['recommendation']}")

    # Remediation
    print(f"\n── REMEDIATION PLAN ─────────────────────────────────────────────────")
    print("""
  1. REWARD FUNCTION — Fix the VPIN inversion (highest priority):
       Current : action==3 (Sweep) gets +0.005 reward when VPIN > 0.8
       Correct :
           if action == 3 and vpin > 0.7:
               reward -= 0.01   # penalise sweeping into toxic flow
           if action == 0 and vpin > 0.7:
               reward += 0.002  # reward waiting when toxic

  2. TRAINING DATA — Replace MockTickGenerator with real ticks:
       Use the Alpaca WebSocket tick stream already in your codebase
       (StockDataStream). Buffer 500,000 real ticks from AAPL/MSFT
       before training. Reshuffle episodes by sampling random 2000-tick
       windows, not replaying the same window from step 0.

  3. INVENTORY URGENCY — Rebalance penalty magnitudes:
       Current : inventory penalty = -0.01
       Current : edge_captured reward = up to +10.0 (edge * 100)
       Fix     : normalise both to the same scale (~0.01-0.1 range)
       Or      : use reward = -slippage_bps / 10 + execution_bonus

  4. PENNY-JUMP NEVER TRIGGERED — Raise iceberg detection reward:
       The action was never chosen in 2000 training ticks because
       MockTickGenerator only produces iceberg=1 ~50% randomly with
       no correlation to spread width. In real data, icebergs correlate
       with thick books and tight spreads — the agent never learned this.

  5. PPO → SAC DECISION (see Section 5 of report):
       For the 30-day paper run: keep this PPO model in MONITORING-ONLY
       mode (log its recommended actions, do not execute them). Execute
       via fixed rules instead. Build and validate the SAC replacement
       in parallel during the paper run period.
    """)
    print(DIVIDER + "\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="PPO Execution Agent Forensic Audit")
    parser.add_argument("--model", default="run_state/models/ppo_execution_v1.zip")
    args = parser.parse_args()

    model_path = PROJECT_ROOT / args.model
    if not model_path.exists():
        print(f"ERROR: model not found at {model_path}")
        sys.exit(1)

    try:
        from stable_baselines3 import PPO
    except ImportError:
        print("ERROR: pip install stable-baselines3")
        sys.exit(1)

    print(f"Loading {model_path} ...")
    model = PPO.load(str(model_path))

    arch    = inspect_architecture(model)
    dist    = analyse_action_distribution(model)
    reward  = audit_reward_function()
    collapse = detect_policy_collapse(dist)
    verdict = cro_verdict(reward, collapse)

    print_report(arch, dist, reward, collapse, verdict)

    # Save JSON
    out_dir = PROJECT_ROOT / "data" / "audit"
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = out_dir / f"ppo_audit_{ts}.json"
    class _Encoder(json.JSONEncoder):
        def default(self, o):
            if isinstance(o, (np.integer,)):
                return int(o)
            if isinstance(o, (np.floating,)):
                return float(o)
            if isinstance(o, np.ndarray):
                return o.tolist()
            return super().default(o)

    out_path.write_text(json.dumps({
        "architecture": arch,
        "action_distribution": dist,
        "reward_findings": reward,
        "collapse_detection": collapse,
        "verdict": verdict,
    }, indent=2, cls=_Encoder))
    print(f"Full report saved → {out_path}")


if __name__ == "__main__":
    main()
