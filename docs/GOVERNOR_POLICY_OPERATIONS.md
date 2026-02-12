# Governor Policy Operations

This document describes the APEX governor policy workflow:
- Policy scope: `asset_class + regime` (asset class first, regime refinement)
- Walk-forward tuning cadence: weekly by default, daily option for unstable crypto
- Promotion: auto in non-prod; manual approval required in production
- Hard kill-switch: drawdown and rolling Sharpe decay

## Key Files
- Active policies: `data/governor_policies/active_policies.json`
- Candidate policies: `data/governor_policies/candidate_policies.json`
- Tuning state: `data/governor_policies/tuning_state.json`
- Audit trail: `data/governor_policies/policy_audit_log.jsonl`

## Tune Policies
Run walk-forward tuning from signal outcome history:

```bash
venv/bin/python scripts/tune_governor.py --cadence weekly --environment staging
```

## Approve Staged Policy (Production)
When running in production mode, candidates are staged for manual sign-off:

```bash
venv/bin/python scripts/approve_governor_policy.py \
  --policy-id EQUITY:risk_off:wf-20260212233000 \
  --approver "risk-committee@company.com"
```

## Operational API (Admin Only)
- `GET /ops/governor/policies/active`  
  List active policies by `asset_class:regime`.
- `GET /ops/governor/policies/candidates?status=staged`  
  List staged/candidate/rejected policy snapshots.
- `POST /ops/governor/policies/approve`  
  Manually approve staged policy with reason:

```json
{
  "policy_id": "EQUITY:risk_off:wf-20260212233000",
  "reason": "Risk committee approval after paper validation"
}
```

- `POST /ops/governor/policies/rollback`  
  Roll back active policy to previous or specific version:

```json
{
  "asset_class": "EQUITY",
  "regime": "risk_off",
  "reason": "Live OOS drift exceeded threshold",
  "target_version": "wf-20260205120000"
}
```

- `GET /ops/governor/policies/audit?asset_class=EQUITY&regime=risk_off&limit=100`  
  Append-only audit events (`policy_audit_log.jsonl`) with chained hashes.

## Kill-Switch Trigger Rule
Kill-switch activates when configured logic is breached (`OR` by default):
- `current_drawdown > KILL_SWITCH_DD_MULTIPLIER * historical_mdd_baseline`
- `rolling_sharpe_63d < KILL_SWITCH_SHARPE_FLOOR`

When triggered:
- all positions are flattened once,
- new entries are blocked,
- metric `apex_risk_kill_switch_active` is set to `1`.
