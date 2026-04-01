"""
quant_system/governance/monitoring.py
================================================================================
Governance & Dynamic Circuit Breakers (EXTENDED AUDIT TRACE)
================================================================================
"""

import numpy as np
import pandas as pd
import json
import hashlib
import logging
from typing import Dict, List, Any
from scipy.stats import spearmanr, entropy
from datetime import datetime

logger = logging.getLogger(__name__)

class GovernanceMonitor:
    def __init__(self):
        self.history_ic = []
        self.history_sharpe = []
        self.audit_trace = []
        self.model_hashes = {}

    def hash_model_state(self, model_name: str, config: Dict) -> str:
        cfg_str = json.dumps(config, sort_keys=True)
        h = hashlib.sha256(f"{model_name}{cfg_str}".encode()).hexdigest()[:16]
        self.model_hashes[model_name] = {'hash': h, 'ts': datetime.now().isoformat()}
        return h

    def log_decision_trace(self, state: List[float], 
                           action_dict: dict,
                           expected_ret: float, realized_ret: float, 
                           regime: int, 
                           expected_impact: float, 
                           penalty_applied: float, recent_cost_spike: float,
                           cost_breakdown: Dict):
        """UPGRADE 6: Precision Tracing of Micro-Filters and Conviction Booleans."""
        self.audit_trace.append({
            'ts': datetime.now().isoformat(),
            'state_shape': len(state), 
            'action_executed': round(action_dict['action'], 4),
            'action_after_inertia': round(action_dict.get('action_after_inertia', 0.0), 4),
            'action_raw': round(action_dict['raw_action'], 4),
            'filtered_micro_trade': action_dict.get('filtered_micro_trade', False),
            'action_before_threshold': round(action_dict.get('action_before_threshold', 0.0), 4),
            'action_after_threshold': round(action_dict.get('action_after_threshold', 0.0), 4),
            'conviction_floor_applied': action_dict.get('conviction_floor_applied', False),
            'global_scaling_applied': round(action_dict.get('global_scaling_applied', 1.0), 2),
            'confidence': round(action_dict['confidence'], 4),
            'adjusted_epsilon': round(action_dict['epsilon'], 4),
            'edge_strength': round(action_dict['edge_strength'], 4),
            'boosted_action_flag': action_dict.get('boosted_action_flag', False),
            'max_action_cap_applied': round(action_dict['max_action'], 4),
            'blocked_by_no_trade_zone': action_dict['blocked_by_ntz'],
            'regime': regime,
            'expected_future_impact': round(expected_impact, 4),
            'smooth_penalty_applied': round(penalty_applied, 4),
            'recent_cost_spike_mem': round(recent_cost_spike, 4),
            'expected_return': round(expected_ret, 4),
            'realized_return': round(realized_ret, 4),
            'cost_breakdown': cost_breakdown
        })

    def export_audit_logs(self, directory: str = "logs/"):
        import os
        os.makedirs(directory, exist_ok=True)
        with open(os.path.join(directory, 'decision_trace.json'), 'w') as f:
            json.dump(self.audit_trace[-1000:], f, indent=2)
        pd.DataFrame(self.model_hashes).T.to_csv(os.path.join(directory, 'institutional_audit.csv'))

    def detect_kl_divergence(self, train_dist: np.ndarray, live_dist: np.ndarray, bins: int = 50) -> float:
        hist_train, bin_edges = np.histogram(train_dist, bins=bins, density=True)
        hist_live, _ = np.histogram(live_dist, bins=bin_edges, density=True)
        hist_train = np.clip(hist_train, 1e-8, None)
        hist_live = np.clip(hist_live, 1e-8, None)
        return float(entropy(hist_train, hist_live))

    def evaluate_live_governance_tick(
        self,
        recent_meta_probs: np.ndarray,
        recent_realized_returns: np.ndarray,
        current_drawdown: float,
        current_volatility: float,
        slippage_spike: bool,
        train_feat_dist: np.ndarray = None,
        live_feat_dist: np.ndarray = None
    ) -> Dict[str, bool]:
        retrain_req = False
        halt_req = False
        
        if len(recent_meta_probs) > 30 and len(recent_realized_returns) > 30:
            ic, _ = spearmanr(recent_meta_probs, recent_realized_returns)
            self.history_ic.append(ic)
        
        if len(self.history_ic) > 20:
            rolling_ic = np.array(self.history_ic[-20:])
            z_ic = (self.history_ic[-1] - np.mean(rolling_ic[:-1])) / (np.std(rolling_ic[:-1]) + 1e-8)
            
            if z_ic < -2.0:
                retrain_req = True
                
        if train_feat_dist is not None and live_feat_dist is not None:
            if self.detect_kl_divergence(train_feat_dist, live_feat_dist) > 1.5:
                retrain_req = True

        if current_drawdown < -0.15 and current_volatility > 0.05 and slippage_spike:
            halt_req = True
            
        return {'retrain_required': retrain_req, 'emergency_halt': halt_req}
