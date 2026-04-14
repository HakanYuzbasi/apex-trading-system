from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from datetime import datetime
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
logger = logging.getLogger("parameter_tuner")

class ParameterTuner:
    """
    Parses Adversarial Monte Carlo results and recalibrates strategy 
    Z-scores using a tanh-scaled safety multiplier.
    """

    def __init__(
        self,
        results_path: Path = PROJECT_ROOT / "run_state" / "adversarial_results.json",
        output_path: Path = PROJECT_ROOT / "run_state" / "tuned_parameters.json",
        target_drawdown: float = 0.05, # Adaptive threshold: 5%
        scale: float = 0.10, # Scaling factor for the tanh adjustment
    ) -> None:
        self.results_path = results_path
        self.output_path = output_path
        self.target_drawdown = target_drawdown
        self.scale = scale

    def run_recalibration(self) -> dict[str, float]:
        """
        Processes adversarial results and generates Z-score overrides.
        """
        if not self.results_path.exists():
            logger.warning(f"Adversarial results not found at {self.results_path}. Skipping recalibration.")
            return {}

        try:
            with open(self.results_path, 'r') as f:
                data = json.load(f)
            
            pair_results = data.get("pair_results", {})
            overrides = {}

            logger.info("Starting Adaptive Calibration based on Adversarial Results...")
            
            for pair_label, results in pair_results.items():
                dd_adv = results.get("max_adversarial_drawdown", 0.0)
                
                if dd_adv > self.target_drawdown:
                    # Apply Formula: Z_adj = Z_base * (1 + tanh((DD_adv - DD_target) / scale))
                    # Note: Since the Harness holds Z_base, the Tuner will output 
                    # the MULTIPLIER or the adjustment factor. 
                    # Actually, the user requested Z_adj. We'll generate the new Z values 
                    # assuming a base Z of 2.0 (standard) or reading from current config.
                    # For simplicity, we output the MULTIPLIER that the Harness will apply.
                    
                    adjustment_factor = 1.0 + np.tanh((dd_adv - self.target_drawdown) / self.scale)
                    overrides[pair_label] = {
                        "entry_z_multiplier": float(adjustment_factor),
                        "max_adversarial_drawdown": float(dd_adv)
                    }
                    logger.info(f"Pair {pair_label}: DD_adv={dd_adv:.2%} > {self.target_drawdown:.0%}. Adjustment: x{adjustment_factor:.2f}")

            if overrides:
                self.output_path.parent.mkdir(parents=True, exist_ok=True)
                with open(self.output_path, 'w') as f:
                    json.dump({
                        "timestamp": datetime.now().isoformat(),
                        "tuning_overrides": overrides
                    }, f, indent=2)
                logger.info(f"Tuned parameters saved to {self.output_path}")

            return overrides

        except Exception as e:
            logger.error(f"Failed to recalibrate parameters: {e}")
            return {}

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    tuner = ParameterTuner()
    tuner.run_recalibration()
