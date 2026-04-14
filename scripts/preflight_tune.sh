#!/bin/bash

# scripts/preflight_tune.sh - Monday Open Pre-Flight Sequence

set -e

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
export PYTHONPATH="$PROJECT_ROOT"

echo "🚀 STARTING MONDAY PRE-FLIGHT SEQUENCE..."

# 1. Run Adversarial Monte Carlo (50 epochs per pair)
echo "👾 Phase 1: Running Adversarial Stress Tests..."
python3 "$PROJECT_ROOT/scripts/run_adversarial_mc.py"

# 2. Run Parameter Tuner
echo "🔧 Phase 2: Recalibrating Strategy Z-Scores..."
python3 "$PROJECT_ROOT/quant_system/ml/parameter_tuner.py"

# 3. Sentiment Warden Sweep (09:00 AM ET - 30m before bell)
echo "👁️ Phase 3: AI Sentiment Warden Sweep for Landmines..."
python3 "$PROJECT_ROOT/scripts/sentiment_sweep.py"

# 4. Launch Global Harness v6
echo "💹 Phase 4: Launching Global Harness (v6) for Monday Open..."
python3 "$PROJECT_ROOT/scripts/run_global_harness_v3.py"
