#!/bin/bash
# automated_retraining.sh
# Scheduled execution wrapper for APEX ML continuous learning pipeline.
#
# Recommended cron setup (Run every Saturday at 2:00 AM):
# 0 2 * * 6 /Users/hakanyuzbasioglu/apex-trading/scripts/automated_retraining.sh >> /Users/hakanyuzbasioglu/apex-trading/logs/automated_retraining.log 2>&1

set -euo pipefail

# Resolve base paths
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASE_DIR="$(dirname "$SCRIPT_DIR")"
VENV_DIR="$BASE_DIR/venv"
LOG_DIR="$BASE_DIR/logs"

# Ensure directories exist
mkdir -p "$LOG_DIR"

# Source environment
if [[ -f "$BASE_DIR/.env" ]]; then
    set -a
    source "$BASE_DIR/.env"
    set +a
fi

# Activate virtual environment
if [[ -d "$VENV_DIR" ]]; then
    source "$VENV_DIR/bin/activate"
else
    echo "ERROR: Virtual environment not found at $VENV_DIR"
    exit 1
fi

echo "================================================================="
echo "Starting Automated Retraining: $(date)"
echo "================================================================="

cd "$BASE_DIR"

# Execute training script with atomicity flag
# - Train on 1500 days of data for robust coverage
# - Force retraining even if metadata exists
# - Use a temporary staging directory and atomically swap it upon success
python scripts/train_production_models.py \
    --days 1500 \
    --force \
    --model-dir "models/saved_ultimate_staging" \
    --atomic-replace "models/saved_ultimate"

EXIT_CODE=$?

if [[ $EXIT_CODE -eq 0 ]]; then
    echo "✅ Automated Retraining completed successfully at $(date)."
else
    echo "❌ Automated Retraining failed with exit code $EXIT_CODE."
fi

echo "================================================================="
exit $EXIT_CODE
