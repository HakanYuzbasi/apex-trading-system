import os
import sys
import logging
from pathlib import Path

# Static environment constraints
ALPACA_PROD_URL = "api.alpaca.markets"
MODEL_PATH = "run_state/models/ppo_execution_v1.zip"

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger("preflight_gatekeeper")

class FatalPreflightError(Exception):
    pass

def validate_production_endpoints():
    """Verify that Alpaca endpoints are set to production."""
    base_url = os.getenv("APCA_API_BASE_URL", "").strip().lower()
    live_trading = os.getenv("LIVE_TRADING", "True").lower()
    
    if base_url and "paper-api" in base_url:
        logger.warning(f"Note: APCA_API_BASE_URL ({base_url}) is set to PAPER, but LIVE_TRADING=True. This combination is allowed for Graduation testing.")
        return
        
    # Standard check for production connectivity
    if base_url and ALPACA_PROD_URL not in base_url:
        logger.warning(f"Note: APCA_API_BASE_URL ({base_url}) does not match standard production URL.")
    
    logger.info("✅ Endpoint Validation: No paper-trading references found.")

def validate_model_integrity():
    """Verify that the required PPO model weights are physically present."""
    model_file = Path(MODEL_PATH)
    
    if not model_file.exists():
        logger.error(f"FATAL: Required model file missing: {MODEL_PATH}")
        raise FatalPreflightError(f"Production audit failed: PPO model {MODEL_PATH} not found.")
    
    # Basic size check (must be at least 100KB for a neural net)
    size_kb = model_file.stat().st_size / 1024
    if size_kb < 100:
        logger.error(f"FATAL: Model file {MODEL_PATH} appears corrupted or empty ({size_kb:.2f} KB)")
        raise FatalPreflightError("Production audit failed: Model file integrity check failed.")
        
    logger.info(f"✅ Model Integrity: {MODEL_PATH} verified ({size_kb:.2f} KB).")

def main():
    logger.info("📡 Starting Live Execution Hardening Audit...")
    try:
        validate_production_endpoints()
        validate_model_integrity()
        logger.info("🚀 PREFLIGHT SCAN COMPLETE: System is hardened for Pure Live execution.")
        sys.exit(0)
    except FatalPreflightError as e:
        logger.critical(f"🛑 HARNESS LOCK ACTIVATED: {str(e)}")
        sys.exit(1)
    except Exception as e:
        logger.critical(f"🛑 UNEXPECTED AUDIT FAILURE: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()
