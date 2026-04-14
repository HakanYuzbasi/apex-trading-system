#!/usr/bin/env python3

import os
import subprocess
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("tensorboard_daemon")

def start_tensorboard():
    log_dir = Path(__file__).resolve().parents[1] / "run_state" / "tensorboard"
    
    # Ensure directory exists to avoid crash
    log_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Starting TensorBoard server monitoring {log_dir} ...")
    logger.info(f"Access at http://localhost:6006")
    
    try:
        subprocess.run(["tensorboard", "--logdir", str(log_dir), "--host", "0.0.0.0", "--port", "6006"])
    except KeyboardInterrupt:
        logger.info("Shutting down TensorBoard Daemon.")

if __name__ == "__main__":
    start_tensorboard()
