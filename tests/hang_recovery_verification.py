
import os
import time
import signal
import subprocess
import json
from pathlib import Path

def test_hang_recovery():
    print("🚀 Starting APEX Hang Recovery Test...")
    
    # 1. Create a dummy heartbeat file if needed by the automation watchdog
    heartbeat_file = Path("heartbeat.json")
    if heartbeat_file.exists():
        heartbeat_file.unlink()

    # 2. Start the harness in a subprocess
    # We'll use a mocked version or just run it with a very short timeout
    print("📦 Starting Global Harness...")
    process = subprocess.Popen(
        ["python3", "scripts/run_global_harness_v3.py"],
        env={**os.environ, "WATCHDOG_HARD_EXIT_TIMEOUT": "10", "HEALTH_CHECK_INTERVAL": "2"},
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE
    )

    try:
        print("⏳ Waiting for system to initialize...")
        time.sleep(5)
        
        # 3. Simulate a hang by freezing the process (SIGSTOP)
        print("🥶 Simulating system hang (SIGSTOP)...")
        process.send_signal(signal.SIGSTOP)
        
        start_time = time.time()
        timeout = 20 # Should exit within 10s + buffer
        
        while time.time() - start_time < timeout:
            if process.poll() is not None:
                print(f"✅ System exited with code {process.returncode} after hang detection.")
                return True
            time.sleep(1)
            
        print("❌ Test FAILED: System did not perform hard-exit within timeout.")
        process.kill()
        return False
        
    finally:
        if process.poll() is None:
            process.kill()

if __name__ == "__main__":
    success = test_hang_recovery()
    if success:
        print("🎉 Hang recovery stabilization VERIFIED.")
    else:
        print("🚨 Hang recovery stabilization FAILED.")
        exit(1)
