import os
import time
import logging
import requests
from typing import Optional

# Setup standalone logging to avoid interference with main log files during panic
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger("dead_man")

# Configuration from Environment Variables
API_KEY = os.environ.get("ALPACA_API_KEY")
API_SECRET = os.environ.get("ALPACA_SECRET_KEY")
BASE_URL = "https://paper-api.alpaca.markets" # Assuming paper for safety, production would be api.alpaca.markets
TELEGRAM_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID")

class DeadMansSwitch:
    """
    Independent watcher that liquidates all positions if drawdown threshold is breached.
    Bypasses internal trading libraries to ensure operational integrity during system crashes.
    """
    def __init__(self, drawdown_threshold: float = 0.05):
        self.drawdown_threshold = drawdown_threshold
        self.daily_high_equity: float = 0.0
        self.headers = {
            "APCA-API-KEY-ID": API_KEY,
            "APCA-API-SECRET-KEY": API_SECRET
        }

    def _send_telegram(self, message: str):
        if not TELEGRAM_TOKEN or not TELEGRAM_CHAT_ID:
            logger.warning("Telegram configuration missing. Alert skipped.")
            return

        url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
        payload = {"chat_id": TELEGRAM_CHAT_ID, "text": message}
        try:
            requests.post(url, json=payload, timeout=5)
        except Exception as e:
            logger.error(f"Failed to send Telegram alert: {e}")

    def _liquidate_all(self):
        """Cancels all orders and closes all positions."""
        logger.critical("☢️ KILL SWITCH ENGAGED: Liquidating everything...")
        
        # 1. Cancel all open orders
        try:
            requests.delete(f"{BASE_URL}/v2/orders", headers=self.headers, timeout=10)
            logger.info("  - All orders cancelled.")
        except Exception as e:
            logger.error(f"Failed to cancel orders: {e}")

        # 2. Close all positions
        try:
            requests.delete(f"{BASE_URL}/v2/positions", headers=self.headers, timeout=10)
            logger.info("  - All positions closed.")
        except Exception as e:
            logger.error(f"Failed to close positions: {e}")

        self._send_telegram("☢️ KILL SWITCH ENGAGED. 5% Drawdown Broken. All positions closed.")

    def run_loop(self, interval: int = 30):
        if not API_KEY or not API_SECRET:
            logger.error("Alpaca API keys missing. Exit.")
            return

        logger.info("Dead Man's Switch active. Monitoring account value...")
        
        while True:
            try:
                # Fetch account value
                response = requests.get(f"{BASE_URL}/v2/account", headers=self.headers, timeout=5)
                response.raise_for_status()
                data = response.json()
                
                current_equity = float(data["equity"])
                
                # Update daily peak
                if current_equity > self.daily_high_equity:
                    self.daily_high_equity = current_equity
                    logger.info(f"New daily high equity: ${self.daily_high_equity:.2f}")

                # Check drawdown
                if self.daily_high_equity > 0:
                    drawdown = (self.daily_high_equity - current_equity) / self.daily_high_equity
                    if drawdown >= self.drawdown_threshold:
                        logger.warning(f"Drawdown breach detected: {drawdown*100:.2f}%")
                        self._liquidate_all()
                        break # Stop loop after liquidation

                logger.debug(f"Equity: ${current_equity:.2f} | Max: ${self.daily_high_equity:.2f} | DD: {(self.daily_high_equity - current_equity) / self.daily_high_equity * 100 if self.daily_high_equity > 0 else 0:.2f}%")

            except Exception as e:
                logger.error(f"Error in dead man's loop: {e}")

            time.sleep(interval)

if __name__ == "__main__":
    switch = DeadMansSwitch(drawdown_threshold=0.05)
    switch.run_loop()
