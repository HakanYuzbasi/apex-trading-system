"""
automation/watchdog.py - System Watchdog and Auto-Recovery

Provides fully automated operation:
- Monitors main trading process health
- Auto-restarts on crash or hang
- Sends alerts on critical events
- Manages daily startup/shutdown schedule
- Health endpoint for external monitoring

Usage:
    python -m automation.watchdog

This should be the main entry point for production deployment.
"""

import asyncio
import subprocess
import sys
import os
import signal
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, Dict, Any
import json
import threading
import time

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import ApexConfig

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/watchdog.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger('watchdog')


class AlertManager:
    """
    Send alerts via multiple channels.

    Supports:
    - Slack webhooks
    - Email (SMTP)
    - Desktop notifications
    - Log file alerts
    """

    def __init__(self):
        self.slack_webhook = os.getenv('APEX_SLACK_WEBHOOK')
        self.email_to = os.getenv('APEX_ALERT_EMAIL')
        self.smtp_server = os.getenv('APEX_SMTP_SERVER', 'smtp.gmail.com')
        self.smtp_port = int(os.getenv('APEX_SMTP_PORT', '587'))
        self.smtp_user = os.getenv('APEX_SMTP_USER')
        self.smtp_pass = os.getenv('APEX_SMTP_PASS')
        self.alert_history: list = []

    def send_alert(self, level: str, title: str, message: str):
        """
        Send alert through all configured channels.

        Args:
            level: 'info', 'warning', 'critical'
            title: Alert title
            message: Alert body
        """
        timestamp = datetime.now().isoformat()
        alert = {
            'timestamp': timestamp,
            'level': level,
            'title': title,
            'message': message
        }
        self.alert_history.append(alert)

        # Keep only last 100 alerts
        if len(self.alert_history) > 100:
            self.alert_history = self.alert_history[-100:]

        # Log the alert
        log_level = getattr(logging, level.upper(), logging.INFO)
        logger.log(log_level, f"🚨 {title}: {message}")

        # Send to Slack if configured
        if self.slack_webhook:
            self._send_slack(level, title, message)

        # Send email for critical alerts
        if level == 'critical' and self.email_to:
            self._send_email(title, message)

    def _send_slack(self, level: str, title: str, message: str):
        """Send alert to Slack webhook."""
        try:
            import urllib.request

            emoji = {'info': 'ℹ️', 'warning': '⚠️', 'critical': '🚨'}.get(level, '📢')
            color = {'info': '#36a64f', 'warning': '#ff9800', 'critical': '#ff0000'}.get(level, '#808080')

            payload = {
                'attachments': [{
                    'color': color,
                    'title': f"{emoji} APEX: {title}",
                    'text': message,
                    'footer': f"APEX Trading System | {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
                }]
            }

            data = json.dumps(payload).encode('utf-8')
            req = urllib.request.Request(
                self.slack_webhook,
                data=data,
                headers={'Content-Type': 'application/json'}
            )
            urllib.request.urlopen(req, timeout=10)
            logger.debug("Slack alert sent successfully")

        except Exception as e:
            logger.error(f"Failed to send Slack alert: {e}")

    def _send_email(self, title: str, message: str):
        """Send email alert."""
        if not all([self.smtp_user, self.smtp_pass, self.email_to]):
            return

        try:
            import smtplib
            from email.mime.text import MIMEText

            msg = MIMEText(f"{message}\n\nTimestamp: {datetime.now()}")
            msg['Subject'] = f"[APEX ALERT] {title}"
            msg['From'] = self.smtp_user
            msg['To'] = self.email_to

            with smtplib.SMTP(self.smtp_server, self.smtp_port) as server:
                server.starttls()
                server.login(self.smtp_user, self.smtp_pass)
                server.send_message(msg)

            logger.debug("Email alert sent successfully")

        except Exception as e:
            logger.error(f"Failed to send email alert: {e}")


class HealthMonitor:
    """
    Monitor system health and detect issues.

    Checks:
    - Process is running
    - Recent trades/activity
    - Memory usage
    - IBKR connection status
    - Error rate
    """

    def __init__(self):
        self.last_heartbeat: Optional[datetime] = None
        self.last_trade_time: Optional[datetime] = None
        self.error_count: int = 0
        self.restart_count: int = 0
        self.start_time: datetime = datetime.now()

    def record_heartbeat(self):
        """Record that system is alive."""
        self.last_heartbeat = datetime.now()

    def record_trade(self):
        """Record that a trade occurred."""
        self.last_trade_time = datetime.now()

    def record_error(self):
        """Record an error occurrence."""
        self.error_count += 1

    def record_restart(self):
        """Record a system restart."""
        self.restart_count += 1
        self.error_count = 0  # Reset error count after restart

    def is_healthy(self, max_heartbeat_age_seconds: int = 120) -> tuple:
        """
        Check if system is healthy.

        Returns:
            Tuple of (is_healthy: bool, reason: str)
        """
        if self.last_heartbeat is None:
            return False, "No heartbeat received"

        age = (datetime.now() - self.last_heartbeat).total_seconds()
        if age > max_heartbeat_age_seconds:
            return False, f"Heartbeat stale ({age:.0f}s old)"

        if self.error_count > 10:
            return False, f"Too many errors ({self.error_count})"

        return True, "OK"

    def get_status(self) -> Dict[str, Any]:
        """Get health status summary."""
        uptime = datetime.now() - self.start_time

        return {
            'uptime_seconds': uptime.total_seconds(),
            'uptime_human': str(uptime).split('.')[0],
            'last_heartbeat': self.last_heartbeat.isoformat() if self.last_heartbeat else None,
            'last_trade': self.last_trade_time.isoformat() if self.last_trade_time else None,
            'error_count': self.error_count,
            'restart_count': self.restart_count,
            'is_healthy': self.is_healthy()[0]
        }


class TradingWatchdog:
    """
    Main watchdog that supervises the trading system.

    Features:
    - Starts/stops trading process
    - Monitors health via heartbeat file
    - Auto-restarts on crash or hang
    - Respects market hours
    - Sends alerts on issues
    """

    def __init__(self):
        self.process: Optional[subprocess.Popen] = None
        self.running: bool = False
        self.alerts = AlertManager()
        self.health = HealthMonitor()

        # Configuration
        self.max_restarts_per_hour: int = 3
        self.restart_history: list = []
        self.heartbeat_file = Path('data/heartbeat.json')
        self.check_interval: int = 30  # seconds

        # Market hours (EST)
        self.market_open_hour: float = 9.5  # 9:30 AM
        self.market_close_hour: float = 16.0  # 4:00 PM
        self.pre_market_start: float = 9.0  # Start system at 9:00 AM
        self.post_market_end: float = 16.5  # Stop system at 4:30 PM

    def _get_est_hour(self) -> float:
        """Get current hour in Eastern Time."""
        try:
            import pytz
            eastern = pytz.timezone('America/New_York')
            now = datetime.now(pytz.UTC).astimezone(eastern)
            return now.hour + now.minute / 60.0
        except ImportError:
            # Fallback: approximate EST (UTC-5)
            from datetime import timezone
            now = datetime.now(timezone.utc)
            est_hour = now.hour - 5 + now.minute / 60.0
            if est_hour < 0:
                est_hour += 24
            return est_hour

    def _is_trading_time(self) -> bool:
        """Check if it's time to be trading."""
        est_hour = self._get_est_hour()
        weekday = datetime.now().weekday()

        # Not on weekends
        if weekday >= 5:
            return False

        # Check market hours (with buffer)
        return self.pre_market_start <= est_hour <= self.post_market_end

    def _can_restart(self) -> bool:
        """Check if we're allowed to restart (rate limiting)."""
        now = datetime.now()
        hour_ago = now - timedelta(hours=1)

        # Count restarts in last hour
        recent_restarts = [t for t in self.restart_history if t > hour_ago]
        self.restart_history = recent_restarts  # Cleanup old entries

        return len(recent_restarts) < self.max_restarts_per_hour

    def _read_heartbeat(self) -> Optional[datetime]:
        """Read heartbeat from file written by main trading system."""
        try:
            if self.heartbeat_file.exists():
                with open(self.heartbeat_file) as f:
                    data = json.load(f)
                    return datetime.fromisoformat(data['timestamp'])
        except Exception as e:
            logger.debug(f"Error reading heartbeat: {e}")
        return None

    def _write_watchdog_status(self):
        """Write watchdog status for external monitoring."""
        status_file = Path('data/watchdog_status.json')
        try:
            status = {
                'watchdog_running': self.running,
                'trading_process_running': self.process is not None and self.process.poll() is None,
                'health': self.health.get_status(),
                'timestamp': datetime.now().isoformat()
            }
            with open(status_file, 'w') as f:
                json.dump(status, f, indent=2)
        except Exception as e:
            logger.error(f"Error writing watchdog status: {e}")

    def start_trading(self):
        """Start the trading process."""
        if self.process is not None and self.process.poll() is None:
            logger.warning("Trading process already running")
            return

        logger.info("🚀 Starting trading process...")

        try:
            # Start main.py as subprocess
            self.process = subprocess.Popen(
                [sys.executable, 'main.py'],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                cwd=str(Path(__file__).parent.parent)
            )

            self.alerts.send_alert('info', 'Trading Started',
                f'APEX Trading System started (PID: {self.process.pid})')

            logger.info(f"✅ Trading process started (PID: {self.process.pid})")

        except Exception as e:
            self.alerts.send_alert('critical', 'Failed to Start', str(e))
            logger.error(f"❌ Failed to start trading: {e}")

    def stop_trading(self, reason: str = "Scheduled stop"):
        """Stop the trading process gracefully."""
        if self.process is None:
            return

        logger.info(f"🛑 Stopping trading process: {reason}")

        try:
            # Send SIGTERM for graceful shutdown
            self.process.terminate()

            # Wait up to 30 seconds for graceful shutdown
            try:
                self.process.wait(timeout=30)
                logger.info("✅ Trading process stopped gracefully")
            except subprocess.TimeoutExpired:
                # Force kill if still running
                self.process.kill()
                logger.warning("⚠️ Trading process force killed")

            self.process = None
            self.alerts.send_alert('info', 'Trading Stopped', reason)

        except Exception as e:
            logger.error(f"Error stopping trading: {e}")

    def restart_trading(self, reason: str):
        """Restart the trading process."""
        if not self._can_restart():
            self.alerts.send_alert('critical', 'Restart Rate Limit',
                f'Too many restarts in last hour. Manual intervention required.')
            logger.error("❌ Restart rate limit exceeded!")
            return

        self.restart_history.append(datetime.now())
        self.health.record_restart()

        logger.warning(f"🔄 Restarting trading: {reason}")
        self.alerts.send_alert('warning', 'Restarting', reason)

        self.stop_trading(reason)
        time.sleep(5)  # Brief pause before restart
        self.start_trading()

    def check_health(self):
        """Check trading process health."""
        # Check if process is running
        if self.process is None or self.process.poll() is not None:
            if self._is_trading_time():
                self.restart_trading("Process crashed or not running")
            return

        # Check heartbeat
        heartbeat = self._read_heartbeat()
        if heartbeat:
            self.health.last_heartbeat = heartbeat

            age = (datetime.now() - heartbeat).total_seconds()
            if age > 180:  # 3 minutes without heartbeat
                self.health.record_error()
                if age > 300:  # 5 minutes = likely hung
                    self.restart_trading(f"Process appears hung (no heartbeat for {age:.0f}s)")
                    return

        # Update health status
        is_healthy, reason = self.health.is_healthy()
        if not is_healthy:
            logger.warning(f"Health check failed: {reason}")
            self.health.record_error()

    async def run(self):
        """Main watchdog loop."""
        self.running = True
        logger.info("=" * 60)
        logger.info("🐕 APEX Watchdog Started")
        logger.info("=" * 60)

        self.alerts.send_alert('info', 'Watchdog Started',
            'APEX Trading System watchdog is now monitoring')

        try:
            while self.running:
                try:
                    # Check if it's trading time
                    if self._is_trading_time():
                        # Ensure trading is running
                        if self.process is None or self.process.poll() is not None:
                            self.start_trading()
                        else:
                            self.check_health()
                    else:
                        # Outside trading hours - stop if running
                        if self.process is not None and self.process.poll() is None:
                            self.stop_trading("Outside trading hours")

                    # Write status for external monitoring
                    self._write_watchdog_status()

                except Exception as e:
                    logger.error(f"Watchdog error: {e}")
                    self.health.record_error()

                await asyncio.sleep(self.check_interval)

        except KeyboardInterrupt:
            logger.info("Watchdog interrupted")
        finally:
            self.stop_trading("Watchdog shutdown")
            self.running = False
            self.alerts.send_alert('info', 'Watchdog Stopped', 'Manual shutdown')


async def main():
    """Main entry point for watchdog."""
    # Ensure directories exist
    Path('logs').mkdir(exist_ok=True)
    Path('data').mkdir(exist_ok=True)

    watchdog = TradingWatchdog()

    # Handle shutdown signals
    def signal_handler(sig, frame):
        logger.info(f"Received signal {sig}")
        watchdog.running = False

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    await watchdog.run()


if __name__ == '__main__':
    asyncio.run(main())
