#!/bin/bash
#
# APEX Trading System - Installation Script
# Makes the system fully automated with auto-start on boot
#

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
SERVICE_NAME="apex-watchdog"
USER=$(whoami)

echo "═══════════════════════════════════════════════════════════"
echo "  APEX Trading System - Automation Installer"
echo "═══════════════════════════════════════════════════════════"
echo ""
echo "Project directory: $PROJECT_DIR"
echo "User: $USER"
echo ""

# Check if running as root
if [ "$EUID" -ne 0 ]; then
    echo "⚠️  This script needs sudo to install the systemd service."
    echo "   Re-running with sudo..."
    echo ""
    exec sudo "$0" "$@"
fi

# Create log directory
echo "📁 Creating directories..."
mkdir -p "$PROJECT_DIR/logs"
mkdir -p "$PROJECT_DIR/data"
chown -R $SUDO_USER:$SUDO_USER "$PROJECT_DIR/logs" 2>/dev/null || true
chown -R $SUDO_USER:$SUDO_USER "$PROJECT_DIR/data" 2>/dev/null || true

# Update service file with correct paths
echo "📝 Configuring service file..."
SERVICE_FILE="/etc/systemd/system/${SERVICE_NAME}.service"

cat > "$SERVICE_FILE" << EOF
[Unit]
Description=APEX Trading System Watchdog
After=network.target network-online.target
Wants=network-online.target

[Service]
Type=simple
User=${SUDO_USER:-$USER}
Group=${SUDO_USER:-$USER}
WorkingDirectory=$PROJECT_DIR

# Run the watchdog
ExecStart=/usr/bin/python3 -m automation.watchdog

# Restart policy
Restart=always
RestartSec=30

# Graceful shutdown
TimeoutStopSec=60

# Logging
StandardOutput=append:$PROJECT_DIR/logs/watchdog_stdout.log
StandardError=append:$PROJECT_DIR/logs/watchdog_stderr.log

[Install]
WantedBy=multi-user.target
EOF

echo "✅ Service file created: $SERVICE_FILE"

# Reload systemd
echo "🔄 Reloading systemd..."
systemctl daemon-reload

# Enable service
echo "🔧 Enabling service..."
systemctl enable "$SERVICE_NAME"

echo ""
echo "═══════════════════════════════════════════════════════════"
echo "  ✅ Installation Complete!"
echo "═══════════════════════════════════════════════════════════"
echo ""
echo "Commands:"
echo "  Start:   sudo systemctl start $SERVICE_NAME"
echo "  Stop:    sudo systemctl stop $SERVICE_NAME"
echo "  Status:  sudo systemctl status $SERVICE_NAME"
echo "  Logs:    sudo journalctl -u $SERVICE_NAME -f"
echo ""
echo "The watchdog will automatically:"
echo "  • Start on boot"
echo "  • Restart the trading system if it crashes"
echo "  • Start/stop trading based on market hours"
echo "  • Send alerts on critical events (if configured)"
echo ""
echo "To configure alerts, set these environment variables:"
echo "  APEX_SLACK_WEBHOOK=https://hooks.slack.com/..."
echo "  APEX_ALERT_EMAIL=your-email@example.com"
echo ""
