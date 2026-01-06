#!/bin/bash
# APEX Trading System - Setup Script

echo "════════════════════════════════════════════════════════"
echo "  APEX TRADING SYSTEM - Setup"
echo "════════════════════════════════════════════════════════"

# Create venv
python3 -m venv venv
source venv/bin/activate

# Install
pip install --upgrade pip
pip install -r requirements.txt

# Config
cp .env.example .env

echo "✅ Setup complete!"
echo ""
echo "Run: python main.py"
