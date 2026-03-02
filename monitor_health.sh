#!/bin/bash
# Real-time Health Monitor for Apex Trading System

LOG_FILE="/private/tmp/apex_main.log"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "🏥 APEX HEALTH MONITOR - Real-Time Trading Status"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo

# Check if processes are running
check_process() {
    if [ -f ".run/$1.pid" ]; then
        PID=$(cat ".run/$1.pid" 2>/dev/null)
        if ps -p "$PID" > /dev/null 2>&1; then
            echo -e "${GREEN}✅ $2 running (PID $PID)${NC}"
            return 0
        fi
    fi
    echo -e "${RED}❌ $2 NOT running${NC}"
    return 1
}

echo "📊 System Status:"
check_process "apex_trading" "Trading Engine"
check_process "apex_api" "API Server"
check_process "apex_frontend" "Frontend"
echo

# Check broker connections
echo "🔌 Broker Health:"
if [ -f "data/users/admin/trading_state.json" ]; then
    IBKR_HEALTHY=$(jq -r '.broker_heartbeats.ibkr.healthy' data/users/admin/trading_state.json 2>/dev/null)
    ALPACA_HEALTHY=$(jq -r '.broker_heartbeats.alpaca.healthy' data/users/admin/trading_state.json 2>/dev/null)
    CAPITAL=$(jq -r '.capital' data/users/admin/trading_state.json 2>/dev/null)
    DAILY_PNL=$(jq -r '.daily_pnl' data/users/admin/trading_state.json 2>/dev/null)
    POSITIONS=$(jq -r '.open_positions' data/users/admin/trading_state.json 2>/dev/null)
    TRADES=$(jq -r '.total_trades' data/users/admin/trading_state.json 2>/dev/null)

    if [ "$IBKR_HEALTHY" = "true" ]; then
        echo -e "${GREEN}✅ IBKR Connected${NC}"
    else
        echo -e "${RED}❌ IBKR Disconnected${NC}"
    fi

    if [ "$ALPACA_HEALTHY" = "true" ]; then
        echo -e "${GREEN}✅ Alpaca Connected${NC}"
    else
        echo -e "${RED}❌ Alpaca Disconnected${NC}"
    fi
    echo

    echo "💰 Portfolio:"
    echo -e "   Capital: ${BLUE}\$$CAPITAL${NC}"
    echo -e "   Daily P&L: ${BLUE}\$$DAILY_PNL${NC}"
    echo -e "   Positions: ${BLUE}$POSITIONS${NC}"
    echo -e "   Total Trades: ${BLUE}$TRADES${NC}"
    echo
fi

# Check recent errors
echo "⚠️  Recent Issues (last 5 min):"
if [ -f "$LOG_FILE" ]; then
    ERRORS=$(tail -n 1000 "$LOG_FILE" | grep -E "ERROR|CRITICAL|blocked by equity" | tail -n 5)
    if [ -z "$ERRORS" ]; then
        echo -e "${GREEN}   No recent errors${NC}"
    else
        echo -e "${YELLOW}$ERRORS${NC}"
    fi
else
    echo "   Log file not found"
fi
echo

# Check recent signals
echo "📈 Recent Signals (last 10):"
if [ -f "$LOG_FILE" ]; then
    SIGNALS=$(tail -n 2000 "$LOG_FILE" | grep -E "📊.*signal=" | tail -n 10)
    if [ -z "$SIGNALS" ]; then
        echo -e "${YELLOW}   No signals generated yet${NC}"
    else
        echo "$SIGNALS" | while read line; do
            if echo "$line" | grep -q "BULLISH"; then
                echo -e "${GREEN}   $line${NC}"
            elif echo "$line" | grep -q "BEARISH"; then
                echo -e "${RED}   $line${NC}"
            else
                echo "   $line"
            fi
        done
    fi
else
    echo "   Log file not found"
fi
echo

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "💡 Commands:"
echo "   Monitor live: tail -f $LOG_FILE | grep -E '(signal=|ENTRY|EXIT|Portfolio:)'"
echo "   Restart: ./apex_ctl.sh restart"
echo "   Stop: ./apex_ctl.sh stop"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
