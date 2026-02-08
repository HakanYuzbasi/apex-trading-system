#!/bin/bash
set -euo pipefail

BASE_DIR="$(cd "$(dirname "$0")/.." && pwd)"
LOG_DIR="$BASE_DIR/logs"
PID_DIR="/tmp"

TRADING_PID_FILE="$PID_DIR/apex_trading.pid"
API_PID_FILE="$PID_DIR/apex_api.pid"
FRONTEND_PID_FILE="$PID_DIR/apex_frontend.pid"

mkdir -p "$LOG_DIR"

usage() {
  echo "Usage: $0 {start|stop|restart|status}"
}

is_running() {
  local pid_file="$1"
  if [[ -f "$pid_file" ]]; then
    local pid
    pid="$(cat "$pid_file")"
    if kill -0 "$pid" 2>/dev/null; then
      return 0
    fi
  fi
  return 1
}

start_trading() {
  if is_running "$TRADING_PID_FILE"; then
    echo "Trading engine already running (PID $(cat "$TRADING_PID_FILE"))"
    return
  fi
  cd "$BASE_DIR"
  nohup venv/bin/python main.py > /private/tmp/apex_main.log 2>&1 &
  echo $! > "$TRADING_PID_FILE"
  sleep 1
  if ! is_running "$TRADING_PID_FILE"; then
    rm -f "$TRADING_PID_FILE"
    echo "Trading engine failed to start. Check /private/tmp/apex_main.log"
    return 1
  fi
  echo "Started trading engine (PID $(cat "$TRADING_PID_FILE"))"
}

start_api() {
  if is_running "$API_PID_FILE"; then
    echo "API already running (PID $(cat "$API_PID_FILE"))"
    return
  fi
  cd "$BASE_DIR"
  lsof -ti:8000 | xargs kill 2>/dev/null || true
  sleep 1
  nohup venv/bin/python -m uvicorn api.server:app --reload --port 8000 > "$LOG_DIR/api.log" 2>&1 &
  echo $! > "$API_PID_FILE"
  sleep 1
  if ! is_running "$API_PID_FILE"; then
    rm -f "$API_PID_FILE"
    echo "API failed to start. Check $LOG_DIR/api.log"
    return 1
  fi
  echo "Started API (PID $(cat "$API_PID_FILE"))"
}

start_frontend() {
  if is_running "$FRONTEND_PID_FILE"; then
    echo "Frontend already running (PID $(cat "$FRONTEND_PID_FILE"))"
    return
  fi
  cd "$BASE_DIR/frontend"
  lsof -ti:3000 | xargs kill 2>/dev/null || true
  lsof -ti:3001 | xargs kill 2>/dev/null || true
  rm -f "$BASE_DIR/frontend/.next/dev/lock" 2>/dev/null || true
  sleep 1
  nohup npm run dev > "$LOG_DIR/frontend.log" 2>&1 &
  echo $! > "$FRONTEND_PID_FILE"
  sleep 1
  if ! is_running "$FRONTEND_PID_FILE"; then
    rm -f "$FRONTEND_PID_FILE"
    echo "Frontend failed to start. Check $LOG_DIR/frontend.log"
    return 1
  fi
  echo "Started frontend (PID $(cat "$FRONTEND_PID_FILE"))"
}

stop_service() {
  local name="$1"
  local pid_file="$2"
  if is_running "$pid_file"; then
    local pid
    pid="$(cat "$pid_file")"
    kill "$pid" 2>/dev/null || true
    sleep 1
    if kill -0 "$pid" 2>/dev/null; then
      kill -9 "$pid" 2>/dev/null || true
    fi
    rm -f "$pid_file"
    echo "Stopped $name"
  else
    echo "$name not running"
  fi
}

status_service() {
  local name="$1"
  local pid_file="$2"
  if is_running "$pid_file"; then
    echo "$name: RUNNING (PID $(cat "$pid_file"))"
  else
    echo "$name: STOPPED"
  fi
}

case "${1:-}" in
  start)
    start_trading
    start_api
    start_frontend
    echo "All services started."
    ;;
  stop)
    stop_service "frontend" "$FRONTEND_PID_FILE"
    stop_service "api" "$API_PID_FILE"
    stop_service "trading engine" "$TRADING_PID_FILE"
    echo "All services stopped."
    ;;
  restart)
    "$0" stop
    "$0" start
    ;;
  status)
    status_service "trading engine" "$TRADING_PID_FILE"
    status_service "api" "$API_PID_FILE"
    status_service "frontend" "$FRONTEND_PID_FILE"
    ;;
  *)
    usage
    exit 1
    ;;
esac
