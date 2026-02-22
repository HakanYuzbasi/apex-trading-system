#!/bin/bash
set -euo pipefail

BASE_DIR="$(cd "$(dirname "$0")/.." && pwd)"
LOG_DIR="$BASE_DIR/logs"
PID_DIR="$BASE_DIR/.run"
MAIN_LOG_FILE="/private/tmp/apex_main.log"
ENV_FILE="$BASE_DIR/.env"

TRADING_PID_FILE="$PID_DIR/apex_trading.pid"
API_PID_FILE="$PID_DIR/apex_api.pid"
FRONTEND_PID_FILE="$PID_DIR/apex_frontend.pid"

mkdir -p "$LOG_DIR" "$PID_DIR"

usage() {
  echo "Usage: $0 {start|stop|restart|status|logs}"
}

load_env() {
  if [[ -f "$ENV_FILE" ]]; then
    set -a
    # shellcheck disable=SC1090
    source "$ENV_FILE"
    set +a
  fi
}

ensure_prereqs() {
  if [[ ! -x "$BASE_DIR/venv/bin/python" ]]; then
    echo "Missing interpreter: $BASE_DIR/venv/bin/python"
    return 1
  fi
  if ! command -v npm >/dev/null 2>&1; then
    echo "Missing npm on PATH"
    return 1
  fi
}

read_pid() {
  local pid_file="$1"
  if [[ ! -f "$pid_file" ]]; then
    return 1
  fi
  local pid
  pid="$(cat "$pid_file" 2>/dev/null || true)"
  if [[ "$pid" =~ ^[0-9]+$ ]]; then
    echo "$pid"
    return 0
  fi
  return 1
}

pid_alive() {
  local pid="$1"
  kill -0 "$pid" 2>/dev/null
}

is_running() {
  local pid_file="$1"
  local pid
  if pid="$(read_pid "$pid_file")" && pid_alive "$pid"; then
    return 0
  fi
  return 1
}

cleanup_stale_pid_file() {
  local pid_file="$1"
  if [[ -f "$pid_file" ]] && ! is_running "$pid_file"; then
    rm -f "$pid_file"
  fi
}

kill_descendants() {
  local parent_pid="$1"
  local children
  children="$(pgrep -P "$parent_pid" 2>/dev/null || true)"
  if [[ -n "$children" ]]; then
    local child
    for child in $children; do
      kill_descendants "$child"
      kill "$child" 2>/dev/null || true
    done
  fi
}

check_live_trading_guard() {
  local live="${APEX_LIVE_TRADING:-false}"
  local confirmed="${APEX_LIVE_TRADING_CONFIRMED:-false}"
  if [[ "$live" == "true" && "$confirmed" != "true" ]]; then
    echo "Refusing to start: APEX_LIVE_TRADING=true but APEX_LIVE_TRADING_CONFIRMED!=true"
    echo "Set APEX_LIVE_TRADING_CONFIRMED=true only if you explicitly want live mode."
    return 1
  fi
}

free_port() {
  local port="$1"
  local pids
  pids="$(lsof -ti:"$port" 2>/dev/null || true)"
  if [[ -n "$pids" ]]; then
    local pid
    for pid in $pids; do
      kill "$pid" 2>/dev/null || true
    done
  fi
}

wait_for_boot() {
  local name="$1"
  local pid_file="$2"
  local log_file="$3"
  local attempts="${4:-8}"
  local i
  for ((i = 1; i <= attempts; i++)); do
    if is_running "$pid_file"; then
      sleep 1
      continue
    fi
    break
  done
  if ! is_running "$pid_file"; then
    cleanup_stale_pid_file "$pid_file"
    echo "$name failed to start. Check $log_file"
    if [[ -f "$log_file" ]]; then
      local bind_error=1
      if command -v rg >/dev/null 2>&1; then
        rg -q "operation not permitted|EPERM|address already in use|bind" "$log_file" 2>/dev/null || bind_error=$?
      else
        grep -Eqi "operation not permitted|EPERM|address already in use|bind" "$log_file" 2>/dev/null || bind_error=$?
      fi
      if [[ "$bind_error" -eq 0 ]]; then
        echo "Hint: startup blocked by port bind permission/conflict; see $log_file"
      fi
    fi
    return 1
  fi
  return 0
}

start_trading() {
  cleanup_stale_pid_file "$TRADING_PID_FILE"
  if is_running "$TRADING_PID_FILE"; then
    echo "Trading engine already running (PID $(read_pid "$TRADING_PID_FILE"))"
    return
  fi
  check_live_trading_guard
  cd "$BASE_DIR"
  "$BASE_DIR/venv/bin/python" main.py > "$MAIN_LOG_FILE" 2>&1 &
  echo $! > "$TRADING_PID_FILE"
  if ! wait_for_boot "Trading engine" "$TRADING_PID_FILE" "$MAIN_LOG_FILE" 4; then
    return 1
  fi
  echo "Started trading engine (PID $(read_pid "$TRADING_PID_FILE"))"
}

start_api() {
  cleanup_stale_pid_file "$API_PID_FILE"
  if is_running "$API_PID_FILE"; then
    echo "API already running (PID $(read_pid "$API_PID_FILE"))"
    return
  fi
  cd "$BASE_DIR"
  free_port 8000
  sleep 1
  local api_reload="${APEX_CTL_API_RELOAD:-false}"
  local api_args=(api.server:app --port 8000)
  if [[ "$api_reload" == "true" ]]; then
    api_args+=(--reload)
  fi
  "$BASE_DIR/venv/bin/python" -m uvicorn "${api_args[@]}" > "$LOG_DIR/api.log" 2>&1 &
  echo $! > "$API_PID_FILE"
  if ! wait_for_boot "API" "$API_PID_FILE" "$LOG_DIR/api.log" 6; then
    return 1
  fi
  echo "Started API (PID $(read_pid "$API_PID_FILE"))"
}

start_frontend() {
  cleanup_stale_pid_file "$FRONTEND_PID_FILE"
  if is_running "$FRONTEND_PID_FILE"; then
    echo "Frontend already running (PID $(read_pid "$FRONTEND_PID_FILE"))"
    return
  fi
  cd "$BASE_DIR/frontend"
  free_port 3000
  free_port 3001
  rm -f "$BASE_DIR/frontend/.next/dev/lock" 2>/dev/null || true
  sleep 1
  npm run dev > "$LOG_DIR/frontend.log" 2>&1 &
  echo $! > "$FRONTEND_PID_FILE"
  if ! wait_for_boot "Frontend" "$FRONTEND_PID_FILE" "$LOG_DIR/frontend.log" 6; then
    return 1
  fi
  echo "Started frontend (PID $(read_pid "$FRONTEND_PID_FILE"))"
}

stop_service() {
  local name="$1"
  local pid_file="$2"
  if is_running "$pid_file"; then
    local pid
    pid="$(read_pid "$pid_file")"
    kill_descendants "$pid"
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
    echo "$name: RUNNING (PID $(read_pid "$pid_file"))"
  else
    echo "$name: STOPPED"
  fi
}

show_logs() {
  echo "Trading:  $MAIN_LOG_FILE"
  echo "API:      $LOG_DIR/api.log"
  echo "Frontend: $LOG_DIR/frontend.log"
}

load_env

case "${1:-}" in
  start)
    ensure_prereqs
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
    ensure_prereqs
    "$0" stop
    "$0" start
    ;;
  status)
    status_service "trading engine" "$TRADING_PID_FILE"
    status_service "api" "$API_PID_FILE"
    status_service "frontend" "$FRONTEND_PID_FILE"
    ;;
  logs)
    show_logs
    ;;
  *)
    usage
    exit 1
    ;;
esac
