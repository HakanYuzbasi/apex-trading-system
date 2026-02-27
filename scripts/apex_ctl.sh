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
  echo "Usage: $0 {start|stop|restart|status|logs|doctor}"
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

kill_orphan_matches() {
  local pattern="$1"
  local pids
  pids="$(pgrep -f "$pattern" 2>/dev/null || true)"
  if [[ -z "$pids" ]]; then
    return 0
  fi
  local pid
  for pid in $pids; do
    [[ "$pid" == "$$" ]] && continue
    kill "$pid" 2>/dev/null || true
  done
}

enforce_single_instance_stack() {
  # Clean up orphaned processes not tracked by PID files.
  # This avoids split-brain when users run `npm run dev` or `uvicorn` manually.
  kill_orphan_matches "python.*main.py"
  kill_orphan_matches "uvicorn api.server:app --port 8000"
  kill_orphan_matches "$BASE_DIR/frontend/node_modules/.bin/next dev"
  kill_orphan_matches "npm run dev"
  sleep 1
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

json_number_or_zero() {
  local input="$1"
  local expr="$2"
  if command -v jq >/dev/null 2>&1; then
    echo "$input" | jq -r "$expr // 0" 2>/dev/null || echo "0"
  else
    echo "0"
  fi
}

numbers_close() {
  local left="$1"
  local right="$2"
  local tolerance="${3:-0.01}"
  awk -v a="$left" -v b="$right" -v tol="$tolerance" '
    BEGIN {
      d = a - b
      if (d < 0) d = -d
      if (d <= tol) exit 0
      exit 1
    }
  '
}

process_count_for_pattern() {
  local pattern="$1"
  if command -v rg >/dev/null 2>&1; then
    ps -ax 2>/dev/null | rg "$pattern" | rg -v "rg |apex_ctl.sh doctor" | wc -l | tr -d ' '
  else
    ps -ax 2>/dev/null | grep -E "$pattern" | grep -v "grep -E" | grep -v "apex_ctl.sh doctor" | wc -l | tr -d ' '
  fi
}

warn_duplicate_processes() {
  local trading_count api_count next_count npm_count
  trading_count="$(process_count_for_pattern "python(.+ )?main\.py")"
  api_count="$(process_count_for_pattern "uvicorn api\.server:app --port 8000")"
  next_count="$(process_count_for_pattern "node .+next dev")"
  npm_count="$(process_count_for_pattern "npm run dev")"

  echo "Process duplicate scan:"
  echo "  trading(main.py): $trading_count"
  echo "  api(uvicorn:8000): $api_count"
  echo "  frontend(next dev): $next_count"
  echo "  frontend(npm run dev): $npm_count"

  local has_dup=0
  if [[ "$trading_count" -gt 1 || "$api_count" -gt 1 || "$next_count" -gt 1 || "$npm_count" -gt 1 ]]; then
    has_dup=1
  fi
  if [[ "$has_dup" -eq 1 ]]; then
    echo "WARNING: duplicate processes detected; values may diverge across instances."
    return 1
  fi
  echo "OK: no duplicate process patterns detected."
  return 0
}

doctor_endpoint_parity() {
  if ! command -v curl >/dev/null 2>&1; then
    echo "ERROR: curl is required for endpoint checks."
    return 1
  fi
  if ! command -v jq >/dev/null 2>&1; then
    echo "ERROR: jq is required for endpoint checks."
    return 1
  fi

  local pw token status_json balance_json cockpit_json metrics_json
  pw="${APEX_ADMIN_PASSWORD:-}"
  if [[ -z "$pw" && -f "$ENV_FILE" ]]; then
    pw="$(grep -E '^APEX_ADMIN_PASSWORD=' "$ENV_FILE" | tail -n1 | sed -E 's/^[^=]+=//' | sed -E 's/^"(.*)"$/\1/')"
  fi
  if [[ -z "$pw" ]]; then
    echo "ERROR: APEX_ADMIN_PASSWORD is not set; cannot run authenticated endpoint checks."
    return 1
  fi

  token="$(curl -sS --max-time 8 -X POST http://127.0.0.1:8000/auth/login \
    -H "content-type: application/json" \
    -d "{\"username\":\"admin\",\"password\":\"$pw\"}" | jq -r '.access_token // empty')"
  if [[ -z "$token" || "$token" == "null" ]]; then
    echo "ERROR: failed to authenticate against API."
    return 1
  fi

  status_json="$(curl -sS --max-time 8 http://127.0.0.1:8000/status -H "Authorization: Bearer $token" || true)"
  balance_json="$(curl -sS --max-time 8 http://127.0.0.1:8000/portfolio/balance -H "Authorization: Bearer $token" || true)"
  metrics_json="$(curl -sS --max-time 8 http://127.0.0.1:3000/api/v1/metrics -H "Cookie: token=$token" || true)"
  cockpit_json="$(curl -sS --max-time 12 http://127.0.0.1:3000/api/v1/cockpit -H "Cookie: token=$token" || true)"

  if [[ -z "$status_json" || -z "$balance_json" || -z "$cockpit_json" || -z "$metrics_json" ]]; then
    echo "ERROR: one or more endpoints did not return a payload."
    return 1
  fi

  local status_capital balance_capital metrics_capital cockpit_capital
  local status_open metrics_open cockpit_open cockpit_positions_len
  local status_primary_broker cockpit_active_broker
  status_capital="$(json_number_or_zero "$status_json" '.capital')"
  balance_capital="$(json_number_or_zero "$balance_json" '.total_equity')"
  metrics_capital="$(json_number_or_zero "$metrics_json" '.capital')"
  cockpit_capital="$(json_number_or_zero "$cockpit_json" '.status.capital')"

  status_open="$(json_number_or_zero "$status_json" '.open_positions')"
  metrics_open="$(json_number_or_zero "$metrics_json" '.open_positions')"
  cockpit_open="$(json_number_or_zero "$cockpit_json" '.status.open_positions')"
  cockpit_positions_len="$(json_number_or_zero "$cockpit_json" '.positions | length')"
  status_primary_broker="$(echo "$status_json" | jq -r '.primary_execution_broker // "unknown"' 2>/dev/null || echo "unknown")"
  cockpit_active_broker="$(echo "$cockpit_json" | jq -r '.status.active_broker // "unknown"' 2>/dev/null || echo "unknown")"

  echo "Endpoint parity scan:"
  echo "  /status.capital = $status_capital"
  echo "  /portfolio/balance.total_equity = $balance_capital"
  echo "  /api/v1/metrics.capital = $metrics_capital"
  echo "  /api/v1/cockpit.status.capital = $cockpit_capital"
  echo "  /status.open_positions = $status_open"
  echo "  /api/v1/metrics.open_positions = $metrics_open"
  echo "  /api/v1/cockpit.status.open_positions = $cockpit_open"
  echo "  /api/v1/cockpit.positions.length = $cockpit_positions_len"
  echo "  /status.primary_execution_broker = $status_primary_broker"
  echo "  /api/v1/cockpit.status.active_broker = $cockpit_active_broker"

  # Capital should align between frontend routes and portfolio balance.
  local cap_parity_ok=1
  if ! numbers_close "$balance_capital" "$metrics_capital" 1.0 || ! numbers_close "$balance_capital" "$cockpit_capital" 1.0; then
    cap_parity_ok=0
    echo "WARNING: capital mismatch between /portfolio/balance and frontend APIs."
  fi

  # Open positions should align between cockpit count and rows.
  local pos_parity_ok=1
  if [[ "$cockpit_open" != "$cockpit_positions_len" ]]; then
    pos_parity_ok=0
    echo "WARNING: cockpit open_positions differs from cockpit positions length."
  fi

  local broker_parity_ok=1
  if [[ "$status_primary_broker" != "unknown" && "$cockpit_active_broker" != "unknown" && "$cockpit_active_broker" != "multi" ]]; then
    if [[ "$status_primary_broker" != "$cockpit_active_broker" ]]; then
      broker_parity_ok=0
      echo "WARNING: active broker differs from primary_execution_broker."
    fi
  fi

  if [[ "$cap_parity_ok" -eq 1 && "$pos_parity_ok" -eq 1 && "$broker_parity_ok" -eq 1 ]]; then
    echo "OK: endpoint parity checks passed."
    return 0
  fi

  return 1
}

doctor() {
  local rc=0
  echo "=== APEX Doctor ==="
  warn_duplicate_processes || rc=1
  doctor_endpoint_parity || rc=1
  if [[ "$rc" -eq 0 ]]; then
    echo "Doctor result: PASS"
  else
    echo "Doctor result: FAIL"
  fi
  return "$rc"
}

load_env

case "${1:-}" in
  start)
    ensure_prereqs
    enforce_single_instance_stack
    start_trading
    start_api
    start_frontend
    echo "All services started."
    ;;
  stop)
    stop_service "frontend" "$FRONTEND_PID_FILE"
    stop_service "api" "$API_PID_FILE"
    stop_service "trading engine" "$TRADING_PID_FILE"
    enforce_single_instance_stack
    rm -f "$FRONTEND_PID_FILE" "$API_PID_FILE" "$TRADING_PID_FILE"
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
  doctor)
    doctor
    ;;
  *)
    usage
    exit 1
    ;;
esac
