#!/bin/bash
# scripts/apex_ctl.sh - Unified Control for Apex V3
set -euo pipefail

BASE_DIR="$(cd "$(dirname "$0")/.." && pwd)"
PID_DIR="$BASE_DIR/.run"
mkdir -p "$PID_DIR"
mkdir -p "$BASE_DIR/logs"

FRONTEND_PID_FILE="$PID_DIR/apex_frontend.pid"

# Detect docker compose (v2 plugin vs legacy docker-compose)
if docker compose version &>/dev/null 2>&1; then
  DC="docker compose"
else
  DC="docker-compose"
fi

usage() {
  echo "Usage: $0 {start|stop|restart|status|logs [service]}"
  echo ""
  echo "  start          Build and start all services (backend + frontend)"
  echo "  stop           Stop all services"
  echo "  restart        Stop then start all services"
  echo "  status         Show status of all services"
  echo "  logs [svc]     Tail logs (default: apex trading engine)"
  echo ""
  echo "  Services: apex, apex-api, frontend, postgres, redis"
}

check_env() {
  echo "🔍 Checking environment..."
  if [ ! -f "$BASE_DIR/.env" ]; then
    echo "  ❌ .env file missing in $BASE_DIR"
    exit 1
  fi
  # Verify critical keys
  local keys=("POSTGRES_PASSWORD" "APEX_SECRET_KEY")
  for key in "${keys[@]}"; do
    if ! grep -q "^$key=" "$BASE_DIR/.env"; then
      echo "  ❌ Missing required key in .env: $key"
      exit 1
    fi
  done
  echo "  ✅ Environment valid."
}

run_migrations() {
  echo "🏗️  Running database migrations..."
  # Use the api container to run migrations
  if ! docker exec -it apex-api alembic upgrade head; then
    echo "  ⚠️  Standard migration failed. Attempting to 'stamp' existing database as current..."
    docker exec -it apex-api alembic stamp head || echo "  ❌ Migration recovery failed."
  fi
}

wait_healthy() {
  local name="$1"
  local retries=30
  echo -n "  Waiting for $name to be healthy..."
  for i in $(seq 1 $retries); do
    status=$(docker inspect --format='{{.State.Health.Status}}' "$name" 2>/dev/null || echo "none")
    if [ "$status" = "healthy" ]; then
      echo " ✅"
      return 0
    fi
    sleep 2
    echo -n "."
  done
  echo " ⚠️  (timed out — check: docker logs $name)"
  return 0  # non-fatal, let the user decide
}

start_all() {
  cd "$BASE_DIR"
  check_env

  echo "🚀 Starting APEX infrastructure (postgres, redis)..."
  $DC up -d postgres redis

  wait_healthy apex-postgres
  wait_healthy apex-redis

  echo "⚙️  Building and starting trading engine..."
  $DC up -d --build apex

  echo "🔌 Starting API server..."
  $DC up -d --build apex-api
  
  # Ensure DB is migrated after API is up but before traffic flows
  run_migrations

  # Frontend — run via Docker (consistent, no local node version issues)
  echo "🌐 Starting frontend..."
  $DC up -d --build frontend

  echo ""
  echo "✅ All services started."
  echo ""
  echo "  Trading engine  →  ws://localhost:8765"
  echo "  API             →  http://localhost:8000"
  echo "  Dashboard       →  http://localhost:3000"
  echo "  Grafana         →  http://localhost:3002  (admin / admin)"
  echo ""
  echo "  Run '$0 status' to check health."
  echo "  Run '$0 logs' to follow trading engine logs."
}

stop_all() {
  cd "$BASE_DIR"
  echo "🛑 Stopping all APEX services..."
  $DC stop

  # Kill any stale local node processes that may be holding port 3000
  # (can happen if a previous 'npm run dev' was left running)
  local port3000_pid
  port3000_pid=$(lsof -ti :3000 2>/dev/null || true)
  if [ -n "$port3000_pid" ]; then
    echo "  Releasing port 3000 (PID $port3000_pid)..."
    kill "$port3000_pid" 2>/dev/null || true
    sleep 1
  fi

  echo "✅ All services stopped."
}

status() {
  cd "$BASE_DIR"
  echo "=== APEX System Status ==="
  $DC ps
  echo ""
  echo "Reachability:"
  curl -sf http://localhost:8000/health > /dev/null 2>&1 \
    && echo "  API    http://localhost:8000  ✅" \
    || echo "  API    http://localhost:8000  ❌ (not reachable)"
  curl -sf http://localhost:3000 > /dev/null 2>&1 \
    && echo "  UI     http://localhost:3000  ✅" \
    || echo "  UI     http://localhost:3000  ❌ (not reachable)"
}

show_logs() {
  cd "$BASE_DIR"
  local svc="${2:-apex}"
  echo "📋 Tailing logs for: $svc  (Ctrl+C to exit)"
  $DC logs --tail 100 -f "$svc"
}

case "${1:-}" in
  start)
    start_all
    ;;
  stop)
    stop_all
    ;;
  restart)
    stop_all
    echo ""
    sleep 2
    start_all
    ;;
  status)
    status
    ;;
  logs)
    show_logs "$@"
    ;;
  *)
    usage
    exit 1
    ;;
esac
