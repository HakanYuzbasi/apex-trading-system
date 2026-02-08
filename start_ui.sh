#!/bin/bash
echo "🚀 Starting APEX SOTA Terminal..."
BASE_DIR="$(cd "$(dirname "$0")" && pwd)"

# Kill any existing processes on ports 8000 and 3000
echo "🧹 Cleaning up existing processes..."
lsof -ti:8000 | xargs kill 2>/dev/null || true
lsof -ti:3000 | xargs kill 2>/dev/null || true
lsof -ti:3001 | xargs kill 2>/dev/null || true
# Escalate if still running
lsof -ti:8000 | xargs kill -9 2>/dev/null || true
lsof -ti:3000 | xargs kill -9 2>/dev/null || true
lsof -ti:3001 | xargs kill -9 2>/dev/null || true
sleep 1

# Start Trading Engine
echo "🤖 Starting Trading Engine..."
cd "$BASE_DIR"
venv/bin/python main.py > /private/tmp/apex_main.log 2>&1 &
TRADING_PID=$!
echo "✓ Trading engine started (PID: $TRADING_PID)"

# Start Backend (API + WebSocket)
echo "📈 Starting API Server (Port 8000)..."
venv/bin/python -m uvicorn api.server:app --reload --port 8000 &
BACKEND_PID=$!
echo "✓ Backend started (PID: $BACKEND_PID)"
echo "   Running on http://localhost:8000"

# Wait for backend to be ready
sleep 2

# Start Frontend
echo "💻 Starting Client Interface (Port 3000)..."
cd "$BASE_DIR/frontend"
npm run dev &
FRONTEND_PID=$!
echo "✓ Frontend started (PID: $FRONTEND_PID)"
echo "   Running on http://localhost:3000"

sleep 2

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "✅ APEX SOTA Terminal is running!"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "📍 Frontend:  http://localhost:3000"
echo "📍 Backend:   http://localhost:8000"
echo ""
echo "Press Ctrl+C to stop both servers"
echo ""

# Cleanup function
cleanup() {
    echo ""
    echo "🛑 Stopping APEX servers..."
    
    # Kill trading engine
    if kill $TRADING_PID 2>/dev/null; then
        echo "✓ Trading engine stopped"
    fi

    # Kill backend
    if kill $BACKEND_PID 2>/dev/null; then
        echo "✓ Backend stopped"
    fi
    
    # Kill frontend
    if kill $FRONTEND_PID 2>/dev/null; then
        echo "✓ Frontend stopped"
    fi
    
    echo "✓ APEX servers stopped"
    exit 0
}

# Handle shutdown
trap cleanup SIGINT SIGTERM

wait
