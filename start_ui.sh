#!/bin/bash
echo "🚀 Starting APEX SOTA Terminal..."

# Start Backend
echo "📈 Starting API Server (Port 8000)..."
python3 -m uvicorn api.server:app --reload --port 8000 &
BACKEND_PID=$!

# Start Frontend
echo "💻 Starting Client Interface (Port 3000)..."
cd frontend
npm run dev &
FRONTEND_PID=$!

# Handle shutdown
trap "kill $BACKEND_PID $FRONTEND_PID; exit" SIGINT SIGTERM

wait
