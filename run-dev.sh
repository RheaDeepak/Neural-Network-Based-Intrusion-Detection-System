#!/usr/bin/env bash
set -e

ROOT_DIR="$(cd "$(dirname "$0")" && pwd)"

PY_BIN="$ROOT_DIR/.venv311/bin/python"
if [ ! -x "$PY_BIN" ]; then
  echo "Python venv not found at $PY_BIN. Create it with: python3.11 -m venv .venv311"
  exit 1
fi

echo "Starting backend on http://localhost:8000"
"$PY_BIN" -m uvicorn backend.app:app --host 0.0.0.0 --port 8000 &
BACKEND_PID=$!

echo "Starting frontend on http://localhost:5173"
cd "$ROOT_DIR/frontend"
npm run dev &
FRONTEND_PID=$!

cleanup() {
  echo "Stopping services..."
  kill $BACKEND_PID $FRONTEND_PID 2>/dev/null || true
}

trap cleanup EXIT

wait
