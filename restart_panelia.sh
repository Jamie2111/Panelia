#!/bin/zsh
set -euo pipefail

ROOT="/Users/jamieobala/Documents/Panelia"
BACKEND_LOG="$ROOT/backend/data/_service_logs/backend.log"
WORKER_LOG="$ROOT/backend/data/_service_logs/worker.log"
FRONTEND_LOG="$ROOT/frontend/.logs/frontend.log"

function kill_port_listeners() {
  local port="$1"
  local pids=""
  pids=$(lsof -tiTCP:"$port" -sTCP:LISTEN 2>/dev/null || true)
  if [ -n "$pids" ]; then
    for pid in ${(f)pids}; do
      kill "$pid" 2>/dev/null || true
    done
  fi
}

function kill_matching_processes() {
  local pattern="$1"
  pkill -f "$pattern" 2>/dev/null || true
}

function force_kill_matching_processes() {
  local pattern="$1"
  pkill -9 -f "$pattern" 2>/dev/null || true
}

function wait_for_processes_to_exit() {
  local pattern="$1"
  local attempts="${2:-10}"
  local delay="${3:-0.5}"
  local i
  for ((i=1; i<=attempts; i++)); do
    if ! pgrep -f "$pattern" >/dev/null 2>&1; then
      return 0
    fi
    sleep "$delay"
  done
  return 1
}

function wait_for_http() {
  local url="$1"
  local attempts="${2:-20}"
  local delay="${3:-1}"
  local i
  for ((i=1; i<=attempts; i++)); do
    if curl -fsSI "$url" >/dev/null 2>&1; then
      return 0
    fi
    sleep "$delay"
  done
  return 1
}

mkdir -p "$ROOT/backend/data/_service_logs"
mkdir -p "$ROOT/frontend/.logs"

echo "Stopping existing frontend/backend processes..."
kill_matching_processes 'next-server|next start|next dev'
kill_matching_processes 'uvicorn .*app.main:app'
kill_matching_processes 'python .* -m uvicorn app\.main:app|Python .* -m uvicorn app\.main:app|-m uvicorn app\.main:app'
kill_matching_processes "$ROOT/workers/worker.py"
kill_matching_processes 'workers/worker\.py'
kill_matching_processes 'python .* -m workers\.worker|Python .* -m workers\.worker|-m workers\.worker'
wait_for_processes_to_exit 'next-server|next start|next dev' || force_kill_matching_processes 'next-server|next start|next dev'
wait_for_processes_to_exit 'uvicorn .*app.main:app|-m uvicorn app\.main:app' || force_kill_matching_processes 'uvicorn .*app.main:app|-m uvicorn app\.main:app'
wait_for_processes_to_exit "$ROOT/workers/worker.py|workers/worker\.py|-m workers\.worker" || force_kill_matching_processes "$ROOT/workers/worker.py|workers/worker\.py|-m workers\.worker"
kill_port_listeners 3000
kill_port_listeners 8010
rm -f /tmp/panelia-worker.lock

echo "Starting backend..."
nohup zsh -lc "cd $ROOT/backend && source .venv/bin/activate && PYTHONPATH=$ROOT/backend python -u -m uvicorn app.main:app --host 127.0.0.1 --port 8010" > "$BACKEND_LOG" 2>&1 &

echo "Starting frontend..."
if [ ! -f "$ROOT/frontend/.next/BUILD_ID" ]; then
  echo "Frontend production build missing; building first..."
  (
    cd "$ROOT/frontend"
    npm run build
  )
fi
nohup zsh -lc "cd $ROOT/frontend && npm run start -- --hostname 127.0.0.1 --port 3000" > "$FRONTEND_LOG" 2>&1 &

echo "Starting worker..."
nohup zsh -lc "cd $ROOT && source $ROOT/backend/.venv/bin/activate && PYTHONPATH=$ROOT/backend PANELIA_WORKER_PREWARM=0 python -u -m workers.worker" >> "$WORKER_LOG" 2>&1 &

if ! wait_for_http "http://127.0.0.1:8010/docs" 20 1; then
  echo "Backend failed to start. Check $BACKEND_LOG"
  exit 1
fi

if ! wait_for_http "http://127.0.0.1:3000" 20 1; then
  echo "Frontend failed to start. Check $FRONTEND_LOG"
  exit 1
fi

LAYOUT_CHUNK=$(curl -fsS "http://127.0.0.1:3000/" | rg -o '/_next/static/chunks/app/layout-[^"]+\.js' -m 1 || true)
if [ -n "$LAYOUT_CHUNK" ] && ! curl -fsSI "http://127.0.0.1:3000$LAYOUT_CHUNK" >/dev/null 2>&1; then
  echo "Frontend started but is serving a missing app layout chunk. Check $FRONTEND_LOG"
  exit 1
fi

echo
echo "Panelia restart complete."
echo "Frontend: http://127.0.0.1:3000"
echo "Backend:  http://127.0.0.1:8010/docs"
echo
echo "Logs:"
echo "  Backend:  $BACKEND_LOG"
echo "  Worker:   $WORKER_LOG"
echo "  Frontend: $FRONTEND_LOG"
