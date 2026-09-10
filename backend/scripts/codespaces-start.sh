#!/usr/bin/env bash
# (Re)launch backend + frontend in background. Safe to run on every start.
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
pkill -f "uvicorn app.main:app" 2>/dev/null || true
pkill -f "vite.*--port 5173" 2>/dev/null || true

cd "$ROOT/backend" && nohup python -m uvicorn app.main:app --port 8000 >/tmp/backend.log 2>&1 &
cd "$ROOT/frontend" && nohup npm run dev -- --port 5173 --host >/tmp/frontend.log 2>&1 &

sleep 3
echo "backend :8000, frontend :5173 — make 5173 Public in the PORTS tab to share"
