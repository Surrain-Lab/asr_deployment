#!/usr/bin/env bash
# Start the web UI and the scheduler, both detached.
#
# Detached matters: this is normally launched over SSH, and neither process may
# die when that connection drops. setsid puts each in its own session, so they
# survive logout without needing systemd (no sudo here, and lingering is off).
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

CONDA="$(python3 -c "import yaml;print(yaml.safe_load(open('config.yaml'))['paths']['conda'])")"
ENV_NAME="${ASRBENCH_ENV:-asrbench}"
HOST="$(python3 -c "import yaml;print(yaml.safe_load(open('config.yaml'))['web']['host'])")"
PORT="$(python3 -c "import yaml;print(yaml.safe_load(open('config.yaml'))['web']['port'])")"

mkdir -p logs run locks

is_up() { [ -f "run/$1.pid" ] && kill -0 "$(cat "run/$1.pid")" 2>/dev/null; }

# start_one <name> <pgrep-pattern> -- <command...>
#
# The pattern must identify only this service. An over-broad pattern (matching
# the shared `conda run` prefix, say) makes starting one service kill the other.
start_one() {
  local name="$1" pattern="$2"; shift 3
  if is_up "$name"; then
    echo "  $name already running (pid $(cat "run/$name.pid"))"
    return
  fi
  # A stop that reached only the `conda run` wrapper can leave the real python
  # alive, holding the port and serving stale code. Clear that before starting.
  local stray
  stray="$(pgrep -u "$USER" -f "$pattern" 2>/dev/null | head -1 || true)"
  if [ -n "${stray:-}" ]; then
    echo "  $name: clearing stray process $stray"
    kill -TERM "$stray" 2>/dev/null || true
    sleep 1
  fi
  setsid "$@" >>"logs/$name.out" 2>&1 &
  echo $! > "run/$name.pid"
  echo "  $name started (pid $!)"
}

echo "ASR Benchmark - starting"
start_one scheduler 'asrbench[.]worker' -- \
  "$CONDA" run -n "$ENV_NAME" --no-capture-output python -m asrbench.worker
start_one web 'uvicorn asrbench[.]web' -- \
  "$CONDA" run -n "$ENV_NAME" --no-capture-output \
  python -m uvicorn asrbench.web.app:app --host "$HOST" --port "$PORT"

sleep 3
echo
"$ROOT/bin/status.sh"
