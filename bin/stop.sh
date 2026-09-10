#!/usr/bin/env bash
# Stop the web UI and scheduler. Jobs already running are NOT touched - their
# runners are detached on purpose, so transcription in flight keeps going.
#
# We signal the process GROUP, not the pid: each service is launched through
# `conda run`, which spawns the real python as a child. Killing only the
# recorded pid leaves that child alive, still holding the port and still
# serving the old code.
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

for name in web scheduler; do
  pf="run/$name.pid"
  if [ ! -f "$pf" ]; then
    echo "  $name not running"
    continue
  fi
  pid="$(cat "$pf")"
  if kill -0 "$pid" 2>/dev/null; then
    kill -TERM -- "-$pid" 2>/dev/null || kill -TERM "$pid" 2>/dev/null
    for _ in 1 2 3 4 5 6 7 8 9 10; do
      kill -0 "$pid" 2>/dev/null || break
      sleep 0.3
    done
    kill -0 "$pid" 2>/dev/null && kill -KILL -- "-$pid" 2>/dev/null
    echo "  $name stopped"
  else
    echo "  $name not running (stale pid file)"
  fi
  rm -f "$pf"
done
echo
echo "Running jobs were left alone. bin/status.sh shows what is still active."
