#!/usr/bin/env bash
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

HOST="$(python3 -c "import yaml;print(yaml.safe_load(open('config.yaml'))['web']['host'])" 2>/dev/null || echo 127.0.0.1)"
PORT="$(python3 -c "import yaml;print(yaml.safe_load(open('config.yaml'))['web']['port'])" 2>/dev/null || echo 8000)"

echo "Services"
for name in web scheduler; do
  pf="run/$name.pid"
  if [ -f "$pf" ] && kill -0 "$(cat "$pf")" 2>/dev/null; then
    printf "  %-10s up    (pid %s)\n" "$name" "$(cat "$pf")"
  else
    printf "  %-10s down\n" "$name"
  fi
done

echo
echo "Jobs"
python3 - <<'PY' 2>/dev/null || echo "  (database not initialised yet)"
import sqlite3, os
if not os.path.exists("jobs.db"):
    raise SystemExit(1)
c = sqlite3.connect("jobs.db"); c.row_factory = sqlite3.Row
rows = c.execute("SELECT status, COUNT(*) n FROM jobs GROUP BY status").fetchall()
if not rows:
    print("  none yet")
for r in rows:
    print(f"  {r['status']:<10} {r['n']}")
act = c.execute(
    "SELECT id, module, label, step_index, step_name, gpu FROM jobs "
    "WHERE status='running' ORDER BY id").fetchall()
if act:
    print()
    print("Active")
    for r in act:
        slot = f"gpu{r['gpu']}" if r['gpu'] is not None else "cpu"
        print(f"  {r['id']:05d}  {r['module']:<10} {slot:<5} "
              f"step {r['step_index']+1}: {r['step_name']}")
PY

echo
echo "Open the UI from your laptop with:"
echo "  ssh -N -L ${PORT}:localhost:${PORT} ${USER}@$(hostname -f 2>/dev/null || hostname)"
echo "  then browse to http://localhost:${PORT}"
